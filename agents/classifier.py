from pathlib import Path
import time

import pandas as pd
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

logger.info("Loading zero-shot classification model... (first run downloads ~1.6GB)")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
logger.info("Model loaded successfully")

CATEGORIES = [
    "Food",          # restaurants, supermarkets, cafes
    "Transport",     # uber, fuel, matatu, flights
    "Rent",          # monthly rent payments
    "Entertainment", # netflix, cinema, spotify, games
    "Health",        # pharmacy, clinic, hospital
    "Shopping",      # clothes, electronics, online shopping
    "Utilities",     # electricity, water, internet, phone bills
    "Other"          # anything that doesn't fit above
]

CONFIDENCE_THRESHOLD = 0.6
UNCERTAIN_CATEGORY = "Uncertain"
UNCERTAIN_LOG_PATH = Path("data/uncertain_transactions.csv")

CLASSIFICATION_CACHE: dict[str, dict] = {}


def _normalize_merchant(merchant_name: str) -> str:
    return str(merchant_name).strip().lower()


def classify_transaction(merchant_name: str) -> dict:
    """
    Takes a merchant name and returns the most likely spending category along with a confidence score.

    Args:
        merchant_name (str): The name of the merchant from the transaction record.

    Returns:
        dict: A dictionary containing the predicted category and its confidence score.
    """

    cache_key = _normalize_merchant(merchant_name)
    if cache_key in CLASSIFICATION_CACHE:
        logger.info(f"Cache hit for merchant: {merchant_name}")
        return CLASSIFICATION_CACHE[cache_key]

    result = classifier(merchant_name, CATEGORIES)

    top_category = result['labels'][0]
    confidence_score = round(result['scores'][0], 4)

    is_uncertain = confidence_score < CONFIDENCE_THRESHOLD
    category = UNCERTAIN_CATEGORY if is_uncertain else top_category

    if is_uncertain:
        logger.warning(
            "Low confidence for merchant '%s': %.4f (predicted: %s)",
            merchant_name,
            confidence_score,
            top_category,
        )

    response = {
        "category": category,
        "confidence": confidence_score,
        "raw_category": top_category,
        "is_uncertain": is_uncertain,
    }

    CLASSIFICATION_CACHE[cache_key] = response
    return response


def classify_batch_with_retry(merchant_names: list, max_retries: int = 3, backoff_seconds: float = 0.5) -> list:
    """
    Classifies a batch of merchant names with retry logic and exponential backoff.

    Args:
        merchant_names: list of merchant names to classify.
        max_retries: how many retries to attempt for failures.
        backoff_seconds: base backoff seconds for exponential retry.

    Returns:
        list of classification result dicts.
    """

    results = []
    for merchant_name in merchant_names:
        attempts = 0
        last_error = None
        while attempts <= max_retries:
            try:
                results.append(classify_transaction(merchant_name))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                wait_time = backoff_seconds * (2 ** attempts)
                logger.warning(
                    "Classification failed for '%s' (attempt %s/%s). Retrying in %.2fs",
                    merchant_name,
                    attempts + 1,
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
                attempts += 1

        if last_error is not None:
            logger.error("Classification failed permanently for '%s': %s", merchant_name, last_error)
            results.append({
                "category": UNCERTAIN_CATEGORY,
                "confidence": 0.0,
                "raw_category": "",
                "is_uncertain": True,
                "error": str(last_error),
            })

    return results


def compute_category_spending_stats(df: pd.DataFrame, category_column: str = "predicted_category") -> pd.DataFrame:
    """
    Computes per-category spend totals, percent of total spend, and average transaction size.

    Args:
        df: classified DataFrame containing amount and category columns.
        category_column: column name to group on.

    Returns:
        DataFrame with total_spend, percent_of_total, avg_transaction, and transaction_count.
    """

    grouped = (
        df.groupby(category_column)["amount"]
        .agg(total_spend="sum", avg_transaction="mean", transaction_count="size")
        .reset_index()
    )

    total_spend = grouped["total_spend"].sum() or 1.0
    grouped["percent_of_total"] = (grouped["total_spend"] / total_spend) * 100

    grouped["total_spend"] = grouped["total_spend"].round(2)
    grouped["avg_transaction"] = grouped["avg_transaction"].round(2)
    grouped["percent_of_total"] = grouped["percent_of_total"].round(2)

    return grouped.sort_values("total_spend", ascending=False).reset_index(drop=True)


def log_uncertain_transactions(df: pd.DataFrame, output_path: Path = UNCERTAIN_LOG_PATH) -> None:
    """
    Logs uncertain transactions to a CSV file for manual review.

    Args:
        df: DataFrame containing uncertain transactions.
        output_path: path to write the review file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Logged %s uncertain transactions to %s", len(df), output_path)


def classify_all_transactions(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Loads the cleaned transactions, classifies each transaction, and saves the results to a new CSV file.

    Args:
        input_path (str): Path to the cleaned transactions CSV file.
        output_path (str): Path where the classified transactions CSV will be saved.
    Returns:
        pd.DataFrame: DataFrame containing the classified transactions.

    """

    logger.info(f"Loading cleaned transactions from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Classifying {len(df)} transactions...")

    predicted_categories = []
    confidence_scores = []
    raw_categories = []
    is_uncertain_flags = []

    for i, row in enumerate(df.itertuples(), 1):
        result = classify_transaction(row.merchant)
        predicted_categories.append(result['category'])
        confidence_scores.append(result['confidence'])
        raw_categories.append(result.get('raw_category', result['category']))
        is_uncertain_flags.append(result.get('is_uncertain', False))
        if i % 50 == 0:
            logger.info(f"Classified {i}/{len(df)} transactions")

    df['predicted_category'] = predicted_categories
    df['confidence_score'] = confidence_scores
    df['raw_category'] = raw_categories
    df['is_uncertain'] = is_uncertain_flags

    df.to_csv(output_path, index=False)
    logger.info(f"Saved classified transactions to {output_path}")

    uncertain_df = df[df['is_uncertain']].copy()
    if len(uncertain_df) > 0:
        log_uncertain_transactions(uncertain_df)

    stats_df = compute_category_spending_stats(df)
    logger.info("Category spending statistics:\n%s", stats_df.to_string(index=False))

    return df


def evaluate_classifier(df: pd.DataFrame, sample_size: int = 50) -> dict:
    """
    Takes a random sample of transactions, compares the AI's predicted
    category to the original category in our data, and computes accuracy.

    Note: Our synthetic data already has a 'category' column from when
    we generated it with Faker - we use that as our 'ground truth' to
    measure how well the AI is doing.

    Args:
        df: classified DataFrame (must have 'category' and 'predicted_category')
        sample_size: how many rows to evaluate (50 is enough for a quick check)

    Returns:
        dictionary with accuracy score and a breakdown
    """

    sample = df.sample(n=sample_size, random_state=42)

    correct = sample['predicted_category'] == sample['category']
    accuracy = correct.sum() / sample_size

    logger.info(f"Accuracy on {sample_size} samples: {accuracy:.2%}")

    wrong = sample[~correct][['merchant', 'category', 'predicted_category', 'confidence_score']]

    if len(wrong) > 0:
        logger.info(f"\nMisclassified transactions:\n{wrong.to_string()}")

    return {
        'accuracy': accuracy,
        'correct': int(correct.sum()),
        'total': sample_size,
        'misclassified': wrong.to_dict('records')
    }


if __name__ == "__main__":
    df = classify_all_transactions(
        input_path="data/cleaned_transactions.csv",
        output_path="data/classified_transactions.csv"
    )

    results = evaluate_classifier(df, sample_size=50)

    print("\n" + "=" * 40)
    print("CLASSIFIER EVALUATION RESULTS")
    print("=" * 40)
    print(f"Correct predictions : {results['correct']} / {results['total']}")
    print(f"Accuracy            : {results['accuracy']:.2%}")
    print(f"Target              : 80%+")

    print("\nSample predictions:")
    print(df[['merchant', 'category', 'predicted_category', 'confidence_score']].head(15).to_string())
