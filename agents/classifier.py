from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional runtime dependency for API startup resilience
    pipeline = None

logger = logging.getLogger(__name__)

_MODEL_LOADED = False
classifier = None


def _ensure_classifier():
    """Lazy-load HF pipeline to avoid import-time failures in constrained environments."""
    global classifier, _MODEL_LOADED
    if _MODEL_LOADED:
        return classifier
    if pipeline is None:
        raise RuntimeError("transformers is not installed. Install project requirements to enable classification.")
    logger.info("Loading zero-shot classification model... (first run may download ~1.6GB)")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    _MODEL_LOADED = True
    logger.info("Model loaded successfully")
    return classifier


CATEGORIES = [
    "Food",
    "Transport",
    "Rent",
    "Entertainment",
    "Health",
    "Shopping",
    "Utilities",
    "Other",
]

CONFIDENCE_THRESHOLD = 0.6
UNCERTAIN_CATEGORY = "Uncertain"
UNCERTAIN_LOG_PATH = Path("data/uncertain_transactions.csv")

CLASSIFICATION_CACHE: dict[str, dict] = {}

# Merchant substring → expected category for secondary validation (KES / Kenya context).
MERCHANT_KEYWORD_MAP: dict[str, str] = {
    "naivas": "Food",
    "supermarket": "Food",
    "carrefour": "Food",
    "restaurant": "Food",
    "cafe": "Food",
    "java": "Food",
    "kfc": "Food",
    "uber": "Transport",
    "bolt": "Transport",
    "matatu": "Transport",
    "shell": "Transport",
    "fuel": "Transport",
    "total ": "Transport",
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "cinema": "Entertainment",
    "safaricom": "Utilities",
    "kplc": "Utilities",
    "nairobi water": "Utilities",
    "rent": "Rent",
    "landlord": "Rent",
    "pharmacy": "Health",
    "hospital": "Health",
    "clinic": "Health",
    "m-pesa": "Other",
}


def normalize_input_for_classification(text: str) -> str:
    """Strip, lowercase, remove special characters for classification calls."""
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_cache_key(merchant_name: str) -> str:
    return normalize_input_for_classification(merchant_name)


def keyword_fallback_classify(normalized_text: str) -> dict:
    """Rule-based classification when the primary model fails or as fallback."""
    for needle, cat in sorted(MERCHANT_KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        if needle in normalized_text:
            return {
                "category": cat,
                "confidence": 0.72,
                "raw_category": cat,
                "is_uncertain": False,
                "ambiguous": False,
                "needs_review": False,
                "method": "keyword_fallback",
            }
    return {
        "category": UNCERTAIN_CATEGORY,
        "confidence": 0.0,
        "raw_category": "Other",
        "is_uncertain": True,
        "ambiguous": True,
        "needs_review": True,
        "method": "keyword_fallback",
    }


def _keyword_map_verdict(normalized_text: str) -> str | None:
    for needle, cat in sorted(MERCHANT_KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        if needle in normalized_text:
            return cat
    return None


def _cross_check_keyword(hf_category: str, normalized_text: str) -> bool:
    """Returns True if keyword map disagrees with classifier → needs_review."""
    mapped = _keyword_map_verdict(normalized_text)
    if mapped is None:
        return False
    return mapped != hf_category


def classify_transaction(merchant_name: str) -> dict:
    """
    Classify a merchant string using normalization, optional HF model, ambiguity checks,
    keyword cross-check, cache, and keyword fallback on failure.
    """
    normalized = normalize_input_for_classification(merchant_name)
    cache_key = normalized or _normalize_cache_key(merchant_name)
    if cache_key in CLASSIFICATION_CACHE:
        logger.info("Cache hit for merchant: %s", merchant_name)
        return CLASSIFICATION_CACHE[cache_key]

    hf = _ensure_classifier()
    try:
        result = hf(normalized or merchant_name, CATEGORIES)
    except Exception as exc:
        logger.exception("Primary classifier failed; using keyword fallback: %s", exc)
        response = keyword_fallback_classify(normalized)
        response["error"] = str(exc)
        CLASSIFICATION_CACHE[cache_key] = response
        return response

    top_category = result["labels"][0]
    confidence_score = round(float(result["scores"][0]), 4)
    second_score = float(result["scores"][1]) if len(result["scores"]) > 1 else 0.0
    ambiguous = abs(confidence_score - second_score) < 0.1
    if ambiguous:
        confidence_score = round(max(0.0, confidence_score - 0.15), 4)

    is_uncertain = confidence_score < CONFIDENCE_THRESHOLD
    category = UNCERTAIN_CATEGORY if is_uncertain else top_category
    needs_review = _cross_check_keyword(top_category, normalized) or ambiguous

    response = {
        "category": category,
        "confidence": confidence_score,
        "raw_category": top_category,
        "is_uncertain": is_uncertain,
        "ambiguous": ambiguous,
        "needs_review": needs_review,
        "method": "hf",
        "keyword_map_verdict": _keyword_map_verdict(normalized),
    }

    CLASSIFICATION_CACHE[cache_key] = response
    return response


def classify_batch_with_retry(
    merchant_names: list[str],
    max_retries: int = 3,
    backoff_seconds: float = 0.5,
) -> list[dict]:
    """Classify merchants with up to max_retries and exponential backoff per item."""
    results: list[dict] = []
    for merchant_name in merchant_names:
        attempts = 0
        last_error: Exception | None = None
        while attempts <= max_retries:
            try:
                results.append(classify_transaction(merchant_name))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                wait_time = backoff_seconds * (2**attempts)
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
            fb = keyword_fallback_classify(normalize_input_for_classification(merchant_name))
            fb["error"] = str(last_error)
            results.append(fb)

    return results


def classify_with_context(merchant_name: str, recent_categories: list[str]) -> dict:
    """Bias ambiguous classifications using recent transaction category history."""
    base = classify_transaction(merchant_name)
    if not recent_categories:
        return base

    counts = Counter(recent_categories)
    majority_cat, majority_count = counts.most_common(1)[0]
    if base.get("ambiguous") or base.get("is_uncertain"):
        # If majority recent category matches raw HF top, boost confidence slightly.
        if majority_cat == base.get("raw_category"):
            adj = dict(base)
            adj["confidence"] = round(min(0.95, float(base["confidence"]) + 0.1), 4)
            adj["category"] = majority_cat if adj["confidence"] >= CONFIDENCE_THRESHOLD else UNCERTAIN_CATEGORY
            adj["is_uncertain"] = adj["confidence"] < CONFIDENCE_THRESHOLD
            adj["context_bias"] = majority_cat
            return adj
        if majority_count >= 2 and majority_cat in CATEGORIES:
            adj = dict(base)
            adj["context_suggestion"] = majority_cat
            adj["needs_review"] = True
            return adj
    return base


def feedback_loop(transaction: dict | None = None, correct_category: str | None = None, merchant_name: str | None = None) -> dict:
    """Update merchant cache with a verified label."""
    if transaction is not None:
        merchant = str(transaction.get("merchant", ""))
        cat = correct_category or str(transaction.get("category", ""))
    else:
        merchant = merchant_name or ""
        cat = correct_category or ""

    key = _normalize_cache_key(merchant)
    CLASSIFICATION_CACHE[key] = {
        "category": cat,
        "confidence": 0.99,
        "raw_category": cat,
        "is_uncertain": False,
        "ambiguous": False,
        "needs_review": False,
        "method": "feedback",
    }
    return {"updated": True, "cache_key": key, "cache_size": len(CLASSIFICATION_CACHE)}


def spending_velocity(
    df: pd.DataFrame,
    *,
    category_column: str = "predicted_category",
    date_column: str = "date",
    amount_column: str = "amount",
    recent_days: int = 14,
    prior_days: int = 14,
) -> pd.DataFrame:
    """Detect categories where spend rate in the recent window is >2x the prior window."""
    d = df.copy()
    d[date_column] = pd.to_datetime(d[date_column])
    end = d[date_column].max()
    recent_start = end - pd.Timedelta(days=recent_days)
    prior_start = recent_start - pd.Timedelta(days=prior_days)

    recent = d[d[date_column] > recent_start]
    prior = d[(d[date_column] > prior_start) & (d[date_column] <= recent_start)]

    def rate(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby(category_column)[amount_column].sum() / max(recent_days, 1)

    r = recent.groupby(category_column)[amount_column].sum() / max(recent_days, 1)
    p = prior.groupby(category_column)[amount_column].sum() / max(prior_days, 1)
    joined = pd.DataFrame({"recent_rate": r, "prior_rate": p}).fillna(0)
    joined["accelerated"] = joined["recent_rate"] > (2.0 * joined["prior_rate"].replace(0, 1e-9))
    joined = joined.reset_index().rename(columns={category_column: "category"})
    return joined


def compute_category_spending_stats(df: pd.DataFrame, category_column: str = "predicted_category") -> pd.DataFrame:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Logged %s uncertain transactions to %s", len(df), output_path)


def classify_all_transactions(input_path: str, output_path: str) -> pd.DataFrame:
    logger.info("Loading cleaned transactions from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Classifying %s transactions...", len(df))

    predicted_categories = []
    confidence_scores = []
    raw_categories = []
    is_uncertain_flags = []

    for i, row in enumerate(df.itertuples(), 1):
        result = classify_transaction(row.merchant)
        predicted_categories.append(result["category"])
        confidence_scores.append(result["confidence"])
        raw_categories.append(result.get("raw_category", result["category"]))
        is_uncertain_flags.append(result.get("is_uncertain", False))
        if i % 50 == 0:
            logger.info("Classified %s/%s transactions", i, len(df))

    df["predicted_category"] = predicted_categories
    df["confidence_score"] = confidence_scores
    df["raw_category"] = raw_categories
    df["is_uncertain"] = is_uncertain_flags

    df.to_csv(output_path, index=False)
    logger.info("Saved classified transactions to %s", output_path)

    uncertain_df = df[df["is_uncertain"]].copy()
    if len(uncertain_df) > 0:
        log_uncertain_transactions(uncertain_df)

    stats_df = compute_category_spending_stats(df)
    logger.info("Category spending statistics:\n%s", stats_df.to_string(index=False))

    return df


def evaluate_classifier(df: pd.DataFrame, sample_size: int = 50) -> dict:
    sample = df.sample(n=sample_size, random_state=42)
    correct = sample["predicted_category"] == sample["category"]
    accuracy = correct.sum() / sample_size
    logger.info("Accuracy on %s samples: %.2f%%", sample_size, accuracy * 100)
    wrong = sample[~correct][["merchant", "category", "predicted_category", "confidence_score"]]
    if len(wrong) > 0:
        logger.info("\nMisclassified transactions:\n%s", wrong.to_string())
    return {
        "accuracy": accuracy,
        "correct": int(correct.sum()),
        "total": sample_size,
        "misclassified": wrong.to_dict("records"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
    df = classify_all_transactions(
        input_path="data/cleaned_transactions.csv",
        output_path="data/classified_transactions.csv",
    )
    results = evaluate_classifier(df, sample_size=50)
    print("\n" + "=" * 40)
    print("CLASSIFIER EVALUATION RESULTS")
    print("=" * 40)
    print(f"Correct predictions : {results['correct']} / {results['total']}")
    print(f"Accuracy            : {results['accuracy']:.2%}")
