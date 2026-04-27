from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

ANOMALY_RESULTS_PATH = Path("data/anomaly_results.csv")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features, category encoding, and rolling statistics.
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week_num'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)
    df['month_num'] = df['date'].dt.month

    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    df = df.sort_values('date').reset_index(drop=True)
    df['rolling_7day_avg'] = (
        df['amount']
        .rolling(window=7, min_periods=1)
        .mean()
        .shift(1)
        .fillna(df['amount'].median())
    )
    df['amount_to_avg_ratio'] = df['amount'] / (df['rolling_7day_avg'] + 1)

    return df


def _feature_columns(include_category: bool = True) -> list:
    columns = [
        'amount',
        'amount_to_avg_ratio',
        'rolling_7day_avg',
        'day_of_week_num',
        'is_weekend',
        'month_num'
    ]
    if include_category:
        columns.append('category_encoded')
    return columns


def train_anomaly_detector(df: pd.DataFrame):
    feature_columns = _feature_columns(include_category=True)

    X = df[feature_columns].values

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)

    return model, feature_columns


def detect_anomalies_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs anomaly detection within each category to capture category-specific outliers.

    Returns:
        DataFrame with is_anomaly_by_category and category_anomaly_score columns.
    """

    df = df.copy()
    df['is_anomaly_by_category'] = False
    df['category_anomaly_score'] = 0.0

    for category, group in df.groupby('category'):
        if len(group) < 10:
            logger.warning("Skipping category-level anomaly detection for %s (only %s rows)", category, len(group))
            continue

        category_features = engineer_features(group)
        feature_columns = _feature_columns(include_category=False)
        X = category_features[feature_columns].values

        model = IsolationForest(
            n_estimators=100,
            contamination=0.08,
            random_state=42
        )
        model.fit(X)

        predictions = model.predict(X)
        scores = model.decision_function(X)

        df.loc[category_features.index, 'is_anomaly_by_category'] = predictions == -1
        df.loc[category_features.index, 'category_anomaly_score'] = (-scores).round(4)

    return df


def assign_severity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns LOW, MEDIUM, or HIGH severity to anomalies based on deviation magnitude.
    """

    df = df.copy()
    df['severity_score'] = (df['amount_to_avg_ratio'] * df['anomaly_score']).round(4)
    df['severity'] = "NONE"

    anomalies = df[df['is_anomaly']]
    if anomalies.empty:
        return df

    medium_threshold = anomalies['severity_score'].quantile(0.6)
    high_threshold = anomalies['severity_score'].quantile(0.85)

    df.loc[(df['is_anomaly']) & (df['severity_score'] >= high_threshold), 'severity'] = 'HIGH'
    df.loc[(df['is_anomaly']) & (df['severity_score'] >= medium_threshold) & (df['severity_score'] < high_threshold), 'severity'] = 'MEDIUM'
    df.loc[(df['is_anomaly']) & (df['severity_score'] < medium_threshold), 'severity'] = 'LOW'

    return df


def _category_baselines(df: pd.DataFrame) -> pd.DataFrame:
    baselines = (
        df.groupby('category')['amount']
        .agg(avg_amount='mean', median_amount='median', count='size')
        .reset_index()
    )
    return baselines


def add_anomaly_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates natural language explanations for anomalies.
    """

    df = df.copy()
    baselines = _category_baselines(df)
    baseline_map = baselines.set_index('category')['avg_amount'].to_dict()

    explanations = []
    for row in df.itertuples():
        if not row.is_anomaly:
            explanations.append("")
            continue

        avg_amount = baseline_map.get(row.category, 0) or 0
        ratio = row.amount / (avg_amount + 1)
        explanation = (
            f"This KES {row.amount:,.0f} transaction at {row.merchant} "
            f"is {ratio:,.1f}x higher than your average {row.category} spend "
            f"of KES {avg_amount:,.0f}."
        )
        explanations.append(explanation)

    df['anomaly_explanation'] = explanations
    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects anomalies using Isolation Forest and adds severity scores and explanations.
    """

    df_features = engineer_features(df)
    model, feature_columns = train_anomaly_detector(df_features)

    X = df_features[feature_columns].values
    predictions = model.predict(X)
    df_features['is_anomaly'] = predictions == -1

    anomaly_scores = model.decision_function(X)
    df_features['anomaly_score'] = (-anomaly_scores).round(4)

    df_features = detect_anomalies_by_category(df_features)
    df_features = assign_severity_scores(df_features)
    df_features = add_anomaly_explanations(df_features)

    n_anomalies = df_features['is_anomaly'].sum()
    logger.info("Detected %s anomalies out of %s transactions", n_anomalies, len(df_features))

    return df_features


def inject_test_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    categories = ['Food', 'Transport', 'Entertainment', 'Shopping']
    fake_anomalies = []

    for i in range(n):
        fake_anomalies.append({
            'date': pd.Timestamp('2025-11-01') + pd.Timedelta(days=i * 15),
            'amount': np.random.choice([85000, 120000, 95000, 110000, 75000]),
            'merchant': f'SUSPICIOUS_VENDOR_{i}',
            'category': np.random.choice(categories),
            'is_injected': True
        })

    fake_df = pd.DataFrame(fake_anomalies)
    df = df.copy()
    df['is_injected'] = False
    combined = pd.concat([df, fake_df], ignore_index=True)

    logger.info("Injected %s anomalies - total: %s transactions", n, len(combined))
    return combined


def evaluate_detector(df_with_results: pd.DataFrame) -> dict:
    injected = df_with_results[df_with_results['is_injected'] == True]
    caught = injected[injected['is_anomaly'] == True]
    catch_rate = len(caught) / len(injected) if len(injected) > 0 else 0

    logger.info("Caught %s/%s - catch rate: %.1f%%", len(caught), len(injected), catch_rate * 100)

    return {
        'total_injected': len(injected),
        'caught': len(caught),
        'missed': len(injected) - len(caught),
        'catch_rate': catch_rate
    }


def weekly_anomaly_report(df: pd.DataFrame, days: int = 7) -> dict:
    """
    Summarizes anomalies from the most recent N days.
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    window_start = latest_date - pd.Timedelta(days=days)

    recent = df[(df['date'] >= window_start) & (df['is_anomaly'])]
    total_amount = recent['amount'].sum() if not recent.empty else 0

    report = {
        'window_start': window_start.date().isoformat(),
        'window_end': latest_date.date().isoformat(),
        'anomaly_count': int(recent.shape[0]),
        'total_anomaly_amount': float(total_amount),
        'top_categories': recent['category'].value_counts().to_dict(),
    }

    logger.info("Weekly anomaly report: %s", report)
    return report


def plot_anomalies(df: pd.DataFrame, category: str = None, save_path: str | None = None):
    plot_df = df[df['category'] == category].copy() if category else df.copy()
    title = f'Anomaly Detection - {category}' if category else 'Anomaly Detection - All'

    normal = plot_df[plot_df['is_anomaly'] == False]
    anomalies = plot_df[plot_df['is_anomaly'] == True]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(normal['date'], normal['amount'], c='steelblue', alpha=0.5, s=20, label=f'Normal ({len(normal)})')
    ax.scatter(anomalies['date'], anomalies['amount'], c='red', alpha=0.9, s=100, marker='x', linewidths=2, label=f'Anomaly ({len(anomalies)})')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount (KES)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved anomaly plot to %s", save_path)

    plt.close(fig)


def _save_results(df: pd.DataFrame, output_path: Path = ANOMALY_RESULTS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved anomaly results to %s", output_path)


if __name__ == "__main__":
    df = pd.read_csv("data/classified_transactions.csv")

    print("\n" + "=" * 40)
    print("STEP 1: Injected anomaly test")
    print("=" * 40)
    df_with_fakes = inject_test_anomalies(df, n=10)
    df_results = detect_anomalies(df_with_fakes)
    eval_results = evaluate_detector(df_results)

    print(f"  Total injected : {eval_results['total_injected']}")
    print(f"  Caught         : {eval_results['caught']}")
    print(f"  Missed         : {eval_results['missed']}")
    print(f"  Catch rate     : {eval_results['catch_rate']:.1%}")

    print("\n" + "=" * 40)
    print("STEP 2: Real data anomalies")
    print("=" * 40)
    df_real = detect_anomalies(df)

    top_anomalies = (
        df_real[df_real['is_anomaly'] == True]
        .sort_values('anomaly_score', ascending=False)
        .head(10)
    )[['date', 'merchant', 'category', 'amount', 'anomaly_score', 'severity']]

    print(top_anomalies.to_string(index=False))

    weekly_report = weekly_anomaly_report(df_real)
    print("\nWeekly anomaly report:")
    print(weekly_report)

    plot_anomalies(df_real, category='Food', save_path="data/anomaly_food.png")
    plot_anomalies(df_real, save_path="data/anomaly_all.png")

    _save_results(df_real)
