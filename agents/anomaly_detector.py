from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]

ANOMALY_RESULTS_PATH = BASE_DIR / "data" / "anomaly_results.csv"

# KES monthly household benchmarks (illustrative peer comparison).
PEER_BENCHMARKS_KES: dict[str, float] = {
    "Food": 18500.0,
    "Transport": 9500.0,
    "Utilities": 6200.0,
    "Entertainment": 4500.0,
}


def _resolve_category_column(df: pd.DataFrame) -> str:
    if "predicted_category" in df.columns:
        return "predicted_category"
    return "category"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week_num"] >= 5).astype(int)
    df["month_num"] = df["date"].dt.month
    df["hour_bucket"] = 12  # placeholder when hour not available

    le = LabelEncoder()
    category_col = _resolve_category_column(df)
    df["category_encoded"] = le.fit_transform(df[category_col])

    df["merchant_freq_global"] = df["merchant"].map(df["merchant"].value_counts())

    df = df.sort_values("date").reset_index(drop=True)
    df["rolling_7day_avg"] = (
        df["amount"].rolling(window=7, min_periods=1).mean().shift(1).fillna(df["amount"].median())
    )
    df["amount_to_avg_ratio"] = df["amount"] / (df["rolling_7day_avg"] + 1.0)

    amt = df["amount"].astype(float)
    med = amt.median()
    mad = (amt - med).abs().median()
    if mad == 0 or np.isnan(mad):
        mad = 1.0
    df["amount_z_score"] = 0.6745 * (amt - med) / mad

    return df


def _feature_columns(include_category: bool = True) -> list[str]:
    columns = [
        "amount",
        "amount_to_avg_ratio",
        "rolling_7day_avg",
        "day_of_week_num",
        "is_weekend",
        "month_num",
        "merchant_freq_global",
    ]
    if include_category:
        columns.append("category_encoded")
    return columns


def primary_statistical_detector(df_features: pd.DataFrame) -> pd.Series:
    """Robust z-score thresholding as primary anomaly signal."""
    z = df_features["amount_z_score"].abs()
    ratio = df_features["amount_to_avg_ratio"]
    is_prim = (z >= 3.5) | (ratio >= 8.0)
    return is_prim


def train_isolation_forest(df: pd.DataFrame) -> tuple[IsolationForest, list[str]]:
    feature_columns = _feature_columns(include_category=True)
    X = df[feature_columns].values
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)
    return model, feature_columns


def detect_anomalies_by_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_anomaly_by_category"] = False
    df["category_anomaly_score"] = 0.0

    category_col = _resolve_category_column(df)
    for category, group in df.groupby(category_col):
        if len(group) < 10:
            logger.warning(
                "Skipping category-level anomaly detection for %s (only %s rows)",
                category,
                len(group),
            )
            continue

        category_features = engineer_features(group)
        feature_columns = _feature_columns(include_category=False)
        X = category_features[feature_columns].values

        model = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
        model.fit(X)

        predictions = model.predict(X)
        scores = model.decision_function(X)

        df.loc[category_features.index, "is_anomaly_by_category"] = predictions == -1
        df.loc[category_features.index, "category_anomaly_score"] = (-scores).round(4)

    return df


def category_frequency_rarity(df_features: pd.DataFrame) -> pd.Series:
    freq = df_features["merchant_freq_global"].astype(float)
    return 1.0 / (1.0 + np.log1p(freq))


def composite_severity_score(df: pd.DataFrame) -> pd.Series:
    z = df["amount_z_score"].abs().clip(0, 12)
    ratio = df["amount_to_avg_ratio"].clip(0, 50)
    rarity = category_frequency_rarity(df)
    score = 0.4 * z + 0.3 * ratio + 0.3 * (rarity * 10.0)
    return score


def assign_severity_labels(severity_score: pd.Series) -> pd.Series:
    labels = pd.Series("NONE", index=severity_score.index, dtype=object)
    labels.loc[severity_score > 3.0] = "HIGH"
    labels.loc[(severity_score > 1.5) & (severity_score <= 3.0)] = "MEDIUM"
    labels.loc[(severity_score <= 1.5) & (severity_score > 0)] = "LOW"
    return labels


def contextual_flags(df_features: pd.DataFrame) -> pd.Series:
    """Flag transactions unusual for weekday/weekend pattern or rare merchants."""
    rare_merchant = df_features["merchant_freq_global"] <= 2
    weekend_spike = (df_features["is_weekend"] == 1) & (df_features["amount_to_avg_ratio"] >= 4.0)
    dow_spike = df_features["amount_to_avg_ratio"] >= 6.0
    return rare_merchant | weekend_spike | dow_spike


def deduplicate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Same merchant + amount + date → keep highest severity_score row."""
    if df.empty:
        return df

    df = df.sort_values("severity_score", ascending=False)
    subset_cols = ["merchant", "amount", "date"]
    return df.drop_duplicates(subset=subset_cols, keep="first")


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df_features = engineer_features(df)

    primary_flag = primary_statistical_detector(df_features)

    model, feature_columns = train_isolation_forest(df_features)
    X = df_features[feature_columns].values
    if_predictions = model.predict(X)
    if_scores = (-model.decision_function(X)).round(4)

    df_features["is_anomaly_primary"] = primary_flag
    df_features["is_anomaly_iforest"] = if_predictions == -1
    df_features["iforest_score"] = if_scores

    df_features["is_anomaly"] = df_features["is_anomaly_primary"] | df_features["is_anomaly_iforest"]

    contextual = contextual_flags(df_features)
    df_features.loc[contextual & (df_features["amount_z_score"].abs() >= 2.0), "is_anomaly"] = True

    df_features["severity_score"] = composite_severity_score(df_features).round(4)

    df_features["severity"] = "NONE"
    anom_mask = df_features["is_anomaly"]
    df_features.loc[anom_mask, "severity"] = assign_severity_labels(df_features.loc[anom_mask, "severity_score"])

    df_features = detect_anomalies_by_category(df_features)
    df_features.loc[df_features["is_anomaly_by_category"], "is_anomaly"] = True
    df_features.loc[df_features["is_anomaly_by_category"], "severity_score"] = np.maximum(
        df_features.loc[df_features["is_anomaly_by_category"], "severity_score"],
        1.6,
    )
    df_features.loc[df_features["is_anomaly_by_category"], "severity"] = assign_severity_labels(
        df_features.loc[df_features["is_anomaly_by_category"], "severity_score"]
    )

    df_features = add_anomaly_explanations(df_features)

    dup_anom = df_features[df_features["is_anomaly"]].copy()
    if not dup_anom.empty:
        kept_idx = deduplicate_anomalies(dup_anom).index
        drop_idx = dup_anom.index.difference(kept_idx)
        df_features.loc[drop_idx, "is_anomaly"] = False
        df_features.loc[drop_idx, "severity"] = "NONE"
        df_features.loc[drop_idx, "anomaly_explanation"] = ""

    df_features["anomaly_score"] = df_features[["severity_score", "iforest_score"]].max(axis=1).round(4)

    n_anomalies = int(df_features["is_anomaly"].sum())
    logger.info("Detected %s anomalies out of %s transactions", n_anomalies, len(df_features))
    return df_features


def _category_baselines(df: pd.DataFrame) -> pd.DataFrame:
    category_col = _resolve_category_column(df)
    return df.groupby(category_col)["amount"].agg(avg_amount="mean", median_amount="median", count="size").reset_index().rename(columns={category_col: "category"})


def add_anomaly_explanations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    baselines = _category_baselines(df)
    baseline_map = baselines.set_index("category")["avg_amount"].to_dict()

    explanations: list[str] = []
    for row in df.itertuples():
        if not row.is_anomaly:
            explanations.append("")
            continue

        avg_amount = baseline_map.get(row.category, 0) or 0
        ratio = row.amount / (avg_amount + 1.0)
        explanation = (
            f"This KES {row.amount:,.0f} transaction at {row.merchant} "
            f"is {ratio:,.1f}x higher than your average {row.category} spend "
            f"of KES {avg_amount:,.0f}."
        )
        explanations.append(explanation)

    df["anomaly_explanation"] = explanations
    return df


def inject_test_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    categories = ["Food", "Transport", "Entertainment", "Shopping"]
    fake_anomalies = []

    for i in range(n):
        fake_anomalies.append(
            {
                "date": pd.Timestamp("2026-11-01") + pd.Timedelta(days=i * 15),
                "amount": np.random.choice([85000, 120000, 95000, 110000, 75000]),
                "merchant": f"SUSPICIOUS_VENDOR_{i}",
                "category": np.random.choice(categories),
                "is_injected": True,
            }
        )

    fake_df = pd.DataFrame(fake_anomalies)
    df = df.copy()
    df["is_injected"] = False
    combined = pd.concat([df, fake_df], ignore_index=True)

    logger.info("Injected %s anomalies — total: %s transactions", n, len(combined))
    return combined


def evaluate_detector(df_with_results: pd.DataFrame) -> dict[str, float | int]:
    injected = df_with_results[df_with_results["is_injected"] == True]
    caught = injected[injected["is_anomaly"] == True]
    catch_rate = len(caught) / len(injected) if len(injected) > 0 else 0.0

    logger.info("Caught %s/%s — catch rate: %.1f%%", len(caught), len(injected), catch_rate * 100)

    return {
        "total_injected": len(injected),
        "caught": len(caught),
        "missed": len(injected) - len(caught),
        "catch_rate": catch_rate,
    }


def weekly_anomaly_report(df: pd.DataFrame, days: int = 7) -> dict[str, Any]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    window_start = latest_date - pd.Timedelta(days=days)

    recent = df[(df["date"] >= window_start) & (df["is_anomaly"])]
    total_amount = float(recent["amount"].sum()) if not recent.empty else 0.0

    return {
        "window_start": window_start.date().isoformat(),
        "window_end": latest_date.date().isoformat(),
        "anomaly_count": int(recent.shape[0]),
        "total_anomaly_amount": total_amount,
        "top_categories": recent["category"].value_counts().to_dict(),
    }


def anomaly_trend_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Week-over-week change in anomaly counts."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = df["date"].max()
    w1_start = end - pd.Timedelta(days=7)
    w2_start = end - pd.Timedelta(days=14)

    this_week = int(df[(df["date"] > w1_start) & (df["is_anomaly"])].shape[0])
    prev_week = int(df[(df["date"] > w2_start) & (df["date"] <= w1_start) & (df["is_anomaly"])].shape[0])

    delta = this_week - prev_week
    escalation = "increasing" if delta > 0 and this_week >= 3 else "stable"

    return {
        "this_week_count": this_week,
        "prev_week_count": prev_week,
        "delta": delta,
        "escalation": escalation,
    }


def merchant_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate merchant risk from historical anomaly severity."""
    if df.empty or "is_anomaly" not in df.columns:
        return pd.DataFrame(columns=["merchant", "risk_score", "anomaly_hits"])

    work = df[df["is_anomaly"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["merchant", "risk_score", "anomaly_hits"])

    sev_map = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}
    work["sev_w"] = work["severity"].map(sev_map).fillna(1.0)

    grp = work.groupby("merchant").agg(anomaly_hits=("is_anomaly", "count"), risk_score=("sev_w", "mean"))
    grp = grp.sort_values("risk_score", ascending=False).reset_index()
    grp["risk_score"] = grp["risk_score"].round(3)
    return grp


def peer_comparison(df: pd.DataFrame, benchmarks: dict[str, float] | None = None) -> list[dict[str, Any]]:
    benchmarks = benchmarks or PEER_BENCHMARKS_KES
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    category_col = _resolve_category_column(df)
    totals = df.groupby(category_col)["amount"].sum()
    out: list[dict[str, Any]] = []
    for cat, bench in benchmarks.items():
        spent = float(totals.get(cat, 0.0))
        out.append(
            {
                "category": cat,
                "user_total": round(spent, 2),
                "benchmark_kes": bench,
                "ratio_vs_peer": round(spent / bench, 3) if bench else None,
            }
        )
    return out


def plot_anomalies(df: pd.DataFrame, category: str | None = None, save_path: str | None = None) -> None:
    category_col = _resolve_category_column(df)
    plot_df = df[df[category_col] == category].copy() if category else df.copy()
    if category_col != "category":
        plot_df["category"] = plot_df[category_col]
    title = f"Anomaly Detection — {category}" if category else "Anomaly Detection — All"

    normal = plot_df[plot_df["is_anomaly"] == False]
    anomalies = plot_df[plot_df["is_anomaly"] == True]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(normal["date"], normal["amount"], c="steelblue", alpha=0.5, s=20, label=f"Normal ({len(normal)})")
    ax.scatter(
        anomalies["date"],
        anomalies["amount"],
        c="red",
        alpha=0.9,
        s=100,
        marker="x",
        linewidths=2,
        label=f"Anomaly ({len(anomalies)})",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount (KES)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved anomaly plot to %s", save_path)

    plt.close(fig)


def _save_results(df: pd.DataFrame, output_path: Path = ANOMALY_RESULTS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved anomaly results to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
    df = pd.read_csv(BASE_DIR / "data" / "classified_transactions.csv")
    if "predicted_category" in df.columns:
        df["category"] = df["predicted_category"].fillna(df.get("category"))

    print("\n" + "=" * 40)
    print("STEP 1: Injected anomaly test")
    print("=" * 40)
    df_with_fakes = inject_test_anomalies(df, n=10)
    df_results = detect_anomalies(df_with_fakes)
    eval_results = evaluate_detector(df_results)

    print(f"  Total injected : {eval_results['total_injected']}")
    print(f"  Caught         : {eval_results['caught']}")
    print(f"  Catch rate     : {eval_results['catch_rate']:.1%}")

    df_real = detect_anomalies(df)
    print(anomaly_trend_analysis(df_real))
    print(merchant_risk_score(df_real).head())
