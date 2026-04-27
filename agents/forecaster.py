from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    seasonal_decompose = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CLASSIFIED_PATH = DATA_DIR / "classified_transactions.csv"
FORECASTS_DIR = DATA_DIR / "forecasts"


def ewma_daily_spend(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    """Exponentially weighted moving average; recent observations weighted more heavily."""
    return series.ewm(alpha=alpha, adjust=False).mean()


def prepare_time_series(df: pd.DataFrame, category: str, amount_col: str = "amount") -> pd.DataFrame:
    category_df = df[df["category"] == category].copy()
    category_df["date"] = pd.to_datetime(category_df["date"])
    daily_spend = category_df.groupby("date")[amount_col].sum().reset_index()
    daily_spend = daily_spend.rename(columns={"date": "ds", amount_col: "y"})
    daily_spend = daily_spend.sort_values("ds").reset_index(drop=True)
    logger.info("Prepared %s daily data points for category: %s", len(daily_spend), category)
    return daily_spend


def seasonal_decomposition_table(time_series: pd.DataFrame, period: int = 7) -> dict[str, Any]:
    """
    Decompose daily spend into trend + weekly seasonality + residual when enough history exists.
    """
    if seasonal_decompose is None:
        logger.warning("statsmodels seasonal_decompose unavailable; skipping decomposition")
        return {"available": False, "reason": "statsmodels_missing"}

    if len(time_series) < 2 * period:
        logger.warning(
            "Not enough points for decomposition (%s < %s)",
            len(time_series),
            2 * period,
        )
        return {"available": False, "reason": "insufficient_history"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decomp = seasonal_decompose(
            time_series.set_index("ds")["y"],
            model="additive",
            period=period,
            extrapolate_trend="freq",
        )
    return {
        "available": True,
        "trend": decomp.trend.dropna().tolist(),
        "seasonal": decomp.seasonal.dropna().tolist(),
        "resid": decomp.resid.dropna().tolist(),
    }


def extract_changepoints(model: Prophet) -> list[str]:
    if not hasattr(model, "changepoints"):
        return []
    return [pd.Timestamp(cp).date().isoformat() for cp in model.changepoints]


def calculate_forecast_mae(time_series: pd.DataFrame, forecast: pd.DataFrame, days: int = 7) -> float:
    recent_actuals = time_series.tail(days)
    merged = recent_actuals.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        logger.warning("No overlap between actuals and forecast for MAE calculation")
        return float("nan")
    mae = (merged["y"] - merged["yhat"]).abs().mean()
    return round(float(mae), 2)


def forecast_accuracy_by_category(
    time_series: pd.DataFrame,
    forecast: pd.DataFrame,
    days: int = 30,
) -> dict[str, float | None]:
    """MAE, RMSE, MAPE over overlapping history window."""
    merged = time_series.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        return {"mae": None, "rmse": None, "mape": None}
    tail = merged.tail(days)
    err = tail["y"] - tail["yhat"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err**2).mean()))
    denom = tail["y"].replace(0, np.nan)
    mape = float((err.abs() / denom).mean() * 100) if denom.notna().any() else None
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2) if mape is not None else None}


def data_quality_flags(time_series: pd.DataFrame) -> dict[str, Any]:
    """Flag sparse coverage: many zero-spend days in the daily series."""
    if time_series.empty:
        return {"zero_day_ratio": 1.0, "flag_sparse": True}

    full_range = pd.date_range(time_series["ds"].min(), time_series["ds"].max(), freq="D")
    daily_map = time_series.set_index("ds")["y"].reindex(full_range, fill_value=0.0)
    zero_days = float((daily_map <= 0).mean())
    return {
        "zero_day_ratio": round(zero_days, 4),
        "flag_sparse": zero_days > 0.30,
    }


def detect_recurring_charges(
    df: pd.DataFrame,
    *,
    category: str | None = None,
    tolerance_pct: float = 5.0,
) -> list[dict[str, Any]]:
    """Find merchants with repeating amounts (~±tolerance_pct) on similar calendar days."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    if category:
        work = work[work["category"] == category]
    recurring: list[dict[str, Any]] = []
    for merchant, g in work.groupby("merchant"):
        if len(g) < 3:
            continue
        amounts = g["amount"].values
        dom = g["date"].dt.day.values
        mean_amt = float(np.mean(amounts))
        if mean_amt <= 0:
            continue
        mask = np.abs(amounts - mean_amt) / mean_amt <= (tolerance_pct / 100.0)
        if mask.sum() >= 3 and float(np.std(dom)) <= 4.0:
            recurring.append(
                {
                    "merchant": merchant,
                    "approx_amount": round(mean_amt, 2),
                    "hits": int(mask.sum()),
                    "category": str(g["category"].iloc[0]),
                }
            )
    return recurring


def savings_projection(
    forecast_total_spend: float,
    monthly_income_estimate: float,
    *,
    horizon_days: int = 30,
) -> dict[str, float]:
    """Rough end-of-period savings estimate from income minus forecast spend."""
    daily_income = monthly_income_estimate / 30.0
    projected_income_period = daily_income * horizon_days
    projected_savings = projected_income_period - forecast_total_spend
    return {
        "projected_income_period": round(projected_income_period, 2),
        "projected_savings": round(projected_savings, 2),
        "horizon_days": float(horizon_days),
    }


def what_if_forecast(
    base_forecast_df: pd.DataFrame,
    change_pct: float,
    *,
    category_label: str,
) -> dict[str, Any]:
    """Apply a hypothetical % change to forecasted daily spend."""
    adj = base_forecast_df.copy()
    factor = 1.0 + (change_pct / 100.0)
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        if col in adj.columns:
            adj[col] = adj[col] * factor
    future = adj.tail(30)
    delta_total = float(future["yhat"].sum()) - float(base_forecast_df.tail(30)["yhat"].sum())
    return {
        "category": category_label,
        "change_pct": change_pct,
        "adjusted_total_30d": round(float(future["yhat"].sum()), 2),
        "savings_impact_vs_base": round(-delta_total, 2),
        "forecast_frame": adj,
    }


def forecast_spending(
    df: pd.DataFrame,
    category: str,
    days: int = 30,
    *,
    global_prior_daily: float | None = None,
) -> tuple[pd.DataFrame | None, Prophet | None]:
    """
    Train Prophet with weekly seasonality and 95% intervals.
    For sparse categories (<7 daily points), blend with global prior and continue without crashing.
    """
    logger.info("Training forecast model for category: %s", category)
    time_series = prepare_time_series(df, category)

    _ = seasonal_decomposition_table(time_series)

    if global_prior_daily is None:
        all_daily = df.copy()
        all_daily["date"] = pd.to_datetime(all_daily["date"])
        glob = all_daily.groupby("date")["amount"].sum()
        global_prior_daily = float(glob.mean()) if len(glob) else 1.0

    if len(time_series) < 7:
        logger.warning(
            "Category %s has <%s daily points (%s); using global average prior.",
            category,
            7,
            len(time_series),
        )
        pad_days = max(0, 7 - len(time_series))
        last_ds = time_series["ds"].max() if len(time_series) else pd.Timestamp.today()
        extra_dates = pd.date_range(last_ds + pd.Timedelta(days=1), periods=pad_days, freq="D")
        pad = pd.DataFrame({"ds": extra_dates, "y": global_prior_daily})
        time_series = pd.concat([time_series, pad], ignore_index=True).sort_values("ds")

    time_series["y_ewma"] = ewma_daily_spend(time_series["y"])

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(time_series[["ds", "y"]])

    future_dates = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future_dates)

    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    mae_7d = calculate_forecast_mae(time_series, forecast, days=7)
    forecast.attrs["mae_7d"] = mae_7d
    forecast.attrs["changepoints"] = extract_changepoints(model)
    forecast.attrs["accuracy"] = forecast_accuracy_by_category(time_series, forecast)
    forecast.attrs["data_quality"] = data_quality_flags(time_series)

    logger.info("Forecast complete for %s — next %s days predicted", category, days)
    return forecast, model


def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame, category: str, save_path: str | None = None) -> None:
    actual = prepare_time_series(df, category)
    future_forecast = forecast.tail(30)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(actual["ds"], actual["y"], color="black", s=20, label="Actual spend", zorder=3)
    ax.plot(future_forecast["ds"], future_forecast["yhat"], color="steelblue", linewidth=2, label="Forecast")
    ax.fill_between(
        future_forecast["ds"],
        future_forecast["yhat_lower"],
        future_forecast["yhat_upper"],
        alpha=0.3,
        color="steelblue",
        label="95% confidence interval",
    )
    ax.set_title(f"30-Day Spending Forecast — {category}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Spend (KES)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Chart saved to %s", save_path)
    plt.close(fig)


def get_forecast_summary(forecast: pd.DataFrame, category: str) -> dict[str, Any]:
    future = forecast.tail(30)
    acc = forecast.attrs.get("accuracy") or {}
    dq = forecast.attrs.get("data_quality") or {}

    return {
        "category": category,
        "forecast_days": 30,
        "total_predicted_spend": round(float(future["yhat"].sum()), 2),
        "avg_daily_spend": round(float(future["yhat"].mean()), 2),
        "max_daily_spend": round(float(future["yhat"].max()), 2),
        "min_daily_spend": round(float(future["yhat"].min()), 2),
        "lower_bound_total": round(float(future["yhat_lower"].sum()), 2),
        "upper_bound_total": round(float(future["yhat_upper"].sum()), 2),
        "lower_bound": round(float(future["yhat_lower"].mean()), 2),
        "upper_bound": round(float(future["yhat_upper"].mean()), 2),
        "mae_7d": forecast.attrs.get("mae_7d"),
        "changepoints": forecast.attrs.get("changepoints", []),
        "mae": acc.get("mae"),
        "rmse": acc.get("rmse"),
        "mape": acc.get("mape"),
        "data_quality": dq,
    }


def combine_forecast_summaries(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    if not summaries:
        return pd.DataFrame()
    return pd.DataFrame(summaries)


def check_budget_alerts(summary_df: pd.DataFrame, budget_limits: dict[str, float]) -> pd.DataFrame:
    if summary_df.empty or not budget_limits:
        return pd.DataFrame()

    alerts = summary_df.copy()
    alerts["budget_limit"] = alerts["category"].map(budget_limits)
    alerts = alerts.dropna(subset=["budget_limit"]).copy()
    alerts["over_budget"] = alerts["total_predicted_spend"] > alerts["budget_limit"]
    alerts = alerts[alerts["over_budget"]]

    if not alerts.empty:
        logger.warning("Budget alerts:\n%s", alerts.to_string(index=False))
    return alerts


def _load_budget_limits() -> dict[str, float]:
    budget_env = os.getenv("FINSIGHT_BUDGETS")
    if not budget_env:
        return {}
    try:
        budgets = json.loads(budget_env)
        if isinstance(budgets, dict):
            return {str(k): float(v) for k, v in budgets.items()}
        logger.warning("FINSIGHT_BUDGETS must be a JSON object")
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Failed to parse FINSIGHT_BUDGETS JSON")
    return {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
    logger.info("Loading classified transactions...")
    if not CLASSIFIED_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {CLASSIFIED_PATH}")

    df = pd.read_csv(CLASSIFIED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    categories_to_forecast = ["Food", "Transport", "Entertainment", "Shopping"]
    all_daily_prior = df.groupby(pd.to_datetime(df["date"]).dt.date)["amount"].sum().mean()

    all_summaries: list[dict[str, Any]] = []
    for category in categories_to_forecast:
        logger.info("Forecasting: %s", category)
        forecast, _model = forecast_spending(df, category, days=30, global_prior_daily=float(all_daily_prior))
        if forecast is None:
            logger.warning("Skipping %s — model returned None", category)
            continue
        plot_forecast(
            df=df,
            forecast=forecast,
            category=category,
            save_path=str(FORECASTS_DIR / f"{category.lower()}_forecast.png"),
        )
        all_summaries.append(get_forecast_summary(forecast, category))

    summary_df = combine_forecast_summaries(all_summaries)
    summary_df.to_csv(DATA_DIR / "forecast_summaries.csv", index=False)

    budgets = _load_budget_limits()
    budget_alerts = check_budget_alerts(summary_df, budgets)
    if not budget_alerts.empty:
        print("\nBudget Alerts:")
        print(budget_alerts[["category", "total_predicted_spend", "budget_limit"]].to_string(index=False))

    print(f"\n{'=' * 40}\nALL FORECASTS COMPLETE\n{'=' * 40}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
