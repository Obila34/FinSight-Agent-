from pathlib import Path
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CLASSIFIED_PATH = DATA_DIR / "classified_transactions.csv"
FORECASTS_DIR = DATA_DIR / "forecasts"


def prepare_time_series(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Takes the transactions DataFrame and converts it into a time series format suitable for Prophet forecasting.

    Prophet requires the data in a specific format:
    - A column called 'ds' (datestamp) - the dates
    - A column called 'y' (your value) - the amounts
    """

    category_df = df[df["category"] == category].copy()
    category_df["date"] = pd.to_datetime(category_df["date"])
    daily_spend = category_df.groupby("date")["amount"].sum().reset_index()
    daily_spend = daily_spend.rename(columns={"date": "ds", "amount": "y"})
    daily_spend = daily_spend.sort_values("ds").reset_index(drop=True)

    logger.info("Prepared %s daily data points for category: %s", len(daily_spend), category)
    return daily_spend


def extract_changepoints(model: Prophet) -> list:
    """
    Extracts changepoint dates from a trained Prophet model.

    Returns:
        list of ISO date strings for changepoints.
    """

    if not hasattr(model, "changepoints"):
        return []

    changepoints = [pd.Timestamp(cp).date().isoformat() for cp in model.changepoints]
    return changepoints


def calculate_forecast_mae(time_series: pd.DataFrame, forecast: pd.DataFrame, days: int = 7) -> float:
    """
    Computes MAE between actuals and predictions for the last N days of history.

    Args:
        time_series: DataFrame with ds and y columns.
        forecast: Prophet forecast DataFrame containing yhat.
        days: number of recent days to evaluate.

    Returns:
        MAE value rounded to 2 decimals.
    """

    recent_actuals = time_series.tail(days)
    merged = recent_actuals.merge(forecast[["ds", "yhat"]], on="ds", how="inner")

    if merged.empty:
        logger.warning("No overlap between actuals and forecast for MAE calculation")
        return float("nan")

    mae = (merged["y"] - merged["yhat"]).abs().mean()
    mae = round(float(mae), 2)
    logger.info("MAE over last %s days: %.2f", days, mae)
    return mae


def forecast_spending(df: pd.DataFrame, category: str, days: int = 30) -> tuple:
    """
    Trains a Prophet model on historical spending data for a category and predicts spending for the next days.
    """

    logger.info("Training forecast model for category: %s", category)
    time_series = prepare_time_series(df, category)

    if len(time_series) < 10:
        logger.warning("Not enough data for %s - only %s days", category, len(time_series))
        return None, None

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )

    model.fit(time_series)
    changepoints = extract_changepoints(model)
    if changepoints:
        logger.info("Detected changepoints for %s: %s", category, ", ".join(changepoints))

    future_dates = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future_dates)

    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    mae_7d = calculate_forecast_mae(time_series, forecast, days=7)
    forecast.attrs["mae_7d"] = mae_7d
    forecast.attrs["changepoints"] = changepoints

    logger.info("Forecast complete for %s - next %s days predicted", category, days)
    return forecast, model


def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame, category: str, save_path: str | None = None):
    """
    Creates a chart showing historical spending and the future forecast.
    """

    actual = prepare_time_series(df, category)
    future_forecast = forecast.tail(30)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.scatter(actual["ds"], actual["y"], color="black", s=20, label="Actual spend", zorder=3)

    ax.plot(
        future_forecast["ds"],
        future_forecast["yhat"],
        color="steelblue",
        linewidth=2,
        label="Forecast",
    )

    ax.fill_between(
        future_forecast["ds"],
        future_forecast["yhat_lower"],
        future_forecast["yhat_upper"],
        alpha=0.3,
        color="steelblue",
        label="95% confidence interval",
    )

    ax.set_title(f"30-Day Spending Forecast - {category}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Spend (KES)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Chart saved to %s", save_path)

    plt.close(fig)


def get_forecast_summary(forecast: pd.DataFrame, category: str) -> dict:
    """
    Extracts key numbers from the forecast for easy reading.
    """

    future = forecast.tail(30)

    total_predicted = round(future["yhat"].sum(), 2)
    avg_daily = round(future["yhat"].mean(), 2)
    max_day = round(future["yhat"].max(), 2)
    min_day = round(future["yhat"].min(), 2)

    return {
        "category": category,
        "forecast_days": 30,
        "total_predicted_spend": total_predicted,
        "avg_daily_spend": avg_daily,
        "max_daily_spend": max_day,
        "min_daily_spend": min_day,
        "mae_7d": forecast.attrs.get("mae_7d"),
        "changepoints": forecast.attrs.get("changepoints", []),
    }


def combine_forecast_summaries(summaries: list[dict]) -> pd.DataFrame:
    """
    Combines per-category summaries into a single table.

    Returns:
        DataFrame with one row per category.
    """

    if not summaries:
        return pd.DataFrame()

    return pd.DataFrame(summaries)


def check_budget_alerts(summary_df: pd.DataFrame, budget_limits: dict) -> pd.DataFrame:
    """
    Flags categories where the forecast exceeds configured budgets.

    Args:
        summary_df: DataFrame from combine_forecast_summaries.
        budget_limits: dict mapping category to budget.

    Returns:
        DataFrame of budget alerts.
    """

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


def _load_budget_limits() -> dict:
    budget_env = os.getenv("FINSIGHT_BUDGETS")
    if not budget_env:
        return {}

    try:
        budgets = json.loads(budget_env)
        if isinstance(budgets, dict):
            return budgets
        logger.warning("FINSIGHT_BUDGETS must be a JSON object")
    except json.JSONDecodeError:
        logger.warning("Failed to parse FINSIGHT_BUDGETS JSON")

    return {}


def main() -> None:
    logger.info("Loading classified transactions...")

    if not CLASSIFIED_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {CLASSIFIED_PATH}")

    df = pd.read_csv(CLASSIFIED_PATH)
    df["date"] = pd.to_datetime(df["date"])

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    categories_to_forecast = ["Food", "Transport", "Entertainment", "Shopping"]
    all_summaries = []

    for category in categories_to_forecast:
        print(f"\n{'='*40}")
        print(f"Forecasting: {category}")
        print("=" * 40)

        forecast, model = forecast_spending(df, category, days=30)

        if forecast is None:
            print(f"Skipping {category} - not enough data")
            continue

        plot_forecast(
            df=df,
            forecast=forecast,
            category=category,
            save_path=str(FORECASTS_DIR / f"{category.lower()}_forecast.png"),
        )

        summary = get_forecast_summary(forecast, category)
        all_summaries.append(summary)

        print(f"\nForecast Summary for {category}:")
        print(f"  Total predicted spend (30 days) : KES {summary['total_predicted_spend']:,.2f}")
        print(f"  Average daily spend             : KES {summary['avg_daily_spend']:,.2f}")
        print(f"  Highest predicted day           : KES {summary['max_daily_spend']:,.2f}")
        print(f"  Lowest predicted day            : KES {summary['min_daily_spend']:,.2f}")
        if summary.get("mae_7d") is not None:
            print(f"  MAE (last 7 days)               : KES {summary['mae_7d']:,.2f}")

    summary_df = combine_forecast_summaries(all_summaries)
    summary_df.to_csv(DATA_DIR / "forecast_summaries.csv", index=False)

    budgets = _load_budget_limits()
    budget_alerts = check_budget_alerts(summary_df, budgets)
    if not budget_alerts.empty:
        print("\nBudget Alerts:")
        print(budget_alerts[["category", "total_predicted_spend", "budget_limit"]].to_string(index=False))

    print(f"\n{'='*40}")
    print("ALL FORECASTS COMPLETE")
    print(f"{'='*40}")
    print(summary_df.to_string(index=False))
    print("\nCharts saved to data/forecasts/")


if __name__ == "__main__":
    main()
