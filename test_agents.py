import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pipeline():
    logger.info("Testing data pipeline...")
    from data.pipeline import run_pipeline
    df = run_pipeline(
        input_path="data/transactions.csv",
        output_path="data/cleaned_transactions.csv"
    )
    assert len(df) > 0, "Pipeline returned empty DataFrame"
    assert 'date' in df.columns, "Missing date column"
    assert 'amount' in df.columns, "Missing amount column"
    assert 'merchant' in df.columns, "Missing merchant column"
    logger.info(f"Pipeline OK — {len(df)} rows")
    return True


def test_classifier():
    logger.info("Testing classifier...")
    from agents.classifier import classify_transaction
    result = classify_transaction("Uber")
    assert 'category' in result, "Missing category key"
    assert 'confidence' in result, "Missing confidence key"
    assert result['confidence'] > 0, "Confidence should be > 0"
    logger.info(f"Classifier OK — Uber = {result['category']} ({result['confidence']})")
    return True


def test_classifier_batch_retry():
    logger.info("Testing classifier batch retry...")
    from agents.classifier import classify_batch_with_retry
    results = classify_batch_with_retry(["Uber", "Naivas Supermarket"])
    assert len(results) == 2, "Batch results size mismatch"
    assert all('category' in item for item in results), "Missing category in batch results"
    logger.info("Classifier batch retry OK")
    return True


def test_classifier_category_stats():
    logger.info("Testing classifier category stats...")
    from agents.classifier import compute_category_spending_stats
    df = pd.DataFrame({
        "predicted_category": ["Food", "Food", "Transport"],
        "amount": [100.0, 200.0, 50.0]
    })
    stats = compute_category_spending_stats(df)
    assert "percent_of_total" in stats.columns, "Missing percent_of_total"
    assert stats['total_spend'].sum() == 350.0, "Total spend mismatch"
    logger.info("Classifier category stats OK")
    return True


def test_forecaster():
    logger.info("Testing forecaster...")
    df = pd.read_csv("data/classified_transactions.csv")
    from agents.forecaster import forecast_spending
    forecast, model = forecast_spending(df, category="Food", days=30)
    assert forecast is not None, "Forecast returned None"
    assert len(forecast) > 0, "Forecast is empty"
    assert 'yhat' in forecast.columns, "Missing yhat column"
    logger.info(f"Forecaster OK — {len(forecast)} forecast rows")
    return True


def test_forecaster_mae():
    logger.info("Testing forecaster MAE...")
    from agents.forecaster import calculate_forecast_mae
    time_series = pd.DataFrame({
        "ds": pd.date_range("2026-01-01", periods=7, freq="D"),
        "y": [10, 12, 9, 11, 13, 12, 10]
    })
    forecast = pd.DataFrame({
        "ds": pd.date_range("2026-01-01", periods=7, freq="D"),
        "yhat": [9, 12, 10, 11, 14, 11, 10]
    })
    mae = calculate_forecast_mae(time_series, forecast, days=7)
    assert mae >= 0, "MAE should be non-negative"
    logger.info("Forecaster MAE OK")
    return True


def test_forecaster_budget_alerts():
    logger.info("Testing forecaster budget alerts...")
    from agents.forecaster import check_budget_alerts
    summary_df = pd.DataFrame([
        {"category": "Food", "total_predicted_spend": 20000},
        {"category": "Transport", "total_predicted_spend": 5000}
    ])
    alerts = check_budget_alerts(summary_df, {"Food": 15000, "Transport": 8000})
    assert "Food" in alerts['category'].values, "Food should trigger alert"
    logger.info("Forecaster budget alerts OK")
    return True


def test_anomaly_detector():
    logger.info("Testing anomaly detector...")
    df = pd.read_csv("data/classified_transactions.csv")
    from agents.anomaly_detector import detect_anomalies, inject_test_anomalies, evaluate_detector
    df_with_fakes = inject_test_anomalies(df, n=10)
    df_results = detect_anomalies(df_with_fakes)
    eval_results = evaluate_detector(df_results)
    assert eval_results['catch_rate'] >= 0.7, f"Catch rate too low: {eval_results['catch_rate']:.1%}"
    logger.info(f"Anomaly detector OK — catch rate: {eval_results['catch_rate']:.1%}")
    return True


def test_anomaly_severity_and_explanations():
    logger.info("Testing anomaly severity and explanations...")
    from agents.anomaly_detector import detect_anomalies
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=20, freq="D"),
        "amount": [100, 120, 95, 130, 110, 105, 5000, 115, 98, 102, 100, 99, 101, 103, 105, 107, 109, 111, 113, 115],
        "merchant": ["Test" for _ in range(20)],
        "category": ["Food" for _ in range(20)]
    })
    results = detect_anomalies(df)
    assert "severity" in results.columns, "Missing severity column"
    assert "anomaly_explanation" in results.columns, "Missing anomaly_explanation column"
    logger.info("Anomaly severity and explanations OK")
    return True


def test_weekly_anomaly_report():
    logger.info("Testing weekly anomaly report...")
    from agents.anomaly_detector import weekly_anomaly_report
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10, freq="D"),
        "amount": [100] * 10,
        "merchant": ["Test"] * 10,
        "category": ["Food"] * 10,
        "is_anomaly": [False] * 9 + [True]
    })
    report = weekly_anomaly_report(df)
    assert "anomaly_count" in report, "Missing anomaly_count in report"
    logger.info("Weekly anomaly report OK")
    return True


def test_rag_engine():
    logger.info("Testing RAG engine...")
    from agents.rag_engine import FinanceRAGEngine
    engine = FinanceRAGEngine(data_path="data/classified_transactions.csv")
    answer = engine.ask("How much did I spend on food?")
    assert len(answer) > 0, "RAG engine returned empty answer"
    assert answer != "None", "RAG engine returned None"
    logger.info(f"RAG engine OK — got answer: {answer[:80]}...")
    return True


def test_rag_filters_and_summary():
    logger.info("Testing RAG filters and summary...")
    from agents.rag_engine import apply_filters_to_df, compute_spending_summary
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "amount": [100.0, 200.0],
        "merchant": ["A", "B"],
        "category": ["Food", "Transport"]
    })
    filtered = apply_filters_to_df(df, {"category": "Food"})
    summary = compute_spending_summary(filtered)
    assert summary['total'] == 100.0, "Filtered summary total mismatch"
    logger.info("RAG filters and summary OK")
    return True


def test_rag_date_filter():
    logger.info("Testing RAG date filter...")
    from agents.rag_engine import apply_filters_to_df
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "amount": [100.0, 200.0, 300.0],
        "merchant": ["A", "B", "C"],
        "category": ["Food", "Food", "Food"]
    })
    filtered = apply_filters_to_df(df, {"start_date": "2026-01-02"})
    assert len(filtered) == 2, "Date filter did not limit rows"
    logger.info("RAG date filter OK")
    return True


def test_orchestrator_intents():
    logger.info("Testing orchestrator intent detection...")
    from orchestrator import detect_intents
    intents = detect_intents("Are there anomalies and a forecast for food?")
    intent_names = [item['intent'] for item in intents]
    assert "anomaly" in intent_names, "Anomaly intent missing"
    assert "forecast" in intent_names, "Forecast intent missing"
    logger.info("Orchestrator intent detection OK")
    return True


def test_orchestrator_budget_parse():
    logger.info("Testing orchestrator budget parse...")
    from orchestrator import _extract_budget_limits
    budgets = _extract_budget_limits("budget {\"Food\": 15000}")
    assert budgets.get("Food") == 15000, "Budget parse failed"
    logger.info("Orchestrator budget parse OK")
    return True


if __name__ == "__main__":
    tests = [
        ("Data Pipeline",       test_pipeline),
        ("Classifier",          test_classifier),
        ("Classifier Batch",    test_classifier_batch_retry),
        ("Classifier Stats",    test_classifier_category_stats),
        ("Forecaster",          test_forecaster),
        ("Forecaster MAE",      test_forecaster_mae),
        ("Forecaster Budget",   test_forecaster_budget_alerts),
        ("Anomaly Detector",    test_anomaly_detector),
        ("Anomaly Severity",    test_anomaly_severity_and_explanations),
        ("Anomaly Weekly",      test_weekly_anomaly_report),
        ("RAG Engine",          test_rag_engine),
        ("RAG Filters",         test_rag_filters_and_summary),
        ("RAG Date Filter",     test_rag_date_filter),
        ("Orchestrator Intents", test_orchestrator_intents),
        ("Orchestrator Budget",  test_orchestrator_budget_parse),
    ]
    
    results = []
    
    print("\n" + "="*50)
    print("FINSIGHT AGENT — DAY 7 INTEGRATION TESTS")
    print("="*50)
    
    for name, test_fn in tests:
        print(f"\nRunning: {name}...")
        try:
            passed = test_fn()
            results.append((name, "PASSED", ""))
            print(f"  PASSED")
        except Exception as e:
            results.append((name, "FAILED", str(e)))
            print(f"  FAILED — {e}")
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for name, status, error in results:
        icon = "[OK]" if status == "PASSED" else "[FAIL]"
        line = f"  {icon} {name}: {status}"
        if error:
            line += f" — {error}"
        print(line)
    
    passed_count = sum(1 for _, s, _ in results if s == "PASSED")
    print(f"\n{passed_count}/{len(tests)} tests passed")
    
    if passed_count < len(tests):
        sys.exit(1)