import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import orchestrator as orchestrator_module
from agents.anomaly_detector import detect_anomalies, weekly_anomaly_report
from agents.classifier import CATEGORIES, compute_category_spending_stats
from agents.forecaster import forecast_spending, get_forecast_summary, combine_forecast_summaries, check_budget_alerts
from agents.rag_engine import FinanceRAGEngine

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLASSIFIED_PATH = os.path.join(DATA_DIR, "classified_transactions.csv")

TAGS_METADATA = [
    {"name": "Query", "description": "Query and intent routing endpoints"},
    {"name": "Forecast", "description": "Forecast endpoints"},
    {"name": "Anomalies", "description": "Anomaly detection endpoints"},
    {"name": "Classifier", "description": "Classifier endpoints"},
    {"name": "Conversation", "description": "Conversation context endpoints"},
    {"name": "Insights", "description": "Aggregated dashboard endpoints"},
]

app = FastAPI(
    title="FinSight Agent API",
    description="AI-powered personal finance assistant",
    version="2.0.0",
    openapi_tags=TAGS_METADATA,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    conversation_history: Optional[list[ConversationMessage]] = None
    filters: Optional[dict[str, Any]] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    intent: Optional[str]
    intents: list[str]
    intent_confidence: Optional[dict[str, float]]
    agents_invoked: list[str]
    citations: list[dict]
    cached: bool
    execution_time_ms: float
    timestamp: str


class SummaryResponse(BaseModel):
    total_transactions: int
    total_spend: float
    top_category: str
    avg_transaction: float
    anomaly_count: int
    date_range: str
    execution_time_ms: float
    timestamp: str


class CategoriesResponse(BaseModel):
    categories: list[dict]
    execution_time_ms: float
    timestamp: str


class AnomaliesResponse(BaseModel):
    total_anomalies: int
    shown: int
    anomalies: list[dict]
    execution_time_ms: float
    timestamp: str


class ConversationRequest(BaseModel):
    messages: list[ConversationMessage]
    question: str = Field(..., min_length=1, max_length=500)


class ConversationResponse(BaseModel):
    answer: str
    messages: list[ConversationMessage]
    metadata: dict[str, Any]
    execution_time_ms: float
    timestamp: str


class FilteredQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class FilteredQueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    execution_time_ms: float
    timestamp: str


class ForecastResponse(BaseModel):
    category: str
    total: float
    avg_daily: float
    max_day: float
    budget_alert: bool
    changepoints: list[str]
    mae: Optional[float]


class ForecastAPIResponse(BaseModel):
    forecasts: list[ForecastResponse]
    budget_alerts: list[dict]
    execution_time_ms: float
    timestamp: str


class AnomalyDetail(BaseModel):
    date: str
    merchant: str
    category: str
    amount: float
    severity: str
    explanation: str
    anomaly_score: float


class AnomalyReportResponse(BaseModel):
    window_start: str
    window_end: str
    anomaly_count: int
    total_anomaly_amount: float
    top_categories: dict
    anomalies: list[AnomalyDetail]
    execution_time_ms: float
    timestamp: str


class ClassifierUncertainResponse(BaseModel):
    total_uncertain: int
    transactions: list[dict]
    execution_time_ms: float
    timestamp: str


class ClassifierStatsResponse(BaseModel):
    stats: list[dict]
    execution_time_ms: float
    timestamp: str


class InsightsResponse(BaseModel):
    summary: dict
    top_anomalies: list[AnomalyDetail]
    forecast: list[ForecastResponse]
    uncertain_transactions: list[dict]
    budget_alerts: list[dict]
    execution_time_ms: float
    timestamp: str


def _now_iso() -> str:
    return datetime.now().isoformat()


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(CLASSIFIED_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df


def _messages_to_history(messages: list[ConversationMessage]) -> list[dict]:
    history = []
    last_user = None
    for msg in messages:
        if msg.role == "user":
            last_user = msg.content
        elif msg.role == "assistant" and last_user:
            history.append({"query": last_user, "answer": msg.content, "metadata": {}})
            last_user = None
    return history


def _apply_history(messages: Optional[list[ConversationMessage]]) -> None:
    if not messages:
        return

    history = _messages_to_history(messages)
    if hasattr(orchestrator_module, "_CONVERSATION_HISTORY"):
        orchestrator_module._CONVERSATION_HISTORY.clear()
        orchestrator_module._CONVERSATION_HISTORY.extend(history)


def _compose_answer(intents: list[str], responses: dict) -> str:
    parts = []
    for intent in intents:
        label = intent.upper()
        answer = responses.get(intent, {}).get("answer", "")
        parts.append(f"[{label}]\n{answer}")
    return "\n\n".join(parts)


def _parse_budget_param(budget: Optional[str]) -> dict:
    if not budget:
        return {}

    parsed = {}
    for chunk in budget.split(","):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        key = key.strip()
        try:
            parsed[key] = float(value.strip())
        except ValueError:
            logger.warning("Invalid budget value for %s", key)
    return parsed


def _ensure_rag_engine() -> FinanceRAGEngine:
    if not hasattr(app.state, "rag_engine") or app.state.rag_engine is None:
        app.state.rag_engine = FinanceRAGEngine(data_path=CLASSIFIED_PATH)
    return app.state.rag_engine


def _cached_query_hit(question: str) -> bool:
    if not hasattr(orchestrator_module, "_RESULT_CACHE"):
        return False
    cache = orchestrator_module._RESULT_CACHE
    ttl = getattr(orchestrator_module, "_CACHE_TTL_SECONDS", 300)
    entry = cache.get(question)
    if not entry:
        return False
    return (time.time() - entry.get("timestamp", 0)) <= ttl


def _get_cached_anomalies() -> pd.DataFrame:
    if getattr(app.state, "anomaly_cache", None) is None:
        df = _load_df()
        app.state.anomaly_cache = detect_anomalies(df)
        app.state.anomaly_cache_ts = time.time()
    return app.state.anomaly_cache


def _top_candidate_cache() -> dict:
    if not hasattr(app.state, "candidate_cache"):
        app.state.candidate_cache = {}
    return app.state.candidate_cache


def _get_top_candidates(merchant: str) -> list[dict]:
    from agents.classifier import classifier as hf_classifier

    cache = _top_candidate_cache()
    key = merchant.strip().lower()
    if key in cache:
        return cache[key]

    result = hf_classifier(merchant, CATEGORIES)
    top_labels = result['labels'][:3]
    top_scores = result['scores'][:3]
    candidates = [
        {"category": label, "confidence": round(float(score), 4)}
        for label, score in zip(top_labels, top_scores)
    ]
    cache[key] = candidates
    return candidates


def _validate_date(value: Optional[str]) -> None:
    if value is None:
        return
    try:
        pd.to_datetime(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {value}") from exc


@app.on_event("startup")
def startup_event() -> None:
    """Preload models and caches, validate files, and log startup time."""
    start = time.perf_counter()
    required_files = [CLASSIFIED_PATH]

    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        logger.error("Missing required data files: %s", missing)
        raise RuntimeError(f"Missing required data files: {missing}")

    app.state.rag_engine = FinanceRAGEngine(data_path=CLASSIFIED_PATH)
    app.state.anomaly_cache = detect_anomalies(_load_df())
    app.state.anomaly_cache_ts = time.time()

    logger.info("Startup complete in %.2f ms", (time.perf_counter() - start) * 1000)


@app.get("/", tags=["Query"])
def root():
    """Return API metadata and available endpoints."""
    return {
        "name": "FinSight Agent API",
        "status": "running",
        "version": "2.0.0",
        "endpoints": ["/query", "/summary", "/health"]
    }


@app.get("/health", tags=["Query"])
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": _now_iso()
    }


@app.post("/query", response_model=QueryResponse, tags=["Query"], status_code=status.HTTP_200_OK)
def query(request: QueryRequest):
    """Primary query endpoint with multi-intent routing and optional RAG filters."""
    start_time = time.perf_counter()
    logger.info("Received query: %s", request.question)

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        _apply_history(request.conversation_history)
        cached_hit = _cached_query_hit(request.question)

        result = orchestrator_module.ask(request.question)
        metadata = result.get("metadata", {})
        intents = metadata.get("intents", [])

        if request.filters and "rag" in intents:
            engine = _ensure_rag_engine()
            rag_result = engine.ask_with_summary(request.question, filters=request.filters)
            result["responses"]["rag"] = {
                "answer": rag_result.get("answer", ""),
                "summary": rag_result.get("summary", {}),
                "citations": rag_result.get("citations", []),
            }
            result["answer"] = _compose_answer(intents, result["responses"])

        citations = result.get("responses", {}).get("rag", {}).get("citations", [])
        intent_confidence = metadata.get("intent_confidence")
        intent = intents[0] if intents else None

        execution_time = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            question=request.question,
            answer=result.get("answer", ""),
            intent=intent,
            intents=intents,
            intent_confidence=intent_confidence,
            agents_invoked=intents,
            citations=citations,
            cached=bool(cached_hit),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Query error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query error: {exc}")


@app.post("/query/filtered", response_model=FilteredQueryResponse, tags=["Query"], status_code=status.HTTP_200_OK)
def query_filtered(request: FilteredQueryRequest):
    """Query endpoint that applies metadata filters to the RAG engine."""
    start_time = time.perf_counter()
    logger.info("Received filtered query: %s", request.question)

    _validate_date(request.date_from)
    _validate_date(request.date_to)

    filters = {}
    if request.category:
        filters["category"] = request.category
    if request.date_from:
        filters["start_date"] = request.date_from
    if request.date_to:
        filters["end_date"] = request.date_to

    try:
        engine = _ensure_rag_engine()
        result = engine.ask_with_summary(request.question, filters=filters)
        execution_time = (time.perf_counter() - start_time) * 1000
        return FilteredQueryResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Filtered query error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Filtered query error: {exc}")


@app.post("/conversation", response_model=ConversationResponse, tags=["Conversation"], status_code=status.HTTP_200_OK)
def conversation(request: ConversationRequest):
    """Handles a full conversation history plus a new question."""
    start_time = time.perf_counter()
    logger.info("Received conversation question: %s", request.question)

    try:
        _apply_history(request.messages)
        result = orchestrator_module.ask(request.question)
        execution_time = (time.perf_counter() - start_time) * 1000
        updated_messages = request.messages + [
            ConversationMessage(role="user", content=request.question),
            ConversationMessage(role="assistant", content=result.get("answer", ""))
        ]
        return ConversationResponse(
            answer=result.get("answer", ""),
            messages=updated_messages,
            metadata=result.get("metadata", {}),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Conversation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Conversation error: {exc}")


@app.get("/summary", response_model=SummaryResponse, tags=["Query"], status_code=status.HTTP_200_OK)
def get_summary():
    """Returns an overview summary of spending and anomaly counts."""
    start_time = time.perf_counter()

    try:
        df = _load_df()
        df_anomalies = _get_cached_anomalies()
        anomaly_count = int(df_anomalies['is_anomaly'].sum())

        top_category = (
            df.groupby('category')['amount']
            .sum()
            .idxmax()
        )

        date_range = (
            f"{df['date'].min().strftime('%d %b %Y')} to "
            f"{df['date'].max().strftime('%d %b %Y')}"
        )

        execution_time = (time.perf_counter() - start_time) * 1000
        return SummaryResponse(
            total_transactions=len(df),
            total_spend=round(float(df['amount'].sum()), 2),
            top_category=top_category,
            avg_transaction=round(float(df['amount'].mean()), 2),
            anomaly_count=anomaly_count,
            date_range=date_range,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Summary error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Summary error: {exc}")


@app.get("/categories", response_model=CategoriesResponse, tags=["Classifier"], status_code=status.HTTP_200_OK)
def get_categories():
    """Returns category breakdown summary."""
    start_time = time.perf_counter()

    try:
        df = _load_df()

        breakdown = (
            df.groupby('category')['amount']
            .agg(['count', 'sum', 'mean'])
            .round(2)
            .reset_index()
        )

        result = []
        total_spend = df['amount'].sum()

        for _, row in breakdown.iterrows():
            result.append({
                "category": row['category'],
                "transaction_count": int(row['count']),
                "total_spend": round(float(row['sum']), 2),
                "avg_spend": round(float(row['mean']), 2),
                "percentage_of_total": round((row['sum'] / total_spend) * 100, 1)
            })

        result = sorted(result, key=lambda x: x['total_spend'], reverse=True)
        execution_time = (time.perf_counter() - start_time) * 1000
        return CategoriesResponse(
            categories=result,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Category breakdown error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Category breakdown error: {exc}")


@app.get("/anomalies", response_model=AnomaliesResponse, tags=["Anomalies"], status_code=status.HTTP_200_OK)
def get_anomalies(limit: int = Query(10, ge=1, le=100)):
    """Returns the top anomalous transactions."""
    start_time = time.perf_counter()

    try:
        df_results = _get_cached_anomalies()

        anomalies = (
            df_results[df_results['is_anomaly'] == True]
            .sort_values('anomaly_score', ascending=False)
            .head(limit)
        )

        result = []
        for _, row in anomalies.iterrows():
            result.append({
                "date": str(row['date']),
                "merchant": row['merchant'],
                "category": row['category'],
                "amount": round(float(row['amount']), 2),
                "anomaly_score": round(float(row['anomaly_score']), 4),
                "severity": row.get('severity'),
                "explanation": row.get('anomaly_explanation', "")
            })

        execution_time = (time.perf_counter() - start_time) * 1000
        return AnomaliesResponse(
            total_anomalies=int(df_results['is_anomaly'].sum()),
            shown=len(result),
            anomalies=result,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Anomaly error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Anomaly error: {exc}")


@app.get("/anomalies/report", response_model=AnomalyReportResponse, tags=["Anomalies"], status_code=status.HTTP_200_OK)
def anomaly_report():
    """Returns a weekly anomaly report with detailed anomalies."""
    start_time = time.perf_counter()

    try:
        df_results = _get_cached_anomalies()
        report = weekly_anomaly_report(df_results)

        latest_date = df_results['date'].max()
        window_start = latest_date - pd.Timedelta(days=7)
        recent = df_results[(df_results['date'] >= window_start) & (df_results['is_anomaly'])]

        anomalies = [
            AnomalyDetail(
                date=str(row.date),
                merchant=row.merchant,
                category=row.category,
                amount=float(row.amount),
                severity=row.severity,
                explanation=row.anomaly_explanation,
                anomaly_score=float(row.anomaly_score),
            )
            for row in recent.itertuples()
        ]

        execution_time = (time.perf_counter() - start_time) * 1000
        return AnomalyReportResponse(
            window_start=report['window_start'],
            window_end=report['window_end'],
            anomaly_count=report['anomaly_count'],
            total_anomaly_amount=report['total_anomaly_amount'],
            top_categories=report['top_categories'],
            anomalies=anomalies,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Anomaly report error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Anomaly report error: {exc}")


@app.get("/anomalies/severity/{level}", response_model=AnomaliesResponse, tags=["Anomalies"], status_code=status.HTTP_200_OK)
def anomalies_by_severity(level: str):
    """Returns anomalies filtered by severity level."""
    start_time = time.perf_counter()
    level = level.upper()

    if level not in {"LOW", "MEDIUM", "HIGH"}:
        raise HTTPException(status_code=400, detail="Severity must be low, medium, or high")

    try:
        df_results = _get_cached_anomalies()
        filtered = df_results[(df_results['is_anomaly']) & (df_results['severity'] == level)]

        result = [
            {
                "date": str(row.date),
                "merchant": row.merchant,
                "category": row.category,
                "amount": round(float(row.amount), 2),
                "anomaly_score": round(float(row.anomaly_score), 4),
                "severity": row.severity,
                "explanation": row.anomaly_explanation,
            }
            for row in filtered.itertuples()
        ]

        execution_time = (time.perf_counter() - start_time) * 1000
        return AnomaliesResponse(
            total_anomalies=len(result),
            shown=len(result),
            anomalies=result,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Anomaly severity error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Anomaly severity error: {exc}")


@app.get("/forecast", response_model=ForecastAPIResponse, tags=["Forecast"], status_code=status.HTTP_200_OK)
def forecast(category: Optional[str] = None, budget: Optional[str] = None):
    """Returns 30-day forecast summaries, budget alerts, changepoints, and MAE metrics."""
    start_time = time.perf_counter()

    try:
        df = _load_df()
        categories = sorted(df['category'].unique())
        if category:
            if category not in categories:
                raise HTTPException(status_code=404, detail="Category not found")
            categories = [category]

        summaries = []
        for cat in categories:
            forecast_df, _ = forecast_spending(df, category=cat, days=30)
            if forecast_df is None:
                continue
            summaries.append(get_forecast_summary(forecast_df, cat))

        summary_df = combine_forecast_summaries(summaries)
        budgets = _parse_budget_param(budget)
        budget_alerts_df = check_budget_alerts(summary_df, budgets)
        budget_alerts = budget_alerts_df.to_dict('records') if not budget_alerts_df.empty else []

        alerts_set = {alert.get('category') for alert in budget_alerts}
        forecasts = [
            ForecastResponse(
                category=item['category'],
                total=float(item['total_predicted_spend']),
                avg_daily=float(item['avg_daily_spend']),
                max_day=float(item['max_daily_spend']),
                budget_alert=item['category'] in alerts_set,
                changepoints=item.get('changepoints', []),
                mae=item.get('mae_7d'),
            )
            for item in summary_df.to_dict('records')
        ]

        execution_time = (time.perf_counter() - start_time) * 1000
        return ForecastAPIResponse(
            forecasts=forecasts,
            budget_alerts=budget_alerts,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Forecast error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}")


@app.get("/classifier/uncertain", response_model=ClassifierUncertainResponse, tags=["Classifier"], status_code=status.HTTP_200_OK)
def classifier_uncertain():
    """Returns transactions flagged as uncertain with top candidate categories."""
    start_time = time.perf_counter()

    try:
        df = _load_df()
        if 'is_uncertain' in df.columns:
            uncertain = df[df['is_uncertain'] == True]
        else:
            uncertain = df.iloc[0:0]

        results = []
        for row in uncertain.itertuples():
            candidates = _get_top_candidates(row.merchant)
            results.append({
                "date": str(row.date),
                "merchant": row.merchant,
                "amount": round(float(row.amount), 2),
                "predicted_category": row.predicted_category,
                "confidence_score": round(float(row.confidence_score), 4),
                "top_candidates": candidates,
            })

        execution_time = (time.perf_counter() - start_time) * 1000
        return ClassifierUncertainResponse(
            total_uncertain=len(results),
            transactions=results,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Uncertain classifier error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Uncertain classifier error: {exc}")


@app.get("/classifier/stats", response_model=ClassifierStatsResponse, tags=["Classifier"], status_code=status.HTTP_200_OK)
def classifier_stats():
    """Returns category spending statistics."""
    start_time = time.perf_counter()

    try:
        df = _load_df()
        stats = compute_category_spending_stats(df)
        execution_time = (time.perf_counter() - start_time) * 1000
        return ClassifierStatsResponse(
            stats=stats.to_dict('records'),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Classifier stats error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Classifier stats error: {exc}")


@app.get("/insights", response_model=InsightsResponse, tags=["Insights"], status_code=status.HTTP_200_OK)
def insights(budget: Optional[str] = None):
    """Returns a premium dashboard payload across multiple agents."""
    start_time = time.perf_counter()

    try:
        df = _load_df()
        df_anomalies = _get_cached_anomalies()

        summary = {
            "total_transactions": int(len(df)),
            "total_spend": round(float(df['amount'].sum()), 2),
            "avg_transaction": round(float(df['amount'].mean()), 2),
        }

        top_anomalies_df = (
            df_anomalies[df_anomalies['is_anomaly'] == True]
            .sort_values('anomaly_score', ascending=False)
            .head(5)
        )

        top_anomalies = [
            AnomalyDetail(
                date=str(row.date),
                merchant=row.merchant,
                category=row.category,
                amount=float(row.amount),
                severity=row.severity,
                explanation=row.anomaly_explanation,
                anomaly_score=float(row.anomaly_score),
            )
            for row in top_anomalies_df.itertuples()
        ]

        categories = sorted(df['category'].unique())
        summaries = []
        for cat in categories:
            forecast_df, _ = forecast_spending(df, category=cat, days=30)
            if forecast_df is None:
                continue
            summaries.append(get_forecast_summary(forecast_df, cat))

        summary_df = combine_forecast_summaries(summaries)
        budgets = _parse_budget_param(budget)
        budget_alerts_df = check_budget_alerts(summary_df, budgets)
        budget_alerts = budget_alerts_df.to_dict('records') if not budget_alerts_df.empty else []

        alerts_set = {alert.get('category') for alert in budget_alerts}
        forecast = [
            ForecastResponse(
                category=item['category'],
                total=float(item['total_predicted_spend']),
                avg_daily=float(item['avg_daily_spend']),
                max_day=float(item['max_daily_spend']),
                budget_alert=item['category'] in alerts_set,
                changepoints=item.get('changepoints', []),
                mae=item.get('mae_7d'),
            )
            for item in summary_df.to_dict('records')
        ]

        if 'is_uncertain' in df.columns:
            uncertain = df[df['is_uncertain'] == True]
        else:
            uncertain = df.iloc[0:0]

        uncertain_transactions = [
            {
                "date": str(row.date),
                "merchant": row.merchant,
                "amount": round(float(row.amount), 2),
                "predicted_category": row.predicted_category,
                "confidence_score": round(float(row.confidence_score), 4),
            }
            for row in uncertain.itertuples()
        ]

        execution_time = (time.perf_counter() - start_time) * 1000
        return InsightsResponse(
            summary=summary,
            top_anomalies=top_anomalies,
            forecast=forecast,
            uncertain_transactions=uncertain_transactions,
            budget_alerts=budget_alerts,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso()
        )
    except Exception as exc:
        logger.error("Insights error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Insights error: {exc}")
