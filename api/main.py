from __future__ import annotations

import logging
import os
import io
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Literal, Optional, TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import orchestrator as orchestrator_module
from agents.anomaly_detector import (
    anomaly_trend_analysis,
    detect_anomalies,
    merchant_risk_score,
    peer_comparison,
    weekly_anomaly_report,
)
from agents.classifier import (
    CATEGORIES,
    MERCHANT_KEYWORD_MAP,
    classify_transaction,
    compute_category_spending_stats,
    feedback_loop,
    keyword_fallback_classify,
    normalize_input_for_classification,
)
from agents.forecaster import (
    check_budget_alerts,
    combine_forecast_summaries,
    detect_recurring_charges,
    forecast_spending,
    get_forecast_summary,
    savings_projection,
    what_if_forecast,
)
from agents.orchestrator import circuit_open
from data.pipeline import clean_data

if TYPE_CHECKING:
    from agents.rag_engine import FinanceRAGEngine

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLASSIFIED_PATH = os.path.join(DATA_DIR, "classified_transactions.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_transactions.csv")

TAGS_METADATA = [
    {"name": "Query", "description": "Query and intent routing endpoints"},
    {"name": "Forecast", "description": "Forecast endpoints"},
    {"name": "Anomalies", "description": "Anomaly detection endpoints"},
    {"name": "Classifier", "description": "Classifier endpoints"},
    {"name": "Conversation", "description": "Conversation context endpoints"},
    {"name": "Insights", "description": "Aggregated dashboard endpoints"},
    {"name": "System", "description": "Health and system metadata"},
]

app = FastAPI(
    title="FinSight Agent API",
    description="AI-powered personal finance assistant",
    version="2.1.0",
    openapi_tags=TAGS_METADATA,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        logger.exception("request_id=%s unhandled error", request_id)
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Execution-Time"] = str(elapsed_ms)
    logger.info(
        "request_id=%s %s %s completed in %sms",
        request_id,
        request.method,
        request.url.path,
        elapsed_ms,
    )
    return response


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
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
    intent_confidence: Optional[float]
    agents_invoked: list[str]
    citations: list[dict[str, Any]]
    source_citations: list[dict[str, Any]]
    cached: bool
    execution_time_ms: float
    timestamp: str
    conversation_history: Optional[list[ConversationMessage]] = None
    filters: Optional[dict[str, Any]] = None
    proactive_alert: Optional[str] = None
    answer_quality_score: Optional[float] = None


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
    categories: list[dict[str, Any]]
    execution_time_ms: float
    timestamp: str


class AnomaliesResponse(BaseModel):
    total_anomalies: int
    shown: int
    anomalies: list[dict[str, Any]]
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
    session_summary: Optional[str] = None
    proactive_alert: Optional[str] = None


class FilteredQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class FilteredQueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    summary: dict[str, Any]
    execution_time_ms: float
    timestamp: str


class ForecastResponse(BaseModel):
    category: str
    total_forecast: float
    avg_daily: float
    max_day: float
    lower_bound: float
    upper_bound: float
    budget_alert: bool
    budget_limit: Optional[float] = None
    changepoints: list[str]
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    recurring_charges: list[dict[str, Any]] = Field(default_factory=list)


class ForecastAPIResponse(BaseModel):
    forecasts: list[ForecastResponse]
    budget_alerts: list[dict[str, Any]]
    changepoints: list[str]
    recurring_charges: list[dict[str, Any]]
    savings_projection: dict[str, float]
    execution_time_ms: float
    timestamp: str


class WhatIfForecastResponse(BaseModel):
    category: str
    change_pct: float
    adjusted_total_30d: float
    savings_impact_vs_base: float
    execution_time_ms: float
    timestamp: str


class AnomalyDetail(BaseModel):
    date: str
    merchant: str
    category: str
    amount: float
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    explanation: str
    anomaly_score: float
    merchant_risk_score: Optional[float] = None


class AnomalyReportResponse(BaseModel):
    window_start: str
    window_end: str
    anomaly_count: int
    total_anomaly_amount: float
    top_categories: dict[str, int]
    anomalies: list[AnomalyDetail]
    merchant_risk: list[dict[str, Any]]
    execution_time_ms: float
    timestamp: str


class AnomalyTrendsResponse(BaseModel):
    trend: dict[str, Any]
    execution_time_ms: float
    timestamp: str


class ClassifierUncertainResponse(BaseModel):
    total_uncertain: int
    transactions: list[dict[str, Any]]
    execution_time_ms: float
    timestamp: str


class ClassifierStatsResponse(BaseModel):
    stats: list[dict[str, Any]]
    velocity: list[dict[str, Any]]
    execution_time_ms: float
    timestamp: str


class ClassifierFeedbackRequest(BaseModel):
    transaction_id: str
    correct_category: str


class ClassifierFeedbackResponse(BaseModel):
    ok: bool
    cache_size: int
    detail: str


class InsightsResponse(BaseModel):
    spending_summary: dict[str, Any]
    top_anomalies: list[AnomalyDetail]
    forecast: list[ForecastResponse]
    uncertain_transactions: list[dict[str, Any]]
    budget_warnings: list[dict[str, Any]]
    peer_comparison: list[dict[str, Any]]
    recurring_charges: list[dict[str, Any]]
    savings_projection: Optional[dict[str, float]] = None
    generated_at: str
    execution_time_ms: float
    timestamp: str


class PeerComparisonResponse(BaseModel):
    benchmark_currency: str
    comparisons: list[dict[str, Any]]
    execution_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    is_ready: bool
    uptime_seconds: float


class ReadyResponse(BaseModel):
    status: str
    models_loaded: bool
    details: dict[str, bool]
    timestamp: str


class IngestAnalyzeResponse(BaseModel):
    filename: str
    total_rows: int
    classified_rows: int
    anomaly_count: int
    categories: list[dict[str, Any]]
    anomalies: list[dict[str, Any]]
    records: list[dict[str, Any]]
    execution_time_ms: float
    timestamp: str


def _now_iso() -> str:
    return datetime.now().isoformat()


def _init_model_status() -> dict[str, bool]:
    return {
        "rag_engine": False,
        "anomaly_cache": False,
        "classifier_model": False,
    }


def _all_models_loaded() -> bool:
    status_map = getattr(app.state, "model_status", _init_model_status())
    return bool(status_map) and all(status_map.values())


def _set_model_loaded(name: str, loaded: bool) -> None:
    if not hasattr(app.state, "model_status"):
        app.state.model_status = _init_model_status()
    app.state.model_status[name] = bool(loaded)


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(CLASSIFIED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if "predicted_category" in df.columns:
        df["category"] = df["predicted_category"].fillna(df.get("category"))
    return df


def _messages_to_history(messages: list[ConversationMessage]) -> list[dict[str, Any]]:
    history = []
    last_user: str | None = None
    for msg in messages:
        if msg.role == "user":
            last_user = msg.content
        elif msg.role == "assistant" and last_user:
            history.append({"query": last_user, "answer": msg.content, "metadata": {}})
            last_user = None
    return history


def _category_col(df: pd.DataFrame) -> str:
    return "predicted_category" if "predicted_category" in df.columns else "category"


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


def _parse_budget_param(budget: Optional[str]) -> dict[str, float]:
    if not budget:
        return {}

    parsed: dict[str, float] = {}
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


def _ensure_rag_engine():
    if not hasattr(app.state, "rag_engine") or app.state.rag_engine is None:
        from agents.rag_engine import FinanceRAGEngine

        app.state.rag_engine = FinanceRAGEngine(data_path=CLASSIFIED_PATH)
        _set_model_loaded("rag_engine", True)
        logger.info("RAG engine loaded")
    return app.state.rag_engine


def _get_cached_anomalies() -> pd.DataFrame:
    if getattr(app.state, "anomaly_cache", None) is None:
        df = _load_df()
        app.state.anomaly_cache = detect_anomalies(df)
        app.state.anomaly_cache_ts = time.time()
        _set_model_loaded("anomaly_cache", True)
        logger.info("Anomaly cache loaded")
    return app.state.anomaly_cache


def _top_candidate_cache() -> dict[str, list[dict[str, Any]]]:
    if not hasattr(app.state, "candidate_cache"):
        app.state.candidate_cache = {}
    return app.state.candidate_cache


def _get_top_candidates(merchant: str) -> list[dict[str, Any]]:
    import agents.classifier as clf_mod

    cache = _top_candidate_cache()
    key = normalize_input_for_classification(merchant)
    if key in cache:
        return cache[key]

    try:
        clf_mod._ensure_classifier()
        _set_model_loaded("classifier_model", True)
        logger.info("Classifier model loaded")
        result = clf_mod.classifier(merchant, CATEGORIES)
    except Exception:
        logger.warning("Classifier model unavailable; returning empty top candidates.")
        return []
    top_labels = result["labels"][:3]
    top_scores = result["scores"][:3]
    candidates = [{"category": label, "confidence": round(float(score), 4)} for label, score in zip(top_labels, top_scores)]
    cache[key] = candidates
    return candidates


def _warm_models_background() -> None:
    """Best-effort model warm-up running after server starts accepting traffic."""
    logger.info("Background warm-up started")
    try:
        if os.path.exists(CLASSIFIED_PATH):
            try:
                _ensure_rag_engine()
            except Exception as exc:
                logger.warning("Background warm-up: RAG unavailable: %s", exc)

            try:
                _get_cached_anomalies()
            except Exception as exc:
                logger.warning("Background warm-up: anomaly cache unavailable: %s", exc)

            try:
                _get_top_candidates("uber")
            except Exception as exc:
                logger.warning("Background warm-up: classifier unavailable: %s", exc)
    finally:
        logger.info("Background warm-up finished; model_status=%s", getattr(app.state, "model_status", {}))


def _validate_date(value: Optional[str]) -> None:
    if value is None:
        return
    try:
        pd.to_datetime(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {value}") from exc


def _normalize_uploaded_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column names into expected schema: date, merchant, amount."""
    col_map = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    date_col = pick("date", "transaction_date", "posted_date")
    merchant_col = pick("merchant", "description", "payee", "name")
    amount_col = pick("amount", "value", "transaction_amount")

    if not date_col or not merchant_col or not amount_col:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file must include date/merchant/amount columns (or common aliases like "
                "transaction_date, description, transaction_amount)."
            ),
        )

    normalized = pd.DataFrame(
        {
            "date": df[date_col],
            "merchant": df[merchant_col],
            "amount": pd.to_numeric(df[amount_col], errors="coerce"),
        }
    )

    if "category" in df.columns:
        normalized["category"] = df["category"]

    normalized = normalized.dropna(subset=["date", "merchant", "amount"])
    return normalized


def require_rag_available() -> None:
    if circuit_open("rag"):
        logger.error("RAG circuit breaker open — rejecting request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG subsystem degraded (circuit breaker open). Please retry shortly.",
        )
    try:
        _ensure_rag_engine()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG subsystem unavailable: {exc}",
        ) from exc


@app.on_event("startup")
def startup_event() -> None:
    """Initialize quickly; heavy model loads are lazy/background for Cloud Run startup safety."""
    start = time.perf_counter()
    app.state.start_time = time.time()
    app.state.is_ready = True
    app.state.model_status = _init_model_status()
    app.state.rag_engine = None
    app.state.anomaly_cache = None
    app.state.classifier_stats_cache = None

    for path in [CLASSIFIED_PATH, CLEANED_PATH]:
        if not os.path.exists(path):
            logger.warning("Optional data file missing — continuing with degraded mode: %s", path)

    try:
        from agents.orchestrator import reset_circuit

        for agent in ("rag", "forecast", "anomaly", "classify"):
            reset_circuit(agent)
    except Exception:
        logger.exception("Startup encountered errors while resetting circuit breakers.")

    if os.getenv("FINSIGHT_WARM_MODELS", "false").strip().lower() == "true":
        threading.Thread(target=_warm_models_background, daemon=True).start()

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Startup finished in %.2f ms — is_ready=%s", elapsed_ms, app.state.is_ready)


@app.get("/", tags=["System"], summary="Root metadata")
def root():
    return {
        "name": "FinSight Agent API",
        "status": "running",
        "version": "2.1.0",
        "endpoints": [
            "/query",
            "/summary",
            "/health",
            "/forecast",
            "/anomalies",
            "/insights",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Liveness and readiness")
def health_check():
    uptime = max(0.0, time.time() - getattr(app.state, "start_time", time.time()))
    return HealthResponse(status="healthy", is_ready=True, uptime_seconds=round(uptime, 3))


@app.get("/ready", response_model=ReadyResponse, tags=["System"], summary="Readiness for heavy model features")
def readiness_check():
    details = dict(getattr(app.state, "model_status", _init_model_status()))
    all_loaded = _all_models_loaded()
    if not all_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "warming",
                "models_loaded": False,
                "details": details,
                "timestamp": _now_iso(),
            },
        )
    return ReadyResponse(status="ready", models_loaded=True, details=details, timestamp=_now_iso())


@app.post("/query", response_model=QueryResponse, tags=["Query"], status_code=status.HTTP_200_OK)
def query(http_request: Request, payload: QueryRequest):
    start_time = time.perf_counter()
    rid = getattr(http_request.state, "request_id", None)

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        _apply_history(payload.conversation_history)

        result = orchestrator_module.ask(payload.question)
        metadata = result.get("metadata", {})
        intents = metadata.get("intents", [])

        if payload.filters and "rag" in intents:
            require_rag_available()
            engine = _ensure_rag_engine()
            rag_result = engine.ask_with_summary(payload.question, filters=payload.filters)
            result.setdefault("responses", {})["rag"] = {
                "answer": rag_result.get("answer", ""),
                "summary": rag_result.get("summary", {}),
                "citations": rag_result.get("citations", []),
            }
            result["answer"] = _compose_answer(intents, result["responses"])

        citations = result.get("responses", {}).get("rag", {}).get("citations", [])
        intent_conf_dict = metadata.get("intent_confidence") or {}
        top_conf = float(next(iter(intent_conf_dict.values()))) if intent_conf_dict else None
        intent = intents[0] if intents else None

        execution_time = (time.perf_counter() - start_time) * 1000

        proactive = metadata.get("proactive_alert")
        updated_messages = None
        if payload.conversation_history:
            updated_messages = payload.conversation_history + [
                ConversationMessage(role="user", content=payload.question),
                ConversationMessage(role="assistant", content=result.get("answer", "")),
            ]

        cit_list = citations or metadata.get("source_citations") or []

        return QueryResponse(
            question=payload.question,
            answer=result.get("answer", ""),
            intent=intent,
            intents=intents,
            intent_confidence=top_conf,
            agents_invoked=intents,
            citations=list(cit_list),
            source_citations=list(cit_list),
            cached=bool(metadata.get("cached", False)),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
            conversation_history=updated_messages,
            filters=payload.filters,
            proactive_alert=proactive,
            answer_quality_score=metadata.get("answer_quality_score"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed request_id=%s", rid)
        raise HTTPException(status_code=500, detail=f"Query error: {exc}")


@app.post("/query/filtered", response_model=FilteredQueryResponse, tags=["Query"])
def query_filtered(request: FilteredQueryRequest, _: None = Depends(require_rag_available)):
    start_time = time.perf_counter()
    _validate_date(request.date_from)
    _validate_date(request.date_to)

    filters: dict[str, Any] = {}
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
            summary=result.get("summary", {}),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Filtered query failed")
        raise HTTPException(status_code=500, detail=f"Filtered query error: {exc}")


@app.post("/conversation", response_model=ConversationResponse, tags=["Conversation"])
def conversation(request: ConversationRequest, _: None = Depends(require_rag_available)):
    start_time = time.perf_counter()
    _apply_history(request.messages)
    try:
        result = orchestrator_module.ask(request.question)
        execution_time = (time.perf_counter() - start_time) * 1000
        updated_messages = request.messages + [
            ConversationMessage(role="user", content=request.question),
            ConversationMessage(role="assistant", content=result.get("answer", "")),
        ]

        meta = result.get("metadata", {})
        session_summary = None
        if len(updated_messages) >= 10:
            session_summary = meta.get("answer_quality_score") and "Conversation covers multiple financial intents."

        return ConversationResponse(
            answer=result.get("answer", ""),
            messages=updated_messages,
            metadata=meta,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
            session_summary=session_summary or meta.get("session_summary"),
            proactive_alert=meta.get("proactive_alert"),
        )
    except Exception as exc:
        logger.exception("Conversation error")
        raise HTTPException(status_code=500, detail=f"Conversation error: {exc}")


@app.get("/summary", response_model=SummaryResponse, tags=["Query"])
def get_summary():
    start_time = time.perf_counter()
    try:
        df = _load_df()
        df_anomalies = _get_cached_anomalies()
        anomaly_count = int(df_anomalies["is_anomaly"].sum())

        cat_col = _category_col(df)
        top_category = df.groupby(cat_col)["amount"].sum().idxmax()
        date_range = f"{df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}"

        execution_time = (time.perf_counter() - start_time) * 1000
        return SummaryResponse(
            total_transactions=len(df),
            total_spend=round(float(df["amount"].sum()), 2),
            top_category=str(top_category),
            avg_transaction=round(float(df["amount"].mean()), 2),
            anomaly_count=anomaly_count,
            date_range=date_range,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except Exception as exc:
        logger.exception("Summary error")
        raise HTTPException(status_code=500, detail=f"Summary error: {exc}")


@app.get("/categories", response_model=CategoriesResponse, tags=["Classifier"])
def get_categories():
    start_time = time.perf_counter()
    try:
        df = _load_df()
        cat_col = _category_col(df)
        breakdown = df.groupby(cat_col)["amount"].agg(["count", "sum", "mean"]).round(2).reset_index().rename(columns={cat_col: "category"})

        total_spend = df["amount"].sum()
        result = []
        for _, row in breakdown.iterrows():
            result.append(
                {
                    "category": row["category"],
                    "transaction_count": int(row["count"]),
                    "total_spend": round(float(row["sum"]), 2),
                    "avg_spend": round(float(row["mean"]), 2),
                    "percentage_of_total": round((row["sum"] / total_spend) * 100, 1),
                }
            )

        result = sorted(result, key=lambda x: x["total_spend"], reverse=True)
        execution_time = (time.perf_counter() - start_time) * 1000
        return CategoriesResponse(
            categories=result,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except Exception as exc:
        logger.exception("Category breakdown error")
        raise HTTPException(status_code=500, detail=f"Category breakdown error: {exc}")


@app.get("/anomalies", response_model=AnomaliesResponse, tags=["Anomalies"])
def get_anomalies(limit: int = Query(10, ge=1, le=100)):
    start_time = time.perf_counter()
    try:
        df_results = _get_cached_anomalies()

        anomalies = (
            df_results[df_results["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(limit)
        )

        result = []
        for _, row in anomalies.iterrows():
            result.append(
                {
                    "date": str(row["date"]),
                    "merchant": row["merchant"],
                    "category": row["category"],
                    "amount": round(float(row["amount"]), 2),
                    "anomaly_score": round(float(row["anomaly_score"]), 4),
                    "severity": row.get("severity"),
                    "explanation": row.get("anomaly_explanation", ""),
                }
            )

        execution_time = (time.perf_counter() - start_time) * 1000
        return AnomaliesResponse(
            total_anomalies=int(df_results["is_anomaly"].sum()),
            shown=len(result),
            anomalies=result,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except Exception as exc:
        logger.exception("Anomaly error")
        raise HTTPException(status_code=500, detail=f"Anomaly error: {exc}")


@app.get("/anomalies/report", response_model=AnomalyReportResponse, tags=["Anomalies"])
def anomaly_report():
    start_time = time.perf_counter()
    try:
        df_results = _get_cached_anomalies()
        report = weekly_anomaly_report(df_results)

        latest_date = df_results["date"].max()
        window_start = latest_date - pd.Timedelta(days=7)
        recent = df_results[(df_results["date"] >= window_start) & (df_results["is_anomaly"])]

        risk_tbl = merchant_risk_score(df_results)

        anomalies = []
        for row in recent.itertuples():
            risk_row = risk_tbl[risk_tbl["merchant"] == row.merchant]
            mrs = float(risk_row["risk_score"].iloc[0]) if not risk_row.empty else None
            anomalies.append(
                AnomalyDetail(
                    date=str(row.date),
                    merchant=row.merchant,
                    category=row.category,
                    amount=float(row.amount),
                    severity=row.severity,
                    explanation=row.anomaly_explanation,
                    anomaly_score=float(row.anomaly_score),
                    merchant_risk_score=mrs,
                )
            )

        execution_time = (time.perf_counter() - start_time) * 1000
        return AnomalyReportResponse(
            window_start=report["window_start"],
            window_end=report["window_end"],
            anomaly_count=report["anomaly_count"],
            total_anomaly_amount=report["total_anomaly_amount"],
            top_categories={str(k): int(v) for k, v in report["top_categories"].items()},
            anomalies=anomalies,
            merchant_risk=risk_tbl.head(20).to_dict("records"),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except Exception as exc:
        logger.exception("Anomaly report error")
        raise HTTPException(status_code=500, detail=f"Anomaly report error: {exc}")


@app.get("/anomalies/severity/{level}", response_model=AnomaliesResponse, tags=["Anomalies"])
def anomalies_by_severity(level: str):
    start_time = time.perf_counter()
    normalized = level.strip().lower()
    allowed = {"low", "medium", "high"}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail="Severity must be one of: low, medium, high")

    level_upper = normalized.upper()

    try:
        df_results = _get_cached_anomalies()
        filtered = df_results[(df_results["is_anomaly"]) & (df_results["severity"] == level_upper)]

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
            timestamp=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Anomaly severity error")
        raise HTTPException(status_code=500, detail=f"Anomaly severity error: {exc}")


@app.get("/anomalies/trends", response_model=AnomalyTrendsResponse, tags=["Anomalies"])
def anomalies_trends():
    start_time = time.perf_counter()
    df_results = _get_cached_anomalies()
    trend = anomaly_trend_analysis(df_results)
    execution_time = (time.perf_counter() - start_time) * 1000
    return AnomalyTrendsResponse(trend=trend, execution_time_ms=round(execution_time, 2), timestamp=_now_iso())


@app.get("/forecast", response_model=ForecastAPIResponse, tags=["Forecast"])
def forecast(
    category: Optional[str] = None,
    budget: Optional[str] = None,
    monthly_income: Optional[float] = Query(None, description="Optional monthly income for savings projection."),
):
    start_time = time.perf_counter()
    try:
        df = _load_df()
        cat_col = _category_col(df)
        categories = sorted(df[cat_col].dropna().unique())
        if category:
            if category not in categories:
                raise HTTPException(status_code=404, detail="Category not found")
            categories = [category]

        global_prior = float(df.groupby(df["date"].dt.date)["amount"].sum().mean())

        summaries: list[dict[str, Any]] = []
        all_changepoints: list[str] = []
        recurring_all: list[dict[str, Any]] = []

        for cat in categories:
            forecast_df, _ = forecast_spending(df, category=cat, days=30, global_prior_daily=global_prior)
            if forecast_df is None:
                continue
            summaries.append(get_forecast_summary(forecast_df, cat))
            all_changepoints.extend(forecast_df.attrs.get("changepoints", []))
            recurring_all.extend(detect_recurring_charges(df, category=cat))

        summary_df = combine_forecast_summaries(summaries)
        budgets = _parse_budget_param(budget)
        budget_alerts_df = check_budget_alerts(summary_df, budgets)
        budget_alerts = budget_alerts_df.to_dict("records") if not budget_alerts_df.empty else []

        alerts_set = {str(a.get("category")) for a in budget_alerts}
        budget_limit_map = {str(k): float(v) for k, v in budgets.items()}

        forecasts: list[ForecastResponse] = []
        for item in summary_df.to_dict("records"):
            cat_name = item["category"]
            forecasts.append(
                ForecastResponse(
                    category=cat_name,
                    total_forecast=float(item["total_predicted_spend"]),
                    avg_daily=float(item["avg_daily_spend"]),
                    max_day=float(item["max_daily_spend"]),
                    lower_bound=float(item.get("lower_bound", item.get("min_daily_spend", 0))),
                    upper_bound=float(item.get("upper_bound", item.get("max_daily_spend", 0))),
                    budget_alert=cat_name in alerts_set,
                    budget_limit=budget_limit_map.get(cat_name),
                    changepoints=list(item.get("changepoints", [])),
                    mae=item.get("mae_7d"),
                    rmse=item.get("rmse"),
                    mape=item.get("mape"),
                    recurring_charges=[r for r in recurring_all if r.get("category") == cat_name],
                )
            )

        total_forecast_sum = sum(f.total_forecast for f in forecasts)
        income = monthly_income if monthly_income is not None else float(os.getenv("FINSIGHT_MONTHLY_INCOME", "80000"))
        savings = savings_projection(total_forecast_sum, income, horizon_days=30)

        execution_time = (time.perf_counter() - start_time) * 1000
        return ForecastAPIResponse(
            forecasts=forecasts,
            budget_alerts=budget_alerts,
            changepoints=sorted(set(all_changepoints)),
            recurring_charges=recurring_all,
            savings_projection=savings,
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Forecast error")
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}")


@app.get("/forecast/whatif", response_model=WhatIfForecastResponse, tags=["Forecast"])
def forecast_whatif(category: str, change_pct: float):
    start_time = time.perf_counter()
    df = _load_df()
    global_prior = float(df.groupby(df["date"].dt.date)["amount"].sum().mean())
    forecast_df, _ = forecast_spending(df, category=category, days=30, global_prior_daily=global_prior)
    if forecast_df is None:
        raise HTTPException(status_code=404, detail="Unable to build forecast for category")

    wi = what_if_forecast(forecast_df, change_pct, category_label=category)
    execution_time = (time.perf_counter() - start_time) * 1000
    return WhatIfForecastResponse(
        category=category,
        change_pct=change_pct,
        adjusted_total_30d=float(wi["adjusted_total_30d"]),
        savings_impact_vs_base=float(wi["savings_impact_vs_base"]),
        execution_time_ms=round(execution_time, 2),
        timestamp=_now_iso(),
    )


@app.get("/classifier/uncertain", response_model=ClassifierUncertainResponse, tags=["Classifier"])
def classifier_uncertain():
    start_time = time.perf_counter()
    df = _load_df()

    if "confidence_score" in df.columns and "is_uncertain" in df.columns:
        uncertain = df[(df["confidence_score"] < 0.6) | df["is_uncertain"]]
    elif "confidence_score" in df.columns:
        uncertain = df[df["confidence_score"] < 0.6]
    elif "is_uncertain" in df.columns:
        uncertain = df[df["is_uncertain"]]
    else:
        uncertain = df.iloc[0:0]

    results = []
    for row in uncertain.itertuples():
        verdict = normalize_input_for_classification(str(row.merchant))
        kw_cat = None
        for needle in sorted(MERCHANT_KEYWORD_MAP.keys(), key=len, reverse=True):
            if needle in verdict:
                kw_cat = MERCHANT_KEYWORD_MAP[needle]
                break

        candidates = _get_top_candidates(row.merchant)
        conf_val = float(getattr(row, "confidence_score", 0.0)) if hasattr(row, "confidence_score") else 0.0
        pred_cat = getattr(row, "predicted_category", getattr(row, "category", ""))

        results.append(
            {
                "date": str(row.date),
                "merchant": row.merchant,
                "amount": round(float(row.amount), 2),
                "predicted_category": pred_cat,
                "confidence_score": round(conf_val, 4),
                "top_candidates": candidates,
                "keyword_map_verdict": kw_cat,
            }
        )

    execution_time = (time.perf_counter() - start_time) * 1000
    return ClassifierUncertainResponse(
        total_uncertain=len(results),
        transactions=results,
        execution_time_ms=round(execution_time, 2),
        timestamp=_now_iso(),
    )


@app.get("/classifier/stats", response_model=ClassifierStatsResponse, tags=["Classifier"])
def classifier_stats():
    start_time = time.perf_counter()
    df = _load_df()
    cat_col = "predicted_category" if "predicted_category" in df.columns else "category"
    stats = compute_category_spending_stats(df, category_column=cat_col)

    from agents.classifier import spending_velocity

    vel = spending_velocity(df, category_column=cat_col)

    execution_time = (time.perf_counter() - start_time) * 1000
    return ClassifierStatsResponse(
        stats=stats.to_dict("records"),
        velocity=vel.to_dict("records"),
        execution_time_ms=round(execution_time, 2),
        timestamp=_now_iso(),
    )


@app.post("/classifier/feedback", response_model=ClassifierFeedbackResponse, tags=["Classifier"])
def classifier_feedback(body: ClassifierFeedbackRequest):
    merchant = body.transaction_id
    try:
        df_lookup = _load_df()
        idx = int(body.transaction_id)
        if 0 <= idx < len(df_lookup):
            merchant = str(df_lookup.iloc[idx]["merchant"])
    except ValueError:
        pass

    fb = feedback_loop(correct_category=body.correct_category, merchant_name=str(merchant))
    return ClassifierFeedbackResponse(
        ok=True,
        cache_size=int(fb.get("cache_size", 0)),
        detail="Merchant cache updated.",
    )


@app.get("/insights", response_model=InsightsResponse, tags=["Insights"])
def insights(budget: Optional[str] = None):
    """Aggregate forecast, anomalies, classifier uncertainty, budgets, and peer benchmarks."""
    start_time = time.perf_counter()
    try:
        df = _load_df()
        df_anomalies = _get_cached_anomalies()

        spending_summary = {
            "total_transactions": int(len(df)),
            "total_spend": round(float(df["amount"].sum()), 2),
            "avg_transaction": round(float(df["amount"].mean()), 2),
        }

        top_anomalies_df = df_anomalies[df_anomalies["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(5)

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

        cat_col = _category_col(df)
        categories = sorted(df[cat_col].dropna().unique())
        global_prior = float(df.groupby(df["date"].dt.date)["amount"].sum().mean())

        summaries: list[dict[str, Any]] = []
        recurring_all = detect_recurring_charges(df)
        for cat in categories:
            forecast_df, _ = forecast_spending(df, category=cat, days=30, global_prior_daily=global_prior)
            if forecast_df is None:
                continue
            summaries.append(get_forecast_summary(forecast_df, cat))

        summary_df = combine_forecast_summaries(summaries)
        budgets = _parse_budget_param(budget)
        budget_alerts_df = check_budget_alerts(summary_df, budgets)
        budget_warnings = budget_alerts_df.to_dict("records") if not budget_alerts_df.empty else []

        alerts_set = {str(a.get("category")) for a in budget_warnings}
        forecast_list: list[ForecastResponse] = []
        for item in summary_df.to_dict("records"):
            cname = item["category"]
            forecast_list.append(
                ForecastResponse(
                    category=cname,
                    total_forecast=float(item["total_predicted_spend"]),
                    avg_daily=float(item["avg_daily_spend"]),
                    max_day=float(item["max_daily_spend"]),
                    lower_bound=float(item.get("lower_bound", 0)),
                    upper_bound=float(item.get("upper_bound", 0)),
                    budget_alert=cname in alerts_set,
                    changepoints=list(item.get("changepoints", [])),
                    mae=item.get("mae_7d"),
                    rmse=item.get("rmse"),
                    mape=item.get("mape"),
                    recurring_charges=[r for r in recurring_all if r.get("category") == cname],
                )
            )

        if "confidence_score" in df.columns and "is_uncertain" in df.columns:
            uncertain = df[(df["confidence_score"] < 0.6) | df["is_uncertain"]]
        elif "confidence_score" in df.columns:
            uncertain = df[df["confidence_score"] < 0.6]
        elif "is_uncertain" in df.columns:
            uncertain = df[df["is_uncertain"]]
        else:
            uncertain = df.iloc[0:0]

        uncertain_transactions = []
        for row in uncertain.itertuples():
            cs = float(getattr(row, "confidence_score", 0.0)) if hasattr(row, "confidence_score") else 0.0
            uncertain_transactions.append(
                {
                    "date": str(row.date),
                    "merchant": row.merchant,
                    "amount": round(float(row.amount), 2),
                    "predicted_category": getattr(row, "predicted_category", getattr(row, "category", "")),
                    "confidence_score": round(cs, 4),
                }
            )

        peer = peer_comparison(df)
        income = float(os.getenv("FINSIGHT_MONTHLY_INCOME", "80000"))
        fc_total = float(summary_df["total_predicted_spend"].sum()) if not summary_df.empty else 0.0
        savings = savings_projection(fc_total, income, horizon_days=30)

        execution_time = (time.perf_counter() - start_time) * 1000
        return InsightsResponse(
            spending_summary=spending_summary,
            top_anomalies=top_anomalies,
            forecast=forecast_list,
            uncertain_transactions=uncertain_transactions,
            budget_warnings=budget_warnings,
            peer_comparison=peer,
            recurring_charges=recurring_all,
            savings_projection=savings,
            generated_at=_now_iso(),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except Exception as exc:
        logger.exception("Insights error")
        raise HTTPException(status_code=500, detail=f"Insights error: {exc}")


@app.get("/insights/peer-comparison", response_model=PeerComparisonResponse, tags=["Insights"])
def insights_peer():
    start_time = time.perf_counter()
    df = _load_df()
    comparisons = peer_comparison(df)
    execution_time = (time.perf_counter() - start_time) * 1000
    return PeerComparisonResponse(
        benchmark_currency="KES",
        comparisons=comparisons,
        execution_time_ms=round(execution_time, 2),
        timestamp=_now_iso(),
    )


@app.post("/ingest/analyze", response_model=IngestAnalyzeResponse, tags=["Insights"])
async def ingest_and_analyze(file: UploadFile = File(...)):
    """Upload a transaction file and return categorized + anomaly-enriched analytics."""
    start_time = time.perf_counter()
    name = file.filename or "uploaded_file"
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            source_df = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

        tx_df = _normalize_uploaded_transactions(source_df)
        tx_df = clean_data(tx_df)
        if tx_df.empty:
            raise HTTPException(status_code=400, detail="No valid transactions found after cleaning.")

        predicted = []
        confidence = []
        for row in tx_df.itertuples():
            try:
                result = classify_transaction(str(row.merchant))
            except Exception:
                result = keyword_fallback_classify(normalize_input_for_classification(str(row.merchant)))
            predicted.append(result.get("category", "Other"))
            confidence.append(float(result.get("confidence", 0.0)))

        tx_df["predicted_category"] = predicted
        tx_df["confidence_score"] = confidence
        tx_df["category"] = tx_df["predicted_category"]

        stats = compute_category_spending_stats(tx_df, category_column="predicted_category")
        analyzed = detect_anomalies(tx_df)
        anomalies = analyzed[analyzed["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(30)

        execution_time = (time.perf_counter() - start_time) * 1000
        return IngestAnalyzeResponse(
            filename=name,
            total_rows=int(source_df.shape[0]),
            classified_rows=int(tx_df.shape[0]),
            anomaly_count=int(analyzed["is_anomaly"].sum()),
            categories=stats.to_dict("records"),
            anomalies=anomalies[
                ["date", "merchant", "category", "amount", "severity", "anomaly_score", "anomaly_explanation"]
            ].to_dict("records"),
            records=tx_df[
                ["date", "merchant", "amount", "predicted_category", "confidence_score"]
            ].head(400).to_dict("records"),
            execution_time_ms=round(execution_time, 2),
            timestamp=_now_iso(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingest analyze error")
        raise HTTPException(status_code=500, detail=f"Ingest analyze error: {exc}")

