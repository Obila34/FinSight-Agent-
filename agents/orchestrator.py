from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.anomaly_detector import detect_anomalies, weekly_anomaly_report
from agents.classifier import classify_transaction
from agents.forecaster import (
    combine_forecast_summaries,
    forecast_spending,
    get_forecast_summary,
    check_budget_alerts,
)
logger = logging.getLogger(__name__)

_INTENT_PATTERNS: list[tuple[str, list[tuple[str, float]]]] = [
    (
        "forecast",
        [
            ("forecast", 1.0),
            ("predict", 1.1),
            ("future", 0.9),
            ("next month", 1.2),
            ("next week", 1.1),
            ("budget", 0.7),
        ],
    ),
    (
        "anomaly",
        [
            ("anomaly", 1.3),
            ("anomalies", 1.3),
            ("outlier", 1.2),
            ("unusual", 1.2),
            ("suspicious", 1.4),
            ("fraud", 1.1),
            ("high severity", 1.4),
        ],
    ),
    (
        "classify",
        [
            ("classify", 1.4),
            ("categorise", 1.2),
            ("categorize", 1.2),
            ("merchant category", 1.3),
        ],
    ),
    (
        "rag",
        [
            ("how much", 1.2),
            ("spend", 1.0),
            ("spent", 1.0),
            ("merchant", 0.9),
            ("transaction", 0.9),
            ("summary", 1.1),
            ("average", 0.9),
            ("kes", 0.8),
            ("purchase", 0.8),
            ("breakdown", 1.45),
            ("categories", 1.15),
            ("by category", 1.25),
            ("total spending", 1.2),
        ],
    ),
]

_CACHE_TTL_SECONDS = 300
_RESULT_CACHE: dict[str, dict[str, Any]] = {}

_CONVERSATION_HISTORY: deque[dict[str, Any]] = deque(maxlen=80)
_GRAPH: Any = None
_RAG_ENGINE: Any = None
_CLASSIFIED_DF: pd.DataFrame | None = None

_CIRCUIT_FAILURES: defaultdict[str, int] = defaultdict(int)
_CIRCUIT_OPEN: dict[str, bool] = defaultdict(bool)

_AGENT_SUCCESS_SCORES: defaultdict[str, float] = defaultdict(float)


def reset_circuit(agent: str) -> None:
    _CIRCUIT_OPEN[agent] = False
    _CIRCUIT_FAILURES[agent] = 0


def register_failure(agent: str) -> None:
    _CIRCUIT_FAILURES[agent] += 1
    if _CIRCUIT_FAILURES[agent] >= 3:
        _CIRCUIT_OPEN[agent] = True
        logger.error("Circuit breaker OPEN for agent %s after repeated failures.", agent)


def register_success(agent: str) -> None:
    reset_circuit(agent)


def circuit_open(agent: str) -> bool:
    return bool(_CIRCUIT_OPEN.get(agent))


def _load_classified_df() -> pd.DataFrame:
    global _CLASSIFIED_DF
    if _CLASSIFIED_DF is None:
        _CLASSIFIED_DF = pd.read_csv("data/classified_transactions.csv")
        _CLASSIFIED_DF["date"] = pd.to_datetime(_CLASSIFIED_DF["date"])
    return _CLASSIFIED_DF


def _get_rag_engine() -> Any:
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        from agents.rag_engine import FinanceRAGEngine

        _RAG_ENGINE = FinanceRAGEngine(data_path="data/classified_transactions.csv")
    return _RAG_ENGINE


def detect_intents_scored(query: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    lowered = query.lower()
    aggregate: dict[str, float] = defaultdict(float)

    for intent, weighted_terms in _INTENT_PATTERNS:
        for phrase, weight in weighted_terms:
            if phrase in lowered:
                aggregate[intent] += weight

        if intent == "forecast" and re.search(r"\b\d{4}-\d{2}-\d{2}\b", query):
            aggregate[intent] += 0.4

    for intent in list(aggregate.keys()):
        aggregate[intent] += _AGENT_SUCCESS_SCORES.get(intent, 0.0) * 0.05

    ranked = sorted(aggregate.items(), key=lambda kv: kv[1], reverse=True)
    max_score = ranked[0][1] if ranked else 1.0

    structured: list[dict[str, Any]] = []
    confidence_map: dict[str, float] = {}
    for name, score in ranked:
        conf = round(min(1.0, float(score) / float(max_score + 1e-9)), 4)
        structured.append({"intent": name, "score": round(score, 4), "confidence": conf})
        confidence_map[name] = conf

    return structured, confidence_map


def detect_intents(query: str) -> list[dict[str, Any]]:
    structured, _ = detect_intents_scored(query)
    return structured


def _extract_budget_limits(query: str) -> dict[str, float]:
    if "{" not in query or "}" not in query:
        return {}

    start = query.find("{")
    end = query.rfind("}")
    try:
        budget_limits = json.loads(query[start : end + 1])
        if isinstance(budget_limits, dict):
            return {str(k): float(v) for k, v in budget_limits.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Failed to parse budget limits from query")

    return {}


def _extract_categories(query: str, categories: list[str]) -> list[str]:
    lowered = query.lower()
    return [category for category in categories if category.lower() in lowered]


def _run_forecast_agent(query: str) -> dict[str, Any]:
    df = _load_classified_df()
    categories = sorted(df["category"].unique())
    selected = _extract_categories(query, categories) or categories
    summaries: list[dict[str, Any]] = []

    all_daily_prior = df.groupby(pd.to_datetime(df["date"]).dt.date)["amount"].sum().mean()

    for category in selected:
        forecast, _model = forecast_spending(df, category=category, days=30, global_prior_daily=float(all_daily_prior))
        if forecast is None:
            continue
        summaries.append(get_forecast_summary(forecast, category))

    summary_df = combine_forecast_summaries(summaries)
    budgets = _extract_budget_limits(query)
    budget_alerts = check_budget_alerts(summary_df, budgets)

    answer = summary_df.to_string(index=False) if not summary_df.empty else "No forecast data available."
    if not budget_alerts.empty:
        answer += "\n\nBudget alerts:\n" + budget_alerts.to_string(index=False)

    return {"answer": answer, "summary": summary_df.to_dict("records")}


def _run_anomaly_agent(query: str) -> dict[str, Any]:
    df = _load_classified_df()
    category_filters = _extract_categories(query, sorted(df["category"].unique()))
    analysis_df = df[df["category"].isin(category_filters)] if category_filters else df

    results = detect_anomalies(analysis_df)
    anomalies = results[results["is_anomaly"]].copy()
    weekly_report = weekly_anomaly_report(results)

    top_anomalies = anomalies.sort_values("anomaly_score", ascending=False).head(5)
    answer = top_anomalies[["date", "merchant", "category", "amount", "severity"]].to_string(index=False)

    return {"answer": answer, "weekly_report": weekly_report, "anomaly_count": int(anomalies.shape[0])}


def _run_classify_agent(query: str) -> dict[str, Any]:
    lowered = query.lower()
    merchant = query

    for keyword in ["classify", "category", "categorize", "categorise"]:
        if keyword in lowered:
            merchant = query.lower().replace(keyword, "").strip()
            break

    if not merchant:
        return {"answer": "Provide a merchant name to classify."}

    result = classify_transaction(merchant)
    answer = f"{merchant} → {result['category']} (confidence: {result['confidence']})"
    if result.get("is_uncertain"):
        answer += " [Uncertain]"

    return {"answer": answer, "classification": result}


def _run_rag_agent(query: str) -> dict[str, Any]:
    engine = _get_rag_engine()
    result = engine.ask_with_summary(query)
    return {
        "answer": result["answer"],
        "summary": result["summary"],
        "citations": result["citations"],
        "insights": result.get("insights", []),
    }


def proactive_alert_payload() -> str | None:
    parts: list[str] = []

    df = _load_classified_df()
    alerts = detect_anomalies(df)
    highs = alerts[(alerts["is_anomaly"]) & (alerts["severity"] == "HIGH")]
    if not highs.empty:
        parts.append(f"HIGH severity anomaly flagged on {len(highs)} transactions — review recent large spends.")

    return " ".join(parts) if parts else None


def _score_response_quality(answer: str, citations: list | None, summaries: list | None) -> float:
    score = 0.4
    if len(answer) > 120:
        score += 0.25
    if citations:
        score += min(0.25, 0.05 * len(citations))
    if summaries:
        score += 0.1
    return round(min(score, 1.0), 3)


def session_summary_block() -> str | None:
    if len(_CONVERSATION_HISTORY) < 5:
        return None

    recent = list(_CONVERSATION_HISTORY)[-8:]
    themes = []
    for turn in recent:
        ans = turn.get("answer", "")
        themes.append(ans.split("\n")[0][:160])

    digest = " ".join(themes)
    return digest[:900]


def dedupe_response_text(text: str) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        norm = line.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            out_lines.append(line)
    return "\n".join(out_lines)


def _route_intents(state: dict[str, Any]) -> dict[str, Any]:
    structured, confidence_map = detect_intents_scored(state["query"])

    if not structured:
        state["intents"] = ["rag"]
        state["intent_confidence"] = {"rag": 0.1}
        state["fallback_reason"] = "no_intent_match"
        return state

    top = structured[0]["intent"]
    top_score = structured[0]["score"]
    second_score = structured[1]["score"] if len(structured) > 1 else 0.0

    intents_to_run = [top]
    if len(structured) > 1 and abs(top_score - second_score) <= 0.15 * max(top_score, 1e-9):
        intents_to_run.append(structured[1]["intent"])

    max_confidence = confidence_map.get(top, 0.0)
    if max_confidence < 0.35:
        logger.warning("Low intent confidence (%.4f). Falling back to RAG.", max_confidence)
        state["intents"] = ["rag"]
        state["intent_confidence"] = {"rag": max_confidence}
        state["fallback_reason"] = "low_intent_confidence"
        return state

    state["intents"] = intents_to_run
    state["intent_confidence"] = confidence_map
    state["fallback_reason"] = None
    return state


def _execute_agent(intent: str, query: str) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()

    if circuit_open(intent):
        logger.warning("Circuit open for %s — skipping.", intent)
        return (
            {"answer": f"[{intent.upper()} temporarily unavailable — circuit breaker active]", "circuit_skip": True},
            time.perf_counter() - start,
        )

    runners = {
        "forecast": _run_forecast_agent,
        "anomaly": _run_anomaly_agent,
        "classify": _run_classify_agent,
        "rag": _run_rag_agent,
    }

    fn = runners.get(intent)
    if fn is None:
        return {"answer": ""}, time.perf_counter() - start

    try:
        payload = fn(query)
        register_success(intent)
        _AGENT_SUCCESS_SCORES[intent] += 0.05
        return payload, time.perf_counter() - start
    except Exception:
        logger.exception("Agent %s failed", intent)
        register_failure(intent)
        return {"answer": f"[{intent.upper()} failed — routed to degraded mode]", "error": True}, time.perf_counter() - start


def _run_agents(state: dict[str, Any]) -> dict[str, Any]:
    responses: dict[str, Any] = {}
    timings: dict[str, float] = {}

    for intent in state["intents"]:
        payload, elapsed = _execute_agent(intent, state["query"])
        responses[intent] = payload
        timings[intent] = round(elapsed, 4)

    state["responses"] = responses
    state["timings"] = timings
    return state


def _compose_response(state: dict[str, Any]) -> dict[str, Any]:
    parts: list[str] = []
    all_citations: list[dict[str, Any]] = []

    for intent in state["intents"]:
        result = state["responses"].get(intent, {})
        label = intent.upper()
        answer_piece = result.get("answer", "")
        parts.append(f"[{label}]\n{answer_piece}")
        if intent == "rag":
            all_citations.extend(result.get("citations", []))

    combined = dedupe_response_text("\n\n".join(parts))

    rag_blob = state["responses"].get("rag", {})
    quality = _score_response_quality(
        rag_blob.get("answer", combined),
        rag_blob.get("citations"),
        None,
    )

    if quality < 0.6 and "rag" in state["responses"]:
        logger.warning("Low answer quality %.3f — retrying RAG with tightened prompt.", quality)
        retry = _run_rag_agent(state["query"] + " Provide precise numerical totals referencing transactions.")
        state["responses"]["rag"] = retry
        combined_retry = retry.get("answer", combined)
        combined = dedupe_response_text(combined_retry if combined_retry else combined)
        quality = max(quality, _score_response_quality(combined_retry, retry.get("citations"), None))

    alert = proactive_alert_payload()
    summary_prefix = session_summary_block()

    prefixes: list[str] = []
    if alert:
        prefixes.append(f"Alert: {alert}")
    if summary_prefix:
        prefixes.append(f"Session summary: {summary_prefix}")

    if prefixes:
        combined = "\n\n".join(prefixes + [combined])

    metadata = {
        "timings": state.get("timings", {}),
        "intents": state.get("intents", []),
        "intent_confidence": state.get("intent_confidence", {}),
        "fallback_reason": state.get("fallback_reason"),
        "answer_quality_score": quality,
        "source_citations": all_citations,
        "proactive_alert": alert,
        "session_summary": summary_prefix,
    }

    state["final_answer"] = combined
    state["metadata"] = metadata
    return state


def _build_graph():
    graph = StateGraph(dict)
    graph.add_node("route_intents", _route_intents)
    graph.add_node("run_agents", _run_agents)
    graph.add_node("compose_response", _compose_response)

    graph.set_entry_point("route_intents")
    graph.add_edge("route_intents", "run_agents")
    graph.add_edge("run_agents", "compose_response")
    graph.add_edge("compose_response", END)

    return graph.compile()


def ask(query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    cache_key = hashlib.sha256(f"{query}|{json.dumps(filters or {}, sort_keys=True)}".encode()).hexdigest()
    cache_entry = _RESULT_CACHE.get(cache_key)
    if cache_entry and (time.time() - cache_entry["timestamp"]) <= _CACHE_TTL_SECONDS:
        logger.warning("Returning cached orchestrator response.")
        cached = dict(cache_entry["result"])
        meta = cached.setdefault("metadata", {})
        meta["cached"] = True
        return cached

    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()

    state = {"query": query, "filters": filters or {}, "history": list(_CONVERSATION_HISTORY)}
    result_state = _GRAPH.invoke(state)

    response = {
        "answer": result_state.get("final_answer", ""),
        "metadata": result_state.get("metadata", {}),
        "responses": result_state.get("responses", {}),
    }
    response["metadata"]["cached"] = False

    _CONVERSATION_HISTORY.append(
        {
            "query": query,
            "answer": response["answer"],
            "metadata": response["metadata"],
        }
    )

    _RESULT_CACHE[cache_key] = {"timestamp": time.time(), "result": response}
    return response


def export_openapi_stub() -> dict[str, Any]:
    return {"name": "FinSightOrchestrator", "version": "2.1.0"}
