from collections import deque
import json
import logging
import time
from typing import Any

import pandas as pd
from langgraph.graph import StateGraph, END

from agents.classifier import classify_transaction
from agents.forecaster import forecast_spending, get_forecast_summary, combine_forecast_summaries, check_budget_alerts
from agents.anomaly_detector import detect_anomalies, weekly_anomaly_report
from agents.rag_engine import FinanceRAGEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

_INTENT_KEYWORDS = {
    "forecast": ["forecast", "forecasting", "predict", "future", "next", "budget"],
    "anomaly": ["anomaly", "anomalies", "outlier", "outliers", "suspicious", "unusual"],
    "classify": ["classify", "category", "categorize"],
    "rag": ["how much", "spend", "transaction", "merchant", "summary"],
}

_CACHE_TTL_SECONDS = 300
_RESULT_CACHE: dict[str, dict] = {}
_CONVERSATION_HISTORY = deque(maxlen=5)
_GRAPH = None
_RAG_ENGINE = None
_CLASSIFIED_DF = None


def _get_rag_engine() -> FinanceRAGEngine:
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        _RAG_ENGINE = FinanceRAGEngine(data_path="data/classified_transactions.csv")
    return _RAG_ENGINE


def _load_classified_df() -> pd.DataFrame:
    global _CLASSIFIED_DF
    if _CLASSIFIED_DF is None:
        _CLASSIFIED_DF = pd.read_csv("data/classified_transactions.csv")
        _CLASSIFIED_DF['date'] = pd.to_datetime(_CLASSIFIED_DF['date'])
    return _CLASSIFIED_DF


def detect_intents(query: str) -> list:
    """
    Detects multiple intents in a query using keyword heuristics.

    Returns:
        list of intent dicts with confidence scores.
    """

    lowered = query.lower()
    intents = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        confidence = min(1.0, hits / max(1, len(keywords) // 2))
        if hits > 0:
            intents.append({"intent": intent, "confidence": round(confidence, 2)})

    intents.sort(key=lambda item: item["confidence"], reverse=True)
    return intents


def _extract_budget_limits(query: str) -> dict:
    if "{" not in query or "}" not in query:
        return {}

    start = query.find("{")
    end = query.rfind("}")
    try:
        budget_limits = json.loads(query[start:end + 1])
        if isinstance(budget_limits, dict):
            return budget_limits
    except json.JSONDecodeError:
        logger.warning("Failed to parse budget limits from query")

    return {}


def _extract_categories(query: str, categories: list) -> list:
    lowered = query.lower()
    return [category for category in categories if category.lower() in lowered]


def _run_forecast_agent(query: str) -> dict:
    df = _load_classified_df()
    categories = sorted(df['category'].unique())
    selected = _extract_categories(query, categories) or categories
    summaries = []

    for category in selected:
        forecast, model = forecast_spending(df, category=category, days=30)
        if forecast is None:
            continue
        summaries.append(get_forecast_summary(forecast, category))

    summary_df = combine_forecast_summaries(summaries)
    budgets = _extract_budget_limits(query)
    budget_alerts = check_budget_alerts(summary_df, budgets)

    answer = summary_df.to_string(index=False) if not summary_df.empty else "No forecast data available."
    if not budget_alerts.empty:
        answer += "\n\nBudget alerts:\n" + budget_alerts.to_string(index=False)

    return {
        "answer": answer,
        "summary": summary_df.to_dict('records')
    }


def _run_anomaly_agent(query: str) -> dict:
    df = _load_classified_df()
    category_filters = _extract_categories(query, sorted(df['category'].unique()))
    analysis_df = df[df['category'].isin(category_filters)] if category_filters else df

    results = detect_anomalies(analysis_df)
    anomalies = results[results['is_anomaly'] == True].copy()
    weekly_report = weekly_anomaly_report(results)

    top_anomalies = anomalies.sort_values('anomaly_score', ascending=False).head(5)
    answer = top_anomalies[['date', 'merchant', 'category', 'amount', 'severity']].to_string(index=False)

    return {
        "answer": answer,
        "weekly_report": weekly_report,
        "anomaly_count": int(anomalies.shape[0])
    }


def _run_classify_agent(query: str) -> dict:
    lowered = query.lower()
    merchant = query

    for keyword in ["classify", "category", "categorize"]:
        if keyword in lowered:
            merchant = query.lower().replace(keyword, "").strip()
            break

    if not merchant:
        return {"answer": "Provide a merchant name to classify."}

    result = classify_transaction(merchant)
    answer = f"{merchant} -> {result['category']} (confidence: {result['confidence']})"
    if result.get('is_uncertain'):
        answer += " [Uncertain]"

    return {
        "answer": answer,
        "classification": result
    }


def _run_rag_agent(query: str) -> dict:
    engine = _get_rag_engine()
    result = engine.ask_with_summary(query)
    return {
        "answer": result['answer'],
        "summary": result['summary'],
        "citations": result['citations']
    }


def _route_intents(state: dict) -> dict:
    intents = detect_intents(state['query'])
    if not intents:
        intents = [{"intent": "rag", "confidence": 0.0}]

    max_confidence = intents[0]["confidence"]
    if max_confidence < 0.35:
        logger.warning("Low intent confidence (%.2f). Falling back to RAG.", max_confidence)
        state['intents'] = ["rag"]
        state['intent_confidence'] = {"rag": max_confidence}
        state['fallback_reason'] = "low_intent_confidence"
        return state

    state['intents'] = [item["intent"] for item in intents]
    state['intent_confidence'] = {item["intent"]: item["confidence"] for item in intents}
    return state


def _run_agents(state: dict) -> dict:
    responses = {}
    timings = {}

    for intent in state['intents']:
        start = time.perf_counter()
        if intent == "forecast":
            responses[intent] = _run_forecast_agent(state['query'])
        elif intent == "anomaly":
            responses[intent] = _run_anomaly_agent(state['query'])
        elif intent == "classify":
            responses[intent] = _run_classify_agent(state['query'])
        else:
            responses[intent] = _run_rag_agent(state['query'])
        timings[intent] = round(time.perf_counter() - start, 4)

    state['responses'] = responses
    state['timings'] = timings
    return state


def _compose_response(state: dict) -> dict:
    parts = []
    for intent in state['intents']:
        result = state['responses'].get(intent, {})
        label = intent.upper()
        parts.append(f"[{label}]\n{result.get('answer', '')}")

    answer = "\n\n".join(parts)
    metadata = {
        "timings": state.get('timings', {}),
        "intents": state.get('intents', []),
        "intent_confidence": state.get('intent_confidence', {}),
        "fallback_reason": state.get('fallback_reason')
    }

    state['final_answer'] = answer
    state['metadata'] = metadata
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


def ask(query: str) -> dict:
    """
    Main entry point for the FinSight orchestrator.

    Args:
        query: user question.

    Returns:
        dict containing answer and metadata.
    """

    cache_entry = _RESULT_CACHE.get(query)
    if cache_entry and (time.time() - cache_entry['timestamp']) <= _CACHE_TTL_SECONDS:
        logger.info("Returning cached response for query")
        return cache_entry['result']

    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()

    state = {
        "query": query,
        "history": list(_CONVERSATION_HISTORY)
    }

    result_state = _GRAPH.invoke(state)
    response = {
        "answer": result_state.get('final_answer', ''),
        "metadata": result_state.get('metadata', {}),
        "responses": result_state.get('responses', {})
    }

    _CONVERSATION_HISTORY.append({
        "query": query,
        "answer": response['answer'],
        "metadata": response['metadata']
    })

    _RESULT_CACHE[query] = {
        "timestamp": time.time(),
        "result": response
    }

    return response


if __name__ == "__main__":
    sample_queries = [
        "Are there anomalies in my food spending and what's the forecast?",
        "Classify Uber transaction",
        "How much did I spend on transport?"
    ]

    for query in sample_queries:
        result = ask(query)
        print("\nQuery:", query)
        print("Answer:\n", result['answer'])
        print("Metadata:", result['metadata'])
