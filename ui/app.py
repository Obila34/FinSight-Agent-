"""FinSight Streamlit UI."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("FINSIGHT_API_URL", "http://127.0.0.1:8000")


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API GET {path} failed: {exc}")
        return None


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API POST {path} failed: {exc}")
        return None


def render_dashboard() -> None:
    st.subheader("Overview")
    summary = api_get("/summary")
    if not summary:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", summary.get("total_transactions", 0))
    c2.metric("Total Spend", f"KES {summary.get('total_spend', 0):,.2f}")
    c3.metric("Top Category", summary.get("top_category", "N/A"))
    c4.metric("Anomalies", summary.get("anomaly_count", 0))

    categories = api_get("/categories")
    if categories and categories.get("categories"):
        df = pd.DataFrame(categories["categories"])
        fig = px.bar(df, x="category", y="total_spend", title="Spend by Category", text="total_spend")
        st.plotly_chart(fig, use_container_width=True)


def render_chat() -> None:
    st.subheader("Assistant Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask FinSight anything about your spending...")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    payload = {
        "question": user_prompt,
        "conversation_history": [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
        ],
    }
    answer_data = api_post("/query", payload)
    answer_text = answer_data.get("answer", "No answer returned.") if answer_data else "No answer returned."

    with st.chat_message("assistant"):
        st.markdown(answer_text)
    st.session_state.messages.append({"role": "assistant", "content": answer_text})


def render_categories() -> None:
    st.subheader("Category Analysis")
    payload = api_get("/categories")
    if not payload:
        return
    rows = payload.get("categories", [])
    if not rows:
        st.warning("No category data available.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    fig = px.pie(df, names="category", values="total_spend", title="Category Share")
    st.plotly_chart(fig, use_container_width=True)


def render_anomalies() -> None:
    st.subheader("Anomalies")
    payload = api_get("/anomalies", params={"limit": 50})
    if not payload:
        return

    anomalies = payload.get("anomalies", [])
    if not anomalies:
        st.info("No anomalies found.")
        return

    df = pd.DataFrame(anomalies)
    st.dataframe(df, use_container_width=True)
    severity_counts = df["severity"].fillna("UNKNOWN").value_counts().reset_index()
    severity_counts.columns = ["severity", "count"]
    fig = px.bar(severity_counts, x="severity", y="count", title="Anomaly Severity Distribution")
    st.plotly_chart(fig, use_container_width=True)


def render_forecast() -> None:
    st.subheader("Forecast")
    payload = api_get("/forecast")
    if not payload:
        return

    forecasts = payload.get("forecasts", [])
    if not forecasts:
        st.warning("No forecast data returned.")
        return

    df = pd.DataFrame(forecasts)
    st.dataframe(df, use_container_width=True)
    fig = px.bar(df, x="category", y="total_forecast", title="30-Day Forecast by Category")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="FinSight Agent", layout="wide")
    st.title("FinSight Agent")
    st.caption(f"API: {API_BASE_URL}")

    pages = ["Dashboard", "Chat", "Categories", "Anomalies", "Forecast"]
    page = st.sidebar.radio("Navigate", pages)

    if page == "Dashboard":
        render_dashboard()
    elif page == "Chat":
        render_chat()
    elif page == "Categories":
        render_categories()
    elif page == "Anomalies":
        render_anomalies()
    else:
        render_forecast()


if __name__ == "__main__":
    main()
