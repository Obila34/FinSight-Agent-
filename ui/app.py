"""FinSight Gemini-style UI."""

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
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API POST {path} failed: {exc}")
        return None


def api_upload(path: str, uploaded_file) -> dict[str, Any] | None:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "text/csv")}
        response = requests.post(f"{API_BASE_URL}{path}", files=files, timeout=240)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API file upload failed: {exc}")
        return None


def render_header() -> None:
    st.markdown(
        """
        <style>
        .main-title {font-size: 2.0rem; font-weight: 700; margin-bottom: 0.2rem;}
        .sub-title {color: #8a8f98; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="main-title">FinSight Gemini Studio</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-title">Conversational finance copilot + document intelligence | API: {API_BASE_URL}</div>',
        unsafe_allow_html=True,
    )


def render_chat_tab() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.write("### Copilot Chat")
    st.caption("Use quick prompts or type your own financial question.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Monthly spend summary"):
        st.session_state.prefill_prompt = "Give me a monthly spending summary with top categories and anomalies."
    if c2.button("Budget risk scan"):
        st.session_state.prefill_prompt = "Analyze my budget risk and highlight overspending categories."
    if c3.button("Forecast next 30 days"):
        st.session_state.prefill_prompt = "Forecast my spending for the next 30 days with key drivers."

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask FinSight anything...")
    if not prompt and st.session_state.get("prefill_prompt"):
        prompt = st.session_state.pop("prefill_prompt")

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "question": prompt,
        "conversation_history": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]],
    }
    answer_data = api_post("/query", payload)
    answer_text = answer_data.get("answer", "No answer returned.") if answer_data else "No answer returned."

    with st.chat_message("assistant"):
        st.markdown(answer_text)
    st.session_state.messages.append({"role": "assistant", "content": answer_text})


def render_document_tab() -> None:
    st.write("### Financial Document Analyzer")
    st.caption("Upload a CSV of financial records. The agent cleans, categorizes, and flags anomalies automatically.")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if not uploaded:
        return

    if st.button("Analyze Document", type="primary"):
        result = api_upload("/ingest/analyze", uploaded)
        if not result:
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Rows uploaded", result.get("total_rows", 0))
        m2.metric("Rows classified", result.get("classified_rows", 0))
        m3.metric("Anomalies detected", result.get("anomaly_count", 0))

        categories = pd.DataFrame(result.get("categories", []))
        if not categories.empty:
            fig = px.bar(categories, x=categories.columns[0], y="total_spend", title="Categorized Spend")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(categories, use_container_width=True)

        anomalies = pd.DataFrame(result.get("anomalies", []))
        if not anomalies.empty:
            st.write("#### Top anomalies")
            st.dataframe(anomalies, use_container_width=True)

        records = pd.DataFrame(result.get("records", []))
        if not records.empty:
            st.write("#### Classified transactions")
            st.dataframe(records, use_container_width=True)
            csv_bytes = records.to_csv(index=False).encode("utf-8")
            st.download_button("Download classified CSV", data=csv_bytes, file_name="classified_records.csv", mime="text/csv")


def render_dashboard_tab() -> None:
    st.write("### Live Finance Dashboard")
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
        fig = px.pie(df, names="category", values="total_spend", title="Category Share")
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="FinSight Gemini Studio", layout="wide")
    render_header()
    tab1, tab2, tab3 = st.tabs(["Copilot", "Document Intelligence", "Dashboard"])
    with tab1:
        render_chat_tab()
    with tab2:
        render_document_tab()
    with tab3:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
