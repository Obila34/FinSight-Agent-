from __future__ import annotations

import logging
import math
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

try:
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
except ImportError:
    MetadataFilter = None  # type: ignore[misc, assignment]
    MetadataFilters = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIED_PATH = BASE_DIR / "data" / "classified_transactions.csv"
DEFAULT_CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

load_dotenv()


def _get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


FINANCIAL_KEYWORDS = frozenset(
    {
        "spend",
        "spent",
        "budget",
        "transaction",
        "merchant",
        "money",
        "payment",
        "purchase",
        "shopping",
        "bill",
        "forecast",
        "average",
        "category",
        "credit",
        "debit",
        "kes",
        "income",
        "savings",
        "finance",
        "financial",
        "cash",
        "daily",
        "weekly",
        "monthly",
    }
)


def setup_models() -> bool:
    api_key = _get_openai_api_key()
    if not api_key:
        logger.warning("OpenAI API key not set. Set OPENAI_API_KEY to enable RAG.")
        return False

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    logger.info("OpenAI models configured")
    return True


def transactions_to_documents(df: pd.DataFrame) -> list:
    documents = []
    for i, row in df.iterrows():
        date_str = pd.to_datetime(row["date"]).strftime("%d %B %Y")
        dow = pd.to_datetime(row["date"]).day_name()
        text = (
            f"On {date_str}, there was a {row['category']} transaction "
            f"of KES {row['amount']:,.2f} at {row['merchant']}. "
            f"This was on a {dow}. "
            f"Category: {row['category']}. "
            f"Amount: KES {row['amount']:,.2f}."
        )
        metadata = {
            "date": str(row["date"]),
            "merchant": str(row["merchant"]),
            "amount": float(row["amount"]),
            "category": str(row["category"]),
            "transaction_id": i,
        }
        documents.append(Document(text=text, metadata=metadata))

    logger.info("Created %s documents", len(documents))
    return documents


def build_vector_store(documents: list, persist_dir: str = str(DEFAULT_CHROMA_DIR)) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("transactions")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    logger.info("Vector store built and saved")
    return index


def load_vector_store(persist_dir: str = str(DEFAULT_CHROMA_DIR)) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("transactions")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    logger.info("Vector store loaded from disk")
    return index


def build_metadata_filters(filters: dict | None) -> Any:
    if not filters or MetadataFilters is None:
        return None

    metadata_filters = []

    category = filters.get("category") if isinstance(filters, dict) else None
    if category:
        metadata_filters.append(MetadataFilter(key="category", operator="==", value=category))

    start_date = filters.get("start_date") if isinstance(filters, dict) else None
    end_date = filters.get("end_date") if isinstance(filters, dict) else None

    if start_date:
        metadata_filters.append(MetadataFilter(key="date", operator=">=", value=str(start_date)))
    if end_date:
        metadata_filters.append(MetadataFilter(key="date", operator="<=", value=str(end_date)))

    if not metadata_filters:
        return None

    return MetadataFilters(filters=metadata_filters)


def apply_filters_to_df(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return df

    filtered = df.copy()
    if "category" in filters:
        filtered = filtered[filtered["category"] == filters["category"]]

    if "start_date" in filters:
        filtered = filtered[pd.to_datetime(filtered["date"]) >= pd.to_datetime(filters["start_date"])]

    if "end_date" in filters:
        filtered = filtered[pd.to_datetime(filtered["date"]) <= pd.to_datetime(filters["end_date"])]

    return filtered


def compute_spending_summary(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {"total": 0.0, "average": 0.0, "max": 0.0, "min": 0.0, "count": 0}

    return {
        "total": float(df["amount"].sum()),
        "average": float(df["amount"].mean()),
        "max": float(df["amount"].max()),
        "min": float(df["amount"].min()),
        "count": int(df.shape[0]),
    }


def expand_query(question: str) -> str:
    """Append 2–3 related lexical variants to improve recall."""
    base = question.strip()
    synonyms: list[str] = []
    lowered = base.lower()

    if "food" in lowered:
        synonyms.extend(["groceries", "restaurants"])
    if "transport" in lowered:
        synonyms.extend(["uber", "fuel"])
    if "spent" in lowered or "spend" in lowered:
        synonyms.extend(["payments", "transactions"])

    seen = set()
    parts = [base]
    for term in synonyms[:3]:
        if term not in seen:
            parts.append(term)
            seen.add(term)

    return " ".join(parts)


def intent_clarification(question: str) -> str | None:
    tokens = re.findall(r"\w+", question.lower())
    word_count = len(question.split())

    finance_hit = any(w in FINANCIAL_KEYWORDS for w in tokens)

    if word_count < 5 or not finance_hit:
        return (
            "Could you clarify what financial detail you need "
            "(for example spending by category, time range, or merchant)?"
        )
    return None


def rerank_nodes(nodes: list[Any], decay_days: float = 45.0) -> list[Any]:
    """Re-rank retrieved nodes by similarity × recency weight."""

    def score(node: Any) -> float:
        meta = getattr(node, "metadata", {}) or {}
        score_base = getattr(node, "score", None) or 1.0
        try:
            d = pd.to_datetime(meta.get("date"))
            days_ago = max(0.0, (datetime.now() - d.to_pydatetime()).days)
        except Exception:
            days_ago = 30.0
        recency = math.exp(-days_ago / decay_days)
        return float(score_base) * recency

    return sorted(nodes, key=score, reverse=True)


def answer_confidence_from_nodes(nodes: list[Any]) -> float:
    if not nodes:
        return 0.0
    categories = [((getattr(n, "metadata", {}) or {}).get("category")) for n in nodes]
    top = max(categories.count(c) for c in set(categories)) if categories else 0
    agreement_ratio = top / len(categories)
    return round(min(1.0, 0.35 + 0.65 * agreement_ratio), 3)


def proactive_insights(summary: dict[str, float | int], question: str) -> list[str]:
    insights: list[str] = []
    if summary.get("count", 0) > 50 and summary.get("average", 0) > 3000:
        insights.append("Your average ticket size is elevated versus typical retail spend — review large recurring bills.")
    if "transport" in question.lower():
        insights.append("Consider comparing weekly versus weekend transport spikes to optimise commute spend.")
    if len(insights) < 2 and summary.get("max", 0) > 5 * summary.get("average", 1):
        insights.append("One transaction dominates your filtered window — validate it was intentional.")
    return insights[:2]


def compress_old_memory_turns(memory_list: list[dict[str, str]]) -> str:
    """Flatten older turns into a compact digest for token control."""
    if len(memory_list) <= 5:
        return ""

    older = memory_list[:-5]
    snippets = [f"{item['question'][:80]} → {item['answer'][:120]}" for item in older[-10:]]
    return "Earlier context digest: " + " | ".join(snippets)


def multi_hop_plan(question: str) -> list[str]:
    lowered = question.lower()
    subqueries = [question]
    if " and " in lowered:
        parts = [p.strip() for p in question.split(" and ") if p.strip()]
        if len(parts) > 1:
            subqueries = parts
    if "compare" in lowered and " vs " in lowered:
        subqueries = [s.strip() for s in question.split(" vs ") if s.strip()]
    return subqueries[:4]


class FinanceRAGEngine:
    def __init__(self, data_path: str = str(DEFAULT_CLASSIFIED_PATH), persist_dir: str = str(DEFAULT_CHROMA_DIR)):
        self.openai_ready = setup_models()
        self.persist_dir = persist_dir
        self.data_path = data_path
        self.memory: deque[dict[str, str]] = deque(maxlen=5)
        self.full_turn_history: list[dict[str, str]] = []

        self.index: VectorStoreIndex | None = None
        if self.openai_ready:
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                logger.info("Loading existing vector store from disk")
                self.index = load_vector_store(persist_dir)
            else:
                logger.info("Building vector store from scratch")
                df = pd.read_csv(data_path)
                documents = transactions_to_documents(df)
                self.index = build_vector_store(documents, persist_dir)
        else:
            logger.warning("RAG index skipped because OpenAI is not configured")

        self.df = pd.read_csv(data_path)
        if "predicted_category" in self.df.columns and "category" not in self.df.columns:
            self.df["category"] = self.df["predicted_category"]
        logger.info("FinanceRAGEngine ready")

    def warm(self, dummy_question: str = "total spending summary") -> None:
        """Best-effort embedding warm-up."""
        try:
            _ = self.ask_with_summary(dummy_question)
        except Exception as exc:
            logger.warning("RAG warm-up skipped: %s", exc)

    def _build_prompt(self, question: str) -> str:
        compressed = compress_old_memory_turns(list(self.full_turn_history))
        memory_lines: list[str] = []
        for item in self.memory:
            memory_lines.append(f"Q: {item['question']}")
            memory_lines.append(f"A: {item['answer']}")

        blocks = []
        if compressed:
            blocks.append(compressed)
        if memory_lines:
            blocks.append("Conversation context:\n" + "\n".join(memory_lines))

        prefix = "\n\n".join(blocks)
        if prefix:
            return f"{prefix}\n\nCurrent question: {question}"
        return question

    def _format_citations(self, source_nodes: list[Any]) -> list[dict[str, Any]]:
        citations = []
        for node in source_nodes or []:
            metadata = getattr(node, "metadata", {}) or {}
            citations.append(
                {
                    "transaction_id": metadata.get("transaction_id"),
                    "merchant": metadata.get("merchant"),
                    "date": metadata.get("date"),
                    "category": metadata.get("category"),
                    "amount": metadata.get("amount"),
                }
            )
        return citations

    def spending_summary(self, filters: dict | None = None) -> dict[str, float | int]:
        filtered_df = apply_filters_to_df(self.df, filters)
        return compute_spending_summary(filtered_df)

    def ask_with_summary(self, question: str, filters: dict | None = None) -> dict[str, Any]:
        clarify = intent_clarification(question)
        if clarify:
            return {
                "answer": clarify,
                "summary": self.spending_summary(filters),
                "citations": [],
                "memory": list(self.memory),
                "confidence": 0.0,
                "insights": [],
                "clarification": True,
            }

        expanded = expand_query(question)
        hops = multi_hop_plan(question)
        logger.info("Q: %s", question)

        combined_answer_parts: list[str] = []
        all_nodes: list[Any] = []
        metadata_filters = build_metadata_filters(filters)

        if self.index is None:
            summary = self.spending_summary(filters)
            return {
                "answer": "Retrieval index is not available. Returning structured summary only.",
                "summary": summary,
                "citations": [],
                "memory": list(self.memory),
                "confidence": 0.4,
                "insights": proactive_insights(summary, question),
                "clarification": False,
            }

        retriever = self.index.as_retriever(similarity_top_k=12, filters=metadata_filters)

        for hop in hops:
            prompt = self._build_prompt(hop + " " + expanded)
            nodes = retriever.retrieve(prompt)
            nodes = rerank_nodes(list(nodes))

            if not nodes:
                combined_answer_parts.append(f"(No retrieval results for: {hop})")
                continue

            top_engine = self.index.as_query_engine(
                similarity_top_k=min(8, len(nodes)),
                filters=metadata_filters,
            )
            response = top_engine.query(hop)

            combined_answer_parts.append(str(response.response))
            src = getattr(response, "source_nodes", []) or nodes[:8]
            all_nodes.extend(src)

        answer_text = "\n\n".join(combined_answer_parts).strip()

        unique_nodes = {getattr(n, "node_id", id(n)): n for n in all_nodes}.values()
        ranked = rerank_nodes(list(unique_nodes))
        citations = self._format_citations(ranked[:8])

        confidence = answer_confidence_from_nodes(ranked[:8])
        summary = self.spending_summary(filters)

        if not ranked:
            answer_text = (
                "No matching transactions were found for that query. "
                "Try widening the date range or removing category filters."
            )
            confidence = 0.15

        insights = proactive_insights(summary, question)

        prompt_for_memory = question
        self.memory.append({"question": prompt_for_memory, "answer": answer_text})
        self.full_turn_history.append({"question": prompt_for_memory, "answer": answer_text})

        return {
            "answer": answer_text,
            "summary": summary,
            "citations": citations,
            "memory": list(self.memory),
            "confidence": confidence,
            "insights": insights,
            "clarification": False,
        }

    def ask(self, question: str, filters: dict | None = None) -> str:
        result = self.ask_with_summary(question, filters=filters)
        answer = result["answer"]

        if result.get("insights"):
            answer += "\n\nInsights:\n- " + "\n- ".join(result["insights"])

        if result["citations"]:
            citation_lines = []
            for cite in result["citations"][:5]:
                citation_lines.append(
                    f"- {cite.get('merchant')} on {cite.get('date')} "
                    f"(KES {cite.get('amount')}) [{cite.get('category')}]"
                )
            answer = f"{answer}\n\nSources:\n" + "\n".join(citation_lines)

        logger.info("A: %s", answer[:500])
        return answer


def answer_question(index: VectorStoreIndex | None, question: str, filters: dict | None = None):
    if index is None:
        return None

    metadata_filters = build_metadata_filters(filters)
    query_engine = index.as_query_engine(similarity_top_k=5, filters=metadata_filters)
    return query_engine.query(question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
    engine = FinanceRAGEngine(data_path=str(DEFAULT_CLASSIFIED_PATH))
    print(engine.ask("How much did I spend on food in total?"))
