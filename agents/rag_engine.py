from collections import deque
import os
import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb

try:
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
except ImportError:
    MetadataFilter = None
    MetadataFilters = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

load_dotenv()


def _get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def setup_models() -> bool:
    api_key = _get_openai_api_key()
    if not api_key:
        logger.warning("OpenAI API key not set. Set OPENAI_API_KEY to enable RAG.")
        return False

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key
    )
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    logger.info("OpenAI models configured")
    return True


def transactions_to_documents(df: pd.DataFrame) -> list:
    documents = []

    for i, row in df.iterrows():
        date_str = pd.to_datetime(row['date']).strftime('%d %B %Y')
        text = (
            f"On {date_str}, there was a {row['category']} transaction "
            f"of KES {row['amount']:,.2f} at {row['merchant']}. "
            f"This was on a {row.get('day_of_week', 'weekday')}. "
            f"Category: {row['category']}. "
            f"Amount: KES {row['amount']:,.2f}."
        )
        metadata = {
            'date': str(row['date']),
            'merchant': str(row['merchant']),
            'amount': float(row['amount']),
            'category': str(row['category']),
            'transaction_id': i
        }
        documents.append(Document(text=text, metadata=metadata))

    logger.info("Created %s documents", len(documents))
    return documents


def build_vector_store(documents: list, persist_dir: str = "./data/chroma_db") -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("transactions")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    logger.info("Vector store built and saved")
    return index


def load_vector_store(persist_dir: str = "./data/chroma_db") -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("transactions")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    logger.info("Vector store loaded from disk")
    return index


def build_metadata_filters(filters: dict | None) -> Any:
    """
    Builds metadata filters for LlamaIndex query engines.
    """

    if not filters or MetadataFilters is None:
        return None

    metadata_filters = []

    category = filters.get('category') if isinstance(filters, dict) else None
    if category:
        metadata_filters.append(MetadataFilter(key="category", operator="==", value=category))

    start_date = filters.get('start_date') if isinstance(filters, dict) else None
    end_date = filters.get('end_date') if isinstance(filters, dict) else None

    if start_date:
        metadata_filters.append(MetadataFilter(key="date", operator=">=", value=str(start_date)))
    if end_date:
        metadata_filters.append(MetadataFilter(key="date", operator="<=", value=str(end_date)))

    if not metadata_filters:
        return None

    return MetadataFilters(filters=metadata_filters)


def apply_filters_to_df(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """
    Filters the dataframe by category and date range.
    """

    if not filters:
        return df

    filtered = df.copy()
    if 'category' in filters:
        filtered = filtered[filtered['category'] == filters['category']]

    if 'start_date' in filters:
        filtered = filtered[pd.to_datetime(filtered['date']) >= pd.to_datetime(filters['start_date'])]

    if 'end_date' in filters:
        filtered = filtered[pd.to_datetime(filtered['date']) <= pd.to_datetime(filters['end_date'])]

    return filtered


def compute_spending_summary(df: pd.DataFrame) -> dict:
    """
    Computes total, average, max, min, and count for a transaction set.
    """

    if df.empty:
        return {
            'total': 0.0,
            'average': 0.0,
            'max': 0.0,
            'min': 0.0,
            'count': 0
        }

    return {
        'total': float(df['amount'].sum()),
        'average': float(df['amount'].mean()),
        'max': float(df['amount'].max()),
        'min': float(df['amount'].min()),
        'count': int(df.shape[0])
    }


def answer_question(index: VectorStoreIndex | None, question: str, filters: dict | None = None):
    if index is None:
        return None

    metadata_filters = build_metadata_filters(filters)
    query_engine = index.as_query_engine(similarity_top_k=5, filters=metadata_filters)
    response = query_engine.query(question)
    return response


class FinanceRAGEngine:
    def __init__(self, data_path: str, persist_dir: str = "./data/chroma_db"):
        self.openai_ready = setup_models()
        self.persist_dir = persist_dir
        self.data_path = data_path
        self.memory = deque(maxlen=5)

        self.index = None
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
        logger.info("FinanceRAGEngine ready")

    def _build_prompt(self, question: str) -> str:
        if not self.memory:
            return question

        context_lines = []
        for item in self.memory:
            context_lines.append(f"Q: {item['question']}")
            context_lines.append(f"A: {item['answer']}")

        context = "\n".join(context_lines)
        return f"Conversation context:\n{context}\n\nCurrent question: {question}"

    def _format_citations(self, source_nodes: list) -> list:
        citations = []
        for node in source_nodes or []:
            metadata = getattr(node, 'metadata', {}) or {}
            citations.append({
                'transaction_id': metadata.get('transaction_id'),
                'merchant': metadata.get('merchant'),
                'date': metadata.get('date'),
                'category': metadata.get('category'),
                'amount': metadata.get('amount'),
            })
        return citations

    def spending_summary(self, filters: dict | None = None) -> dict:
        filtered_df = apply_filters_to_df(self.df, filters)
        return compute_spending_summary(filtered_df)

    def ask_with_summary(self, question: str, filters: dict | None = None) -> dict:
        logger.info("Q: %s", question)
        prompt = self._build_prompt(question)

        response = answer_question(self.index, prompt, filters=filters)
        if response is None:
            summary = self.spending_summary(filters)
            answer_text = "OpenAI API key not configured. Returning summary-only response."
            citations = []
        else:
            answer_text = str(response.response)
            citations = self._format_citations(getattr(response, 'source_nodes', []))
        summary = self.spending_summary(filters)

        self.memory.append({'question': question, 'answer': answer_text})

        return {
            'answer': answer_text,
            'summary': summary,
            'citations': citations,
            'memory': list(self.memory),
        }

    def ask(self, question: str, filters: dict | None = None) -> str:
        result = self.ask_with_summary(question, filters=filters)
        answer = result['answer']

        if result['citations']:
            citation_lines = []
            for cite in result['citations'][:5]:
                citation_lines.append(
                    f"- {cite.get('merchant')} on {cite.get('date')} (KES {cite.get('amount')})"
                )
            answer = f"{answer}\n\nSources:\n" + "\n".join(citation_lines)

        logger.info("A: %s", answer)
        return answer


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("FINSIGHT RAG ENGINE - TESTING")
    print("=" * 50)

    engine = FinanceRAGEngine(data_path="data/classified_transactions.csv")

    test_questions = [
        "How much did I spend on food in total?",
        "What is my most expensive transaction?",
        "How much did I spend on transport?",
        "Which merchant did I visit the most?",
        "What was my biggest single purchase?",
        "How much did I spend on entertainment?",
        "What are my top 3 spending categories?",
        "Did I spend more on food or transport?",
        "What was the most expensive food transaction?",
        "How much did I spend at Naivas Supermarket?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {engine.ask(question)}")
        print("-" * 50)
