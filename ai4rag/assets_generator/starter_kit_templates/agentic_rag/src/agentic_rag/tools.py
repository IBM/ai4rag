import sys
from os import getenv
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlite_shim import patch_sqlite3

patch_sqlite3()

from ai4rag.rag.embedding.openai_model import (  # noqa: E402
    OpenAIEmbeddingModel,
    OpenAIEmbeddingParams,
)
from ai4rag.rag.retrieval.retriever import Retriever  # noqa: E402
from ai4rag.rag.vector_store import (  # noqa: E402
    get_vector_store,
    get_vector_store_config,
)
from langchain_core.tools import tool  # noqa: E402
from openai import OpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agentic_rag.config import AgentConfig  # noqa: E402

try:
    import mlflow
    from mlflow.entities import Document as MlflowDocument
except ImportError:
    mlflow = None


def _initialize_retriever(
    maas_api_key: Optional[str] = None,
    maas_base_url: Optional[str] = None,
    embedding_model_id: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    collection_name: Optional[str] = None,
) -> Retriever:
    """Initialize the ai4rag retriever with MaaS embeddings and vector store."""

    if not maas_api_key:
        maas_api_key = getenv("MAAS_API_KEY")
    if not maas_base_url:
        maas_base_url = getenv("MAAS_BASE_URL")
    if not embedding_model_id:
        embedding_model_id = getenv("EMBEDDING_MODEL_ID", "")
    if not embedding_dimension:
        embedding_dimension = int(getenv("EMBEDDING_DIMENSION", "768"))
    if not collection_name:
        collection_name = getenv("MILVUS_COLLECTION_NAME") or getenv("PGVECTOR_COLLECTION_NAME")

    if not maas_api_key or not maas_base_url:
        raise ValueError("MAAS_API_KEY and MAAS_BASE_URL must be set")

    if not collection_name:
        raise RuntimeError(
            "Collection name env var is not set (MILVUS_COLLECTION_NAME or PGVECTOR_COLLECTION_NAME)."
        )

    if not maas_base_url.startswith("https://"):
        raise ValueError(
            f"MaaS base URL must use HTTPS to protect API key transmission. Got: {maas_base_url}"
        )

    client = OpenAI(base_url=maas_base_url, api_key=maas_api_key)

    params = OpenAIEmbeddingParams(
        embedding_dimension=embedding_dimension, context_length=1015
    )
    embedding_model = OpenAIEmbeddingModel(
        client=client, model_id=embedding_model_id, params=params
    )

    provider_type = getenv("PROVIDER_TYPE", "milvus")
    vector_store_config = get_vector_store_config(provider_type)
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        config=vector_store_config,
        collection_name=collection_name,
    )

    method = getenv("RETRIEVAL_METHOD", "simple")
    number_of_chunks = int(getenv("NUMBER_OF_CHUNKS", "5"))
    search_mode = getenv("SEARCH_MODE") or None
    ranker_strategy = getenv("RANKER_STRATEGY") or None
    ranker_alpha_raw = getenv("RANKER_ALPHA")
    ranker_alpha = float(ranker_alpha_raw) if ranker_alpha_raw else None

    retriever = Retriever(
        vector_store=vector_store,
        method=method,
        number_of_chunks=number_of_chunks,
        search_mode=search_mode,
        ranker_strategy=ranker_strategy,
        ranker_alpha=ranker_alpha,
    )

    return retriever


def create_retriever_tool():
    """Factory function that creates a retriever tool with cached retriever instance."""
    _retriever_cache = None

    class RetrieverInput(BaseModel):
        """Schema for the retriever tool input."""

        query: str = Field(
            description="The search query describing what information you need to retrieve."
        )

    @tool("retriever", args_schema=RetrieverInput)
    def retriever_tool(query: str) -> str:
        """Search the knowledge base for information relevant to the query.

        Use this tool when you need to find specific information from the knowledge base
        to answer the user's question accurately.

        Args:
            query: The search query describing what information you need to retrieve.

        Returns:
            Retrieved documents containing relevant information.
        """
        nonlocal _retriever_cache

        if isinstance(query, dict):
            query = query.get("value", query.get("query", str(query)))

        if _retriever_cache is None:
            _retriever_cache = _initialize_retriever()

        retrieved_docs = _retriever_cache.retrieve(query)

        if not retrieved_docs or len(retrieved_docs) == 0:
            return "No relevant information was found in the provided documents for this query."

        formatted_docs = []
        retriever_docs = []
        for i, doc in enumerate(retrieved_docs, 1):
            text_content = getattr(doc, "text", getattr(doc, "page_content", None))
            content = (text_content or "").strip()
            if not content or all(c in "=-_*#|" for c in content):
                continue

            metadata = getattr(doc, "metadata", None) or {}
            source = metadata.get("source", "unknown")

            score = getattr(doc, "score", getattr(doc, "similarity", None))
            if score is not None and isinstance(score, (int, float)):
                score_str = f"{score:.3f}"
            else:
                score_str = "N/A"

            doc_text = f"--- Document {len(formatted_docs) + 1} ---\n"
            doc_text += f"Content: {content}\n"
            doc_text += f"Source: {source}\n"
            doc_text += f"Score: {score_str}"

            formatted_docs.append(doc_text)

            if mlflow:
                retriever_docs.append(
                    MlflowDocument(
                        page_content=content,
                        metadata={
                            "source": source,
                            "score": getattr(doc, "score", None),
                        },
                    )
                )

        if mlflow and retriever_docs:
            with mlflow.start_span(name="retrieve", span_type="RETRIEVER") as span:
                span.set_inputs({"query": query})
                span.set_outputs(retriever_docs)

        if not formatted_docs:
            return "No relevant information was found in the provided documents for this query."

        config = AgentConfig.from_env()
        context = "\n\n".join(
            config.context_template.format(document=document, doc_number=index)
            for index, document in enumerate(formatted_docs, 1)
        )
        try:
            return config.user_message_template.format(
                reference_documents=context,
                question=query,
            )
        except KeyError as exc:
            raise ValueError(f"Prompt template contains unsupported placeholder: {exc}") from exc

    return retriever_tool


retriever_tool = create_retriever_tool()
