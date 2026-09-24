# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from os import getenv

from .sqlite_shim import patch_sqlite3

patch_sqlite3()

from openai import OpenAI  # noqa: E402

from ai4rag.rag.embedding.openai_model import (  # noqa: E402
    OpenAIEmbeddingModel,
    OpenAIEmbeddingParams,
)
from ai4rag.rag.retrieval.retriever import Retriever  # noqa: E402
from ai4rag.rag.vector_store import (  # noqa: E402
    get_vector_store,
    get_vector_store_config,
)


def _initialize_retriever(
    maas_api_key: str | None = None,
    maas_base_url: str | None = None,
    embedding_model_id: str | None = None,
    embedding_dimension: int | None = None,
    collection_name: str | None = None,
    provider_type: str | None = None,
    retrieval_method: str | None = None,
    number_of_chunks: int | None = None,
    search_mode: str | None = None,
    ranker_strategy: str | None = None,
    ranker_alpha: float | None = None,
) -> Retriever:
    """Initialize the ai4rag retriever with MaaS embeddings and vector store."""

    maas_api_key = maas_api_key or getenv("MAAS_API_KEY")
    maas_base_url = maas_base_url or getenv("MAAS_BASE_URL")
    embedding_model_id = embedding_model_id or getenv("EMBEDDING_MODEL_ID", "")
    embedding_dimension = embedding_dimension or int(getenv("EMBEDDING_DIMENSION", "768"))
    collection_name = collection_name or getenv("MILVUS_COLLECTION_NAME") or getenv("PGVECTOR_COLLECTION_NAME")

    if not maas_api_key or not maas_base_url:
        raise ValueError("MAAS_API_KEY and MAAS_BASE_URL must be set")
    if not collection_name:
        raise RuntimeError("Collection name env var is not set (MILVUS_COLLECTION_NAME or PGVECTOR_COLLECTION_NAME).")
    if not maas_base_url.startswith("https://"):
        raise ValueError(f"MaaS base URL must use HTTPS to protect API key transmission. Got: {maas_base_url}")

    client = OpenAI(base_url=maas_base_url, api_key=maas_api_key)
    embedding_model = OpenAIEmbeddingModel(
        client=client,
        model_id=embedding_model_id,
        params=OpenAIEmbeddingParams(embedding_dimension=embedding_dimension, context_length=1015),
    )

    vector_store = get_vector_store(
        embedding_model=embedding_model,
        config=get_vector_store_config(provider_type or getenv("PROVIDER_TYPE", "milvus")),
        collection_name=collection_name,
    )

    ranker_alpha_raw = ranker_alpha if ranker_alpha is not None else getenv("RANKER_ALPHA")
    return Retriever(
        vector_store=vector_store,
        method=retrieval_method or getenv("RETRIEVAL_METHOD", "simple"),
        number_of_chunks=number_of_chunks or int(getenv("NUMBER_OF_CHUNKS", "5")),
        search_mode=search_mode or getenv("SEARCH_MODE") or "vector",
        ranker_strategy=ranker_strategy or getenv("RANKER_STRATEGY") or None,
        ranker_alpha=float(ranker_alpha_raw) if ranker_alpha_raw else None,
    )
