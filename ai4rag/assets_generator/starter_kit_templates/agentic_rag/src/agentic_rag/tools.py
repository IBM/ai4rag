# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from openai import OpenAI

from ai4rag.rag.embedding.openai_model import (
    OpenAIEmbeddingModel,
    OpenAIEmbeddingParams,
)
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.vector_store import (
    get_vector_store,
    get_vector_store_config,
)

from .config import AgentConfig


def initialize_retriever(
    client: OpenAI,
    config: AgentConfig,
) -> Retriever:
    """Initialize the ai4rag retriever with MaaS embeddings and vector store."""
    if not config.collection_name:
        raise ValueError("Collection name is required in the agent configuration.")
    if not config.embedding_model_id:
        raise ValueError("Embedding model ID is required in the agent configuration.")
    if config.embedding_dimension < 1:
        raise ValueError("Embedding dimension must be positive.")

    embedding_model = OpenAIEmbeddingModel(
        client=client,
        model_id=config.embedding_model_id,
        params=OpenAIEmbeddingParams(embedding_dimension=config.embedding_dimension, context_length=1015),
    )

    vector_store = get_vector_store(
        embedding_model=embedding_model,
        config=get_vector_store_config(config.provider_type),
        collection_name=config.collection_name,
    )

    return Retriever(
        vector_store=vector_store,
        method=config.retrieval_method,
        number_of_chunks=config.number_of_chunks,
        search_mode=config.search_mode,
        ranker_strategy=config.ranker_strategy or None,
        ranker_alpha=config.ranker_alpha,
    )
