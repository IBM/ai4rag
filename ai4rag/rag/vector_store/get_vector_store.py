# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from ..embedding.base_model import BaseEmbeddingModel
from .base_vector_store import BaseVectorStore
from .config import BaseVectorStoreConfig, MilvusConfig, MilvusLiteConfig, Neo4jConfig, PGVectorConfig


def get_vector_store(
    embedding_model: BaseEmbeddingModel,
    config: BaseVectorStoreConfig,
    collection_name: str | None = None,
    foundation_model: Any = None,
) -> BaseVectorStore:
    """Get vector store of desired type with chosen settings.

    The backend is selected by ``config.provider``, so the vector store type
    is fully determined by which config class is passed in — no separate
    type string is required.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Embedding model used for embeddings creation.

    config : MilvusConfig | MilvusLiteConfig | Neo4jConfig | PGVectorConfig
        Connection config for the chosen backend. :class:`MilvusConfig` targets a
        remote Milvus server; :class:`MilvusLiteConfig` selects the embedded,
        local Milvus Lite engine.

    collection_name : str | None, default=None
        Name of an existing collection to reuse. When omitted, a new name
        is generated following the ai4rag naming convention (see
        :func:`ai4rag.rag.vector_store.utils.generate_collection_name`).
    foundation_model : Any, default=None
        Optional foundation model passed to :class:`Neo4jGraphStore` for
        entity extraction during :meth:`add_documents`.  Ignored for all
        other backends.

    Returns
    -------
    BaseVectorStore
        Instance of the vector store.

    Raises
    ------
    TypeError
        If ``config`` is not the config class matching its ``provider`` (e.g. a
        ``provider="milvus"`` config that is not a :class:`MilvusConfig`).
    ValueError
        If ``config.provider`` names an unsupported backend.
    """

    match config.provider:
        case "milvus":
            if not isinstance(config, MilvusConfig):
                raise TypeError("MilvusConfig is required when provider='milvus'.")

            from .milvus import MilvusVectorStore

            return MilvusVectorStore(
                embedding_model=embedding_model,
                config=config,
                collection_name=collection_name,
            )

        case "milvus_lite":
            if not isinstance(config, MilvusLiteConfig):
                raise TypeError("MilvusLiteConfig is required when provider='milvus_lite'.")

            from .milvus import MilvusVectorStore

            return MilvusVectorStore(
                embedding_model=embedding_model,
                config=config,
                collection_name=collection_name,
            )

        case "pgvector":
            if not isinstance(config, PGVectorConfig):
                raise TypeError("PGVectorConfig is required when provider='pgvector'.")

            from .pgvector import PGVectorStore

            return PGVectorStore(
                embedding_model=embedding_model,
                config=config,
                collection_name=collection_name,
            )

        case "neo4j":
            if not isinstance(config, Neo4jConfig):
                raise TypeError("Neo4jConfig is required when provider='neo4j'.")

            from .neo4j import Neo4jGraphStore

            return Neo4jGraphStore(
                embedding_model=embedding_model,
                config=config,
                collection_name=collection_name,
                foundation_model=foundation_model,
            )

        case _:
            raise ValueError(f"Vector store provider '{config.provider}' is not supported.")
