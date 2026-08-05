# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ..embedding.base_model import BaseEmbeddingModel
from .base_vector_store import BaseVectorStore
from .chroma import ChromaVectorStore
from .config import BaseVectorStoreConfig, ChromaConfig, MilvusConfig, PGVectorConfig
from .milvus import MilvusVectorStore
from .pgvector import PGVectorStore


def get_vector_store(
    embedding_model: BaseEmbeddingModel,
    config: BaseVectorStoreConfig,
    collection_name: str | None = None,
) -> BaseVectorStore:
    """Get vector store of desired type with chosen settings.

    The backend is selected by ``config.provider``, so the vector store type
    is fully determined by which config class is passed in — no separate
    type string is required.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Embedding model used for embeddings creation.

    config : ChromaConfig | MilvusConfig | PGVectorConfig
        Connection config for the chosen backend.

    collection_name : str | None, default=None
        Name of an existing collection to reuse. When omitted, a new name
        is generated following the ai4rag naming convention (see
        :func:`ai4rag.rag.vector_store.utils.generate_collection_name`).

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
        case "chroma":
            if not isinstance(config, ChromaConfig):
                raise TypeError("ChromaConfig is required when provider='chroma'.")

            return ChromaVectorStore(
                embedding_model=embedding_model,
                config=config,
                collection_name=collection_name,
            )

        case "milvus":
            if not isinstance(config, MilvusConfig):
                raise TypeError("MilvusConfig is required when provider='milvus'.")

            return MilvusVectorStore(
                embedding_model=embedding_model,
                config=config,
                collection_name=collection_name,
            )

        case "pgvector":
            if not isinstance(config, PGVectorConfig):
                raise TypeError("PGVectorConfig is required when provider='pgvector'.")

            return PGVectorStore(
                embedding_model=embedding_model,
                config=config,
                collection_name=collection_name,
            )

        case _:
            raise ValueError(f"Vector store provider '{config.provider}' is not supported.")
