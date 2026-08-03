# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Sequence

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.config import BaseVectorStoreConfig
from ai4rag.rag.vector_store.utils import resolve_collection_name

__all__ = ["BaseVectorStore"]


class BaseVectorStore(ABC):
    """
    Abstract class defining interface for VectorStore in the ai4rag experiment.
    Single instance defines 1 collection/index that can be used to store or retrieve data.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: BaseVectorStoreConfig,
        distance_metric: str,
        collection_name: str | None = None,
    ):
        """Initialize the state shared by every concrete vector store.

        Parameters
        ----------
        embedding_model : BaseEmbeddingModel
            Model used to embed documents and queries.
        config : BaseVectorStoreConfig
            Backend-specific connection parameters.
        distance_metric : str
            Metric used to measure similarity between vectors.
        collection_name : str | None, default=None
            Existing collection to reuse; must start with the ``ai4rag`` prefix.
            When ``None``, a new compliant name is generated (see
            :func:`ai4rag.rag.vector_store.utils.resolve_collection_name`).
        """
        self.embedding_model = embedding_model
        self._config = config
        self.distance_metric = distance_metric
        self._collection_name = resolve_collection_name(collection_name)

    @abstractmethod
    def search(self, query: str, k: int, **kwargs) -> list[AI4RAGChunk]:
        """
        Search for the chunks relevant to the query.
        The method used will be simple similarity search.

        Parameters
        ----------
        query : str
            Question / query for which the similarity search will be executed.

        k : int
            Number of chunks to be returned as a result of similarity search

        **kwargs : Any
            Backend-specific search options (e.g. metadata filters or hybrid
            search parameters). Ignored by backends that do not support them.

        Returns
        -------
        list[AI4RAGChunk]
            List of chunks with content and metadata.
        """

    @abstractmethod
    def add_documents(self, documents: Sequence[AI4RAGChunk]) -> None:
        """
        Add documents to the collection.

        Parameters
        ----------
        documents : Sequence[AI4RAGChunk]
            Chunks to add to the collection.
        """

    @property
    def collection_name(self) -> str:
        """The resolved collection name — reused when supplied, otherwise generated.

        Guaranteed to start with
        :data:`~ai4rag.rag.vector_store.utils.COLLECTION_NAME_PREFIX` and to be a
        valid, length-bounded identifier usable as both a backend collection name
        and a physical SQL table name.
        """
        return self._collection_name
