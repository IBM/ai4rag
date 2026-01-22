# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import abstractmethod, ABC
from typing import Sequence

from langchain_core.documents import Document

from ai4rag.rag.embedding.base_model import EmbeddingModel


__all__ = ["BaseVectorStore"]


class BaseVectorStore(ABC):
    """
    Abstract class defining interface for VectorStore in the ai4rag experiment.
    Single instance defines 1 collection/index that can be used to store or retrieve data.
    """

    def __init__(self, embedding_model: EmbeddingModel, collection_name: str, distance_metric: str):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.distance_metric = distance_metric

    @abstractmethod
    def search(self, query: str, k: int) -> list[dict]:
        """
        Search for the chunks relevant to the query.
        The method used will be simple similarity search.

        Parameters
        ----------
        query : str
            Question / query for which the similarity search will be executed.

        k : int
            Number of chunks to be returned as a result of similarity search

        Returns
        -------
        list[dict]
            List of chunks as dicts with content and metadata.
        """

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        """
        Add documents to the collection.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents to add to the collection.
        """
