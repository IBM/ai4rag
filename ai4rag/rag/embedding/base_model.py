# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ClientT = TypeVar("ClientT")
EmbeddingParamsT = TypeVar("EmbeddingParamsT")


class BaseEmbeddingModel(ABC, Generic[ClientT, EmbeddingParamsT]):
    """Interface definition for Embedding Model that will be used for `ai4rag`."""

    def __init__(self, client: ClientT, model_id: str, params: EmbeddingParamsT | None = None):
        self.client: ClientT = client
        self.model_id = model_id
        self.params: EmbeddingParamsT = params

    def __str__(self) -> str:
        return self.model_id

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEmbeddingModel):
            return NotImplemented

        return self.model_id == other.model_id

    def __hash__(self):
        return hash(self.model_id)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, BaseEmbeddingModel):
            return NotImplemented
        return self.model_id < other.model_id

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents.

        Parameters
        ----------
        texts : list[str]
            List of text-like chunks.

        Returns
        -------
        list[list[float]]
            Embeddings made from the list of texts.
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed query text

        Parameters
        ----------
        query : str
            User's query as text.

        Returns
        -------
        list[float]
            Single embeddings vector made from the query.
        """
