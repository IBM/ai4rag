# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import TypeVar, Generic


ClientT = TypeVar("ClientT")
EmbeddingParamsT = TypeVar("EmbeddingParamsT")


class EmbeddingModel(ABC, Generic[ClientT, EmbeddingParamsT]):
    def __init__(self, client: ClientT, model_id: str, params: EmbeddingParamsT | None = None):
        self.client: ClientT = client
        self.model_id = model_id
        self.params: EmbeddingParamsT = params

    def __str__(self) -> str:
        return self.model_id

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
