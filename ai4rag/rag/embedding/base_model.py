# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any


class BaseEmbeddingModel(ABC):
    def __init__(self, model_id: str, params: dict[str, Any]):
        self.model_id = model_id
        self.params = params

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
