# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Optional, TypedDict

from httpx import Timeout
from llama_stack_client import Client

from .base_model import BaseEmbeddingModel

__all__ = ["LSEmbeddingModel", "LSEmbeddingParams"]


class LSEmbeddingParams(TypedDict, total=False):
    """LLamaStack parameters to be used to create embeddings."""

    embedding_dimension: int
    context_length: int
    timeout: Optional[float | Timeout]
    model_type: Optional[str]
    provider_id: Optional[str]
    provider_resource_id: Optional[str]


class LSEmbeddingModel(BaseEmbeddingModel[Client, LSEmbeddingParams]):
    """Creates embeddings for LLamaStack client."""

    def __init__(self, client: Client, model_id: str, params: LSEmbeddingParams):
        params = params or {}
        super().__init__(client=client, model_id=model_id, params=params)

    def _embed_text(self, text_input: list[str] | str) -> list[list[float]]:
        """Embeds documents.

        Parameters
        ----------
        text_input : list[str] | str
            List of text-like chunks or single text-like chunk.

        Returns
        -------
        list[list[float]]
            Embeddings made from the list of texts or a single text.
        """

        return [
            data.embedding
            for data in self.client.embeddings.create(input=text_input, model=self.model_id).data
            if not isinstance(data.embedding, str)
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds given list of strings.

        Parameters
        ----------
        texts : list[str]
            List of text-like chunks.

        Returns
        -------
        list[list[float]]
            Embeddings made from the list of texts.
        """
        return self._embed_text(text_input=texts)

    def embed_query(self, query: str) -> list[float]:
        """Embeds given query.

        Parameters
        ----------
        query : str
            Single text-like chunk.

        Returns
        -------
        list[]
            Embeddings made from a single text.
        """
        return self._embed_text(text_input=query)[0]
