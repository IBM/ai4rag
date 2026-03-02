# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

from httpx import Timeout
from llama_stack_client import LlamaStackClient

from .base_model import BaseEmbeddingModel

__all__ = ["LSEmbeddingModel", "LSEmbeddingParams"]


@dataclass
class LSEmbeddingParams:
    """LLamaStack parameters to be used to create embeddings."""

    embedding_dimension: int
    context_length: Optional[int] = None
    timeout: Optional[float | Timeout] = None
    model_type: Optional[str] = None
    provider_id: Optional[str] = None
    provider_resource_id: Optional[str] = None


class LSEmbeddingModel(BaseEmbeddingModel[LlamaStackClient, LSEmbeddingParams]):
    """Creates embeddings for LLamaStack client."""

    def __init__(self, client: LlamaStackClient, model_id: str, params: dict | LSEmbeddingParams):
        super().__init__(client=client, model_id=model_id, params=params)

    @property
    def params(self) -> LSEmbeddingParams:
        return self._params

    @params.setter
    def params(self, params: dict | LSEmbeddingParams) -> None:
        if isinstance(params, LSEmbeddingParams):
            self._params = params
        elif isinstance(params, dict):
            self._params = LSEmbeddingParams(**params)
        else:
            raise TypeError(f"Incorrect type of 'params' parameter: {type(params)}.")

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
