# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from openai import OpenAI

from ai4rag.rag.embedding.base_model import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Class wrapping independently deployed embedding model with OpenAI client."""

    def __init__(self, client: OpenAI, model_id: str, params: dict[str, Any] | None = None):
        super().__init__(client=client, model_id=model_id, params=params)

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    @params.setter
    def params(self, params: dict[str, Any] | None) -> None:
        if params is None:
            self._params = {}
        else:
            self._params = params
        if "embedding_dimension" not in self._params:
            self._params["embedding_dimension"] = self._detect_embedding_dimension()

    def _detect_embedding_dimension(self) -> int:
        """Detect embedding dimension by making a minimal embedding call."""
        embedding = self.client.embeddings.create(model=self.model_id, input="test").data[0].embedding
        return len(embedding)

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
        resp = self.client.embeddings.create(model=self.model_id, input=texts)
        return [d.embedding for d in resp.data]

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
        return self.client.embeddings.create(model=self.model_id, input=query).data[0].embedding
