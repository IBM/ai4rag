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

    embedding_dimension: Optional[int] = None
    context_length: Optional[int] = None
    timeout: Optional[float | Timeout] = None
    model_type: Optional[str] = None
    provider_id: Optional[str] = None
    provider_resource_id: Optional[str] = None


class LSEmbeddingModel(BaseEmbeddingModel[LlamaStackClient, LSEmbeddingParams]):
    """Creates embeddings for LLamaStack client."""

    def __init__(self, client: LlamaStackClient, model_id: str, params: dict | LSEmbeddingParams | None = None):
        super().__init__(client=client, model_id=model_id, params=params)

    @property
    def params(self) -> LSEmbeddingParams:
        return self._params

    @params.setter
    def params(self, params: dict | LSEmbeddingParams | None) -> None:
        if params is None:
            self._params = LSEmbeddingParams()
        elif isinstance(params, LSEmbeddingParams):
            self._params = params
        elif isinstance(params, dict):
            self._params = LSEmbeddingParams(**params)
        else:
            raise TypeError(f"Incorrect type of 'params' parameter: {type(params)}.")
        if self._params.embedding_dimension is None:
            self._params.embedding_dimension = self._detect_embedding_dimension()
        if self._params.context_length is None:
            self._params.context_length = self._detect_context_length()

    def _detect_embedding_dimension(self) -> int:
        """Detect embedding dimension by making a minimal embedding call.

        Note: This method is called during initialization when ``embedding_dimension``
        is not explicitly provided.  It issues a real API request to the Llama Stack
        server, so the server must be reachable at construction time.

        Raises
        ------
        RuntimeError
            When the embedding dimension cannot be determined (e.g. the server
            is unreachable or the model is not available).
        """
        try:
            embedding = self._embed_text(text_input="test")[0]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to auto-detect embedding dimension for model '{self.model_id}'. "
                "Provide 'embedding_dimension' explicitly or ensure the embedding service is reachable."
            ) from exc
        return len(embedding)

    def _detect_context_length(self) -> int:
        """Detect maximum context length by probing with increasing input sizes.

        Sends probe texts of descending token counts and returns the largest
        that the model accepts.  Each probe consists of repeated words so that
        one word ≈ one token.

        Raises
        ------
        RuntimeError
            When the context length cannot be determined (e.g. all probes fail).
        """
        probe_sizes = [4096, 2048, 1024, 512, 256]
        for size in probe_sizes:
            probe_text = "word " * size
            try:
                self._embed_text(text_input=probe_text)
                return size
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        raise RuntimeError(
            f"Failed to auto-detect 'context_length' for model '{self.model_id}'. "
            "Provide 'context_length' explicitly or ensure the embedding service is reachable."
        )

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
