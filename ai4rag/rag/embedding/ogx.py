# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

from httpx import Timeout
from ogx_client import OgxClient

from .base_model import BaseEmbeddingModel

# pylint: disable=duplicate-code

__all__ = ["OGXEmbeddingModel", "OGXEmbeddingParams"]


@dataclass
class OGXEmbeddingParams:
    """OGX parameters to be used to create embeddings."""

    embedding_dimension: Optional[int] = None
    context_length: Optional[int] = None
    timeout: Optional[float | Timeout] = None
    model_type: Optional[str] = None
    provider_id: Optional[str] = None
    provider_resource_id: Optional[str] = None


class OGXEmbeddingModel(BaseEmbeddingModel[OgxClient, OGXEmbeddingParams]):
    """Creates embeddings for OGX client."""

    def __init__(self, client: OgxClient, model_id: str, params: dict | OGXEmbeddingParams | None = None):
        super().__init__(client=client, model_id=model_id, params=params)

    @property
    def params(self) -> OGXEmbeddingParams:
        """Get model params."""
        return self._params

    @params.setter
    def params(self, params: dict | OGXEmbeddingParams | None) -> None:
        """Set model params."""
        if params is None:
            self._params = OGXEmbeddingParams()
        elif isinstance(params, OGXEmbeddingParams):
            self._params = params
        elif isinstance(params, dict):
            self._params = OGXEmbeddingParams(**params)
        else:
            raise TypeError(f"Incorrect type of 'params' parameter: {type(params)}.")
        if self._params.embedding_dimension is None:
            self._params.embedding_dimension = self._detect_embedding_dimension()
        if self._params.context_length is None:
            self._params.context_length = self._detect_context_length()

    def _detect_embedding_dimension(self) -> int:
        """Detect embedding dimension by making a minimal embedding call.

        Note: This method is called during initialization when ``embedding_dimension``
        is not explicitly provided.  It issues a real API request to the OGX
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
        """Detect maximum context length via binary search over probe sizes.

        Performs a binary search between 64 and 8192 tokens to find the
        largest input the model accepts.  Each probe consists of repeated
        words so that one word ≈ one token.  The search stops when the
        remaining interval is smaller than 256 tokens, keeping the number
        of API calls to roughly 5.

        Raises
        ------
        RuntimeError
            When the context length cannot be determined (e.g. all probes fail).
        """
        lo, hi, best = 64, 8192, None
        while hi - lo >= 64:
            mid = (lo + hi) // 2
            probe_text = "word " * mid
            try:
                self._embed_text(text_input=probe_text)
                best = mid
                lo = mid + 1
            except Exception:  # pylint: disable=broad-exception-caught
                hi = mid - 1
        if best is not None:
            return best
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
        The maximum batch size supported is 2048 chunks,
        hence we need to do it iteratively.

        Parameters
        ----------
        texts : list[str]
            List of text-like chunks.

        Returns
        -------
        list[list[float]]
            Embeddings made from the list of texts.
        """
        resp = []
        for idx in range(0, len(texts), 2048):
            resp.extend(self._embed_text(text_input=texts[idx : idx + 2048]))

        return resp

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
