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
        if "context_length" not in self._params:
            self._params["context_length"] = self._detect_context_length()

    def _detect_embedding_dimension(self) -> int:
        """Detect embedding dimension by making a minimal embedding call.

        Note: This method is called during initialization when ``embedding_dimension``
        is not present in the params dict.  It issues a real API request to the
        OpenAI API compliant endpoint, so the service must be reachable at construction time.

        Raises
        ------
        RuntimeError
            When the embedding dimension cannot be determined (e.g. the service
            is unreachable or the model is not available).
        """
        try:
            embedding = self.client.embeddings.create(model=self.model_id, input="test").data[0].embedding
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
        while hi - lo >= 256:
            mid = (lo + hi) // 2
            probe_text = "word " * mid
            try:
                self.client.embeddings.create(model=self.model_id, input=probe_text)
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
