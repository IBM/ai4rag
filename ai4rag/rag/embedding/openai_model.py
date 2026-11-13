# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""OpenAI-compatible embedding calls, capability detection, and response validation."""

from dataclasses import dataclass
from functools import partial
from math import isfinite

import httpx
from openai import BadRequestError, OpenAI

from .base_model import BaseEmbeddingModel
from .request_executor import (
    INDEXING_POLICY,
    PROBE_POLICY,
    QUERY_POLICY,
    EmbeddingRequestExecutor,
    RequestPolicy,
    is_context_overflow,
)

__all__ = ["OpenAIEmbeddingModel", "OpenAIEmbeddingParams"]

MIN_CONTEXT_LENGTH = 700


@dataclass
class OpenAIEmbeddingParams:
    """Model capabilities used to validate vectors and repair oversized input.

    Parameters
    ----------
    embedding_dimension : int | None, default=None
        Expected vector width. None triggers detection when assigned to a model.
    context_length : int | None, default=None
        Input context length in tokens. None triggers approximate detection when
        assigned to a model. The model requires at least MIN_CONTEXT_LENGTH.
    """

    embedding_dimension: int | None = None
    context_length: int | None = None


class OpenAIEmbeddingModel(BaseEmbeddingModel[OpenAI, OpenAIEmbeddingParams]):
    """Create validated embeddings through an OpenAI-compatible client.

    Parameters
    ----------
    client : OpenAI
        Shared synchronous client. Request-specific options leave its settings
        unchanged; the model does not close the client or its HTTP transport.
    model_id : str
        Model identifier sent to the embedding endpoint.
    params : dict[str, int | None] | OpenAIEmbeddingParams | None, default=None
        Model capabilities. Missing values are detected during construction,
        which requires a reachable embedding service.

    Notes
    -----
    Document calls use batches of at most 128 texts. EmbeddingRequestExecutor
    owns recovery for each batch; this model performs individual API calls and
    validates their responses. Capability probes disable truncation and the
    extended read-timeout fallback.
    """

    _BATCH_SIZE = 128

    def __init__(
        self, client: OpenAI, model_id: str, params: dict[str, int | None] | OpenAIEmbeddingParams | None = None
    ) -> None:
        """Initialize the client binding and resolve missing model capabilities."""
        super().__init__(client=client, model_id=model_id, params=params)

    @property
    def params(self) -> OpenAIEmbeddingParams:
        """Get the resolved model capabilities.

        Returns
        -------
        OpenAIEmbeddingParams
            The model's parameter object; this is not a defensive copy.
        """
        return self._params

    @params.setter
    def params(self, params: dict[str, int | None] | OpenAIEmbeddingParams | None) -> None:
        """Assign model capabilities, detect missing values, and check context size.

        Parameters
        ----------
        params : dict[str, int | None] | OpenAIEmbeddingParams | None
            Capabilities to assign. None requests detection of both values.
            An OpenAIEmbeddingParams instance is stored and updated in place;
            a dictionary is used to construct a new parameter object.

        Raises
        ------
        TypeError
            If the parameter type or dictionary field names are unsupported.
        RuntimeError
            If a missing capability cannot be detected. The probe failure is
            retained as the exception cause.
        ValueError
            If the resolved context length is below MIN_CONTEXT_LENGTH.
        """
        if params is None:
            self._params = OpenAIEmbeddingParams()
        elif isinstance(params, OpenAIEmbeddingParams):
            self._params = params
        elif isinstance(params, dict):
            self._params = OpenAIEmbeddingParams(**params)
        else:
            raise TypeError(f"Incorrect type of 'params' parameter: {type(params)}.")
        if self._params.embedding_dimension is None:
            self._params.embedding_dimension = self._detect_embedding_dimension()
        if self._params.context_length is None:
            self._params.context_length = self._detect_context_length()
        if self._params.context_length < MIN_CONTEXT_LENGTH:
            raise ValueError(
                f"Embedding model {self.model_id} supports `context_length` {self._params.context_length}. "
                f"Minimal required value is {MIN_CONTEXT_LENGTH}. Use embedding model with greater `context_length`."
            )

    def _detect_embedding_dimension(self) -> int:
        """Infer vector width from a short, validated embedding response.

        The request uses PROBE_POLICY without input repair, server truncation,
        or an extended read timeout.

        Returns
        -------
        int
            Number of values in the probe's nonempty embedding vector.

        Raises
        ------
        RuntimeError
            If the request or response validation fails, with the original
            exception retained as the cause.
        """
        try:
            return len(self._embed_text("test", policy=PROBE_POLICY, truncate_prompt_tokens=None)[0])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to auto-detect embedding dimension for model '{self.model_id}'. "
                "Provide 'embedding_dimension' explicitly or ensure the embedding service is reachable."
            ) from exc

    def _detect_context_length(self) -> int:
        """Estimate context length using repeated-word probes and binary search.

        Search between 64 and 8192 approximate tokens until fewer than 64
        candidate sizes remain. Only a recognized input-length rejection that
        permits recovery lowers the upper bound. Each probe uses PROBE_POLICY
        with server truncation disabled.

        Returns
        -------
        int
            Largest accepted probe size, not an exact tokenizer-derived limit.
            The caller separately checks the minimum supported context length.

        Raises
        ------
        RuntimeError
            If no probe is accepted or a probe fails for another reason. The
            underlying failure is retained as the exception cause.
        """
        lo, hi, best = 64, 8192, None
        try:
            while hi - lo >= 64:
                mid = (lo + hi) // 2
                try:
                    self._embed_text("word " * mid, policy=PROBE_POLICY, truncate_prompt_tokens=None)
                except BadRequestError as exc:
                    if not is_context_overflow(exc):
                        raise
                    hi = mid - 1
                else:
                    best = mid
                    lo = mid + 1
            if best is None:
                raise ValueError("No context-length probe was accepted.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to auto-detect 'context_length' for model '{self.model_id}'. "
                "Provide 'context_length' explicitly or ensure the embedding service is reachable."
            ) from exc
        return best

    def _request_once(
        self, texts: list[str], timeout: httpx.Timeout, truncate_prompt_tokens: int | None
    ) -> list[list[float]]:
        """Make one embedding API call and validate and order its vectors.

        Parameters
        ----------
        texts : list[str]
            Input texts in the order expected in the returned vectors.
        timeout : httpx.Timeout
            Per-phase timeout limits for this attempt.
        truncate_prompt_tokens : int | None
            Provider-specific truncation option forwarded unchanged. On
            compatible backends, -1 requests truncation to the model limit and
            None disables truncation for capability probes.

        Returns
        -------
        list[list[float]]
            Nonempty, finite numeric vectors reordered by response index. Their
            width must match the configured dimension, or be consistent across
            the response when the dimension is not yet known.

        Raises
        ------
        APIStatusError
            If the server returns an unsuccessful HTTP status.
        APIConnectionError
            If the SDK reports a transport failure, including APITimeoutError.
        ValueError
            If response counts, indices, vector contents, or dimensions fail
            validation.

        Notes
        -----
        SDK retries are disabled on a client copy sharing the original HTTP
        transport. This method neither retries nor closes that copy.
        """
        response = self.client.with_options(max_retries=0, timeout=timeout).embeddings.create(
            input=texts,
            model=self.model_id,
            encoding_format="float",
            extra_body={"truncate_prompt_tokens": truncate_prompt_tokens},
        )
        if not isinstance(response.data, list) or len(response.data) != len(texts):
            raise ValueError("Embedding response count does not match the number of input texts.")
        indices = [getattr(item, "index", None) for item in response.data]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices) or sorted(indices) != list(
            range(len(texts))
        ):
            raise ValueError("Embedding response indices are missing, duplicated, or out of range.")
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors: list[list[float]] = []
        dimension = self.params.embedding_dimension
        for item in ordered:
            vector = getattr(item, "embedding", None)
            if (
                not isinstance(vector, list)
                or not vector
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
                    for value in vector
                )
            ):
                raise ValueError("Embedding response must contain nonempty, finite numeric vectors.")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise ValueError("Embedding response vector dimension does not match the model.")
            vectors.append(vector)
        return vectors

    def _embed_text(
        self,
        text_input: list[str] | str,
        *,
        policy: RequestPolicy = QUERY_POLICY,
        truncate_prompt_tokens: int | None = -1,
    ) -> list[list[float]]:
        """Normalize input and execute one batch with independent recovery state.

        Parameters
        ----------
        text_input : list[str] | str
            One string or one batch of strings. This method does not split
            lists into smaller batches.
        policy : RequestPolicy, default=QUERY_POLICY
            Recovery settings for the new executor.
        truncate_prompt_tokens : int | None, default=-1
            Provider-specific truncation option forwarded to every attempt;
            capability probes pass None to disable server truncation.

        Returns
        -------
        list[list[float]]
            Validated vectors in input order, including a one-element list for
            string input.

        Raises
        ------
        APIStatusError
            If an HTTP failure cannot be recovered within the policy limits.
        APIConnectionError
            If a transport failure cannot be recovered within the policy limits.
        ValueError
            If a response fails validation; the same request is not retried.
        TimeoutError
            If the elapsed budget expires before any failed request is recorded.
        """
        texts = [text_input] if isinstance(text_input, str) else text_input
        return EmbeddingRequestExecutor(
            request=partial(self._request_once, truncate_prompt_tokens=truncate_prompt_tokens),
            policy=policy,
            timeout=self.client.timeout,
            context_length=self.params.context_length,
            model_id=self.model_id,
        ).execute(texts)

    def embed_documents(self, texts: list[str], *, policy: RequestPolicy = INDEXING_POLICY) -> list[list[float]]:
        """Embed texts sequentially in batches of up to 128.

        Parameters
        ----------
        texts : list[str]
            Texts to embed, preserving their order.
        policy : RequestPolicy, default=INDEXING_POLICY
            Recovery settings applied independently to each batch. The default
            uses 30/60-second transient-error waits; QUERY_POLICY selects short
            waits for evaluation. Retry waits honor longer server delays and
            add up to 10% positive jitter.

        Returns
        -------
        list[list[float]]
            One validated vector per input text, in input order. Empty input
            returns an empty list without calling the embedding service.

        Raises
        ------
        APIStatusError
            If a batch's HTTP failure cannot be recovered.
        APIConnectionError
            If a batch's transport failure cannot be recovered.
        ValueError
            If a batch response fails validation.
        TimeoutError
            If a batch's elapsed budget expires before a failed request is recorded.

        Notes
        -----
        Recovery retries only the current batch. Earlier batches are retained
        during this call, but a final failure raises without returning partial
        results. Completed batches are not cached for subsequent calls.
        """
        embeddings: list[list[float]] = []
        for idx in range(0, len(texts), self._BATCH_SIZE):
            embeddings.extend(self._embed_text(texts[idx : idx + self._BATCH_SIZE], policy=policy))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a query with short transient-error waits.

        Parameters
        ----------
        query : str
            Query text to embed.

        Returns
        -------
        list[float]
            The validated query vector.

        Raises
        ------
        APIStatusError
            If an HTTP failure cannot be recovered.
        APIConnectionError
            If a transport failure cannot be recovered.
        ValueError
            If the response fails validation.
        TimeoutError
            If the elapsed budget expires before a failed request is recorded.

        Notes
        -----
        Uses QUERY_POLICY, including the read-timeout fallback. Its short
        transient-error waits do not impose a short overall request deadline.
        """
        return self._embed_text(query)[0]
