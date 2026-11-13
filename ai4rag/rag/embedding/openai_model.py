# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from email.utils import mktime_tz, parsedate_tz
from math import isfinite
from random import uniform
from time import monotonic, sleep, time
from typing import Literal

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, OpenAI

from ai4rag import logger
from ai4rag.utils.constants import TokenEstimation

from .base_model import BaseEmbeddingModel

__all__ = ["OpenAIEmbeddingModel", "OpenAIEmbeddingParams"]

MIN_CONTEXT_LENGTH = 700
_TRUNCATION_MARGINS = (0.05, 0.10)
_FALLBACK_TIMEOUT = 1200.0


@dataclass(frozen=True)
class _RequestPolicy:
    """Limits shared by all recovery actions for one embedding batch."""

    retry_delays: tuple[float, ...] = (0.5, 1.0)
    max_attempts: int = 6
    max_elapsed: float = 2400.0
    allow_truncation: bool = True
    fallback_read_timeout: float | None = _FALLBACK_TIMEOUT


_INDEXING_POLICY = _RequestPolicy(retry_delays=(30.0, 60.0))
_QUERY_POLICY = _RequestPolicy()
_PROBE_POLICY = _RequestPolicy(max_attempts=3, max_elapsed=600.0, allow_truncation=False, fallback_read_timeout=None)


def _classify_error(error: APIStatusError | APIConnectionError) -> Literal["truncate", "timeout", "retry", "fail"]:
    """Distinguish input repair, slow reads, temporary failures, and permanent errors."""
    if isinstance(error, APIStatusError) and error.response.headers.get("x-should-retry") == "false":
        return "fail"
    if isinstance(error, APITimeoutError):
        return (
            "retry"
            if isinstance(error.__cause__, (httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout))
            else "timeout"
        )
    if isinstance(error, BadRequestError):
        message = str(error).lower()
        if error.code in {"context_length_exceeded", "input_too_long"} or any(
            phrase in message
            for phrase in ("maximum context length", "input too long", "maximum input length exceeded")
        ):
            return "truncate"
        return "fail"
    if isinstance(error, APIConnectionError) or error.status_code in {408, 409, 429, 500, 502, 503, 504}:
        return "retry"
    return "fail"


def _retry_delay(error: APIStatusError | APIConnectionError, fallback: float) -> float:
    """Respect server delays and add positive jitter without shortening the cooldown."""
    if isinstance(error, APIStatusError):
        headers = error.response.headers
        for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            value = headers.get(name)
            if value is None:
                continue
            try:
                delay = float(value) * scale
            except ValueError:
                try:
                    date = parsedate_tz(value) if name == "retry-after" else None
                    delay = mktime_tz(date) - time() if date else 0.0
                except (ValueError, OverflowError):
                    continue
            if isfinite(delay) and delay > 0:
                fallback = max(fallback, delay)
                break
    return fallback + uniform(0.0, fallback * 0.1)


@dataclass
class OpenAIEmbeddingParams:
    """Parameters to be used to create embeddings."""

    embedding_dimension: int | None = None
    context_length: int | None = None


class OpenAIEmbeddingModel(BaseEmbeddingModel[OpenAI, OpenAIEmbeddingParams]):
    """Create validated embeddings with bounded recovery through an OpenAI-compatible client."""

    _BATCH_SIZE = 128

    def __init__(self, client: OpenAI, model_id: str, params: dict | OpenAIEmbeddingParams | None = None):
        super().__init__(client=client, model_id=model_id, params=params)

    @property
    def params(self) -> OpenAIEmbeddingParams:
        """Get model params."""
        return self._params

    @params.setter
    def params(self, params: dict | OpenAIEmbeddingParams | None) -> None:
        """Set model params and detect missing capabilities."""
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
        """Probe the dimension without attempting truncation before capabilities are known."""
        try:
            return len(self._embed_text("test", policy=_PROBE_POLICY, truncate_prompt_tokens=None)[0])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to auto-detect embedding dimension for model '{self.model_id}'. "
                "Provide 'embedding_dimension' explicitly or ensure the embedding service is reachable."
            ) from exc

    def _detect_context_length(self) -> int:
        """Search 64–8192 approximate tokens; only length rejections narrow the range.

        Probes disable server truncation and share the short recovery policy.
        The search stops when fewer than 64 candidate tokens remain.
        """
        lo, hi, best = 64, 8192, None
        try:
            while hi - lo >= 64:
                mid = (lo + hi) // 2
                try:
                    self._embed_text("word " * mid, policy=_PROBE_POLICY, truncate_prompt_tokens=None)
                except BadRequestError as exc:
                    if _classify_error(exc) != "truncate":
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
        """Make exactly one request and reject incomplete or malformed vectors."""
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
        vectors = []
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

    def _truncate_text(self, text: str, margin: float) -> str:
        """Truncate using the configured context length and a character-based token estimate."""
        max_chars = int(self.params.context_length * (1 - margin) * TokenEstimation.CHARS_PER_TOKEN)
        return text[:max_chars]

    def _embed_text(
        self,
        text_input: list[str] | str,
        *,
        policy: _RequestPolicy = _QUERY_POLICY,
        truncate_prompt_tokens: int | None = -1,
    ) -> list[list[float]]:
        """Own all request attempts, retaining repaired input and timeout state.

        A batch permits at most six HTTP attempts: an initial request, two
        temporary-failure retries, two truncations, and one read-timeout
        extension. Probes permit three attempts without input or timeout repair.
        The elapsed budget is checked before waits and requests; each timeout
        phase is capped to the remaining budget.
        """
        text_input = [text_input] if isinstance(text_input, str) else text_input
        texts = text_input
        timeout = httpx.Timeout(self.client.timeout)
        retries = truncations = 0
        started = monotonic()
        last_error = None
        for attempt in range(policy.max_attempts):
            remaining = policy.max_elapsed - (monotonic() - started)
            if remaining <= 0:
                raise last_error or TimeoutError("Embedding recovery time budget exhausted.")
            try:
                return self._request_once(
                    texts,
                    httpx.Timeout(
                        **{
                            key: min(value, remaining) if value is not None else remaining
                            for key, value in timeout.as_dict().items()
                        }
                    ),
                    truncate_prompt_tokens,
                )
            except (APIStatusError, APIConnectionError) as exc:
                last_error = exc
                if attempt + 1 == policy.max_attempts:
                    raise
                match _classify_error(exc):
                    case "truncate" if policy.allow_truncation and self.params.context_length is not None:
                        if truncations == len(_TRUNCATION_MARGINS):
                            raise
                        repaired = [self._truncate_text(text, _TRUNCATION_MARGINS[truncations]) for text in text_input]
                        if repaired == texts:
                            raise
                        texts = repaired
                        logger.warning(
                            "Truncating embedding input by %.0f%% of context length. Model: %s.",
                            _TRUNCATION_MARGINS[truncations] * 100,
                            self.model_id,
                        )
                        truncations += 1
                    case "timeout" if policy.fallback_read_timeout is not None:
                        # The extended read limit also prevents extending a second time.
                        if timeout.read is None or timeout.read >= policy.fallback_read_timeout:
                            raise
                        timeout = httpx.Timeout(**(timeout.as_dict() | {"read": policy.fallback_read_timeout}))
                        logger.warning(
                            "Extending embedding read timeout to %.0fs once. Model: %s.", timeout.read, self.model_id
                        )
                    case "retry" if retries < len(policy.retry_delays):
                        delay = _retry_delay(exc, policy.retry_delays[retries])
                        remaining = policy.max_elapsed - (monotonic() - started)
                        if delay >= remaining:
                            raise
                        retries += 1
                        logger.warning(
                            "Embedding request failed (%s, status=%s); retry %d/%d in %.1fs. Model: %s.",
                            type(exc).__name__,
                            getattr(exc, "status_code", None),
                            retries,
                            len(policy.retry_delays),
                            delay,
                            self.model_id,
                        )
                        sleep(delay)
                    case _:
                        raise
        raise AssertionError("Embedding attempt budget must be positive.")

    def embed_documents(self, texts: list[str], *, for_indexing: bool = True) -> list[list[float]]:
        """Embed texts sequentially in batches of up to 128.

        Parameters
        ----------
        texts : list[str]
            Texts to embed, preserving their order.
        for_indexing : bool, default=True
            Use 30/60-second transient-error waits for indexing. False selects
            short waits for evaluation. Both add up to 10% positive jitter and
            honor longer server delays. Completed batches remain in memory.

        Returns
        -------
        list[list[float]]
            One validated vector per input text, in input order.
        """
        policy = _INDEXING_POLICY if for_indexing else _QUERY_POLICY
        embeddings = []
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
        """
        return self._embed_text(query)[0]
