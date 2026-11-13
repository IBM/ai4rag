# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Recovery policies and execution for OpenAI-compatible embedding requests."""

from collections.abc import Callable
from dataclasses import dataclass
from email.utils import mktime_tz, parsedate_tz
from math import isfinite
from random import uniform
from time import monotonic, sleep, time
from typing import Literal

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError

from ai4rag import logger
from ai4rag.utils.constants import TokenEstimation

_TRUNCATION_MARGINS = (0.05, 0.10)


@dataclass(frozen=True)
class RequestPolicy:
    """Recovery limits shared by all attempts for one embedding batch.

    Parameters
    ----------
    retry_delays : tuple[float, ...], default=(0.5, 1.0)
        Minimum waits in seconds for successive transient-error retries. The
        tuple length limits those retries; server hints and jitter can extend
        each wait. Input repairs and timeout fallback do not consume these waits.
    max_attempts : int, default=6
        Maximum request callback invocations, including the initial attempt,
        input repairs, and timeout fallback. Must be positive.
    max_elapsed : float, default=2400.0
        Elapsed budget in seconds for one execution. Checked before requests
        and waits; it is not a hard cancellation deadline for an active request.
    allow_truncation : bool, default=True
        Allow input repair after confirmed context overflow.
    fallback_read_timeout : float | None, default=1200.0
        Read timeout in seconds for one fallback after a read or unknown-phase
        timeout. Applied only when the original read limit is finite and
        shorter. None disables the fallback.
    """

    retry_delays: tuple[float, ...] = (0.5, 1.0)
    max_attempts: int = 6
    max_elapsed: float = 2400.0
    allow_truncation: bool = True
    fallback_read_timeout: float | None = 1200.0


INDEXING_POLICY = RequestPolicy(retry_delays=(30.0, 60.0))
QUERY_POLICY = RequestPolicy()
PROBE_POLICY = RequestPolicy(max_attempts=3, max_elapsed=600.0, allow_truncation=False, fallback_read_timeout=None)


def is_context_overflow(error: BadRequestError) -> bool:
    """Identify a confirmed input-length rejection that permits recovery.

    Parameters
    ----------
    error : BadRequestError
        The embedding service's rejection.

    Returns
    -------
    bool
        True if a recognized error code or message identifies context overflow
        and the response does not contain ``x-should-retry: false``. This does
        not check whether a policy allows truncation or the input can be shortened.
    """
    if error.response.headers.get("x-should-retry") == "false":
        return False
    message = str(error).lower()
    return error.code in {"context_length_exceeded", "input_too_long"} or any(
        phrase in message for phrase in ("maximum context length", "input too long", "maximum input length exceeded")
    )


def _classify_error(error: APIStatusError | APIConnectionError) -> Literal["truncate", "timeout", "retry", "fail"]:
    """Select a recovery category without applying policy limits.

    Parameters
    ----------
    error : APIStatusError | APIConnectionError
        SDK failure from one request attempt. APITimeoutError is handled before
        its APIConnectionError parent class.

    Returns
    -------
    Literal["truncate", "timeout", "retry", "fail"]
        ``truncate`` for recognized context overflow; ``timeout`` for read or
        unknown-phase timeouts; ``retry`` for connect/write/pool timeouts, other
        connection failures, and HTTP 408/409/429/500/502/503/504. All other
        errors, including status responses disabling retries, yield ``fail``.
    """
    if isinstance(error, APIStatusError) and error.response.headers.get("x-should-retry") == "false":
        return "fail"
    if isinstance(error, APITimeoutError):
        return (
            "retry"
            if isinstance(error.__cause__, (httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout))
            else "timeout"
        )
    if isinstance(error, BadRequestError):
        return "truncate" if is_context_overflow(error) else "fail"
    if isinstance(error, APIConnectionError) or error.status_code in {408, 409, 429, 500, 502, 503, 504}:
        return "retry"
    return "fail"


def _retry_delay(error: APIStatusError | APIConnectionError, fallback: float) -> float:
    """Calculate a retry wait from the configured minimum and server hints.

    Parameters
    ----------
    error : APIStatusError | APIConnectionError
        Failure whose response may supply retry headers. Connection errors
        have no server hint.
    fallback : float
        Nonnegative minimum delay in seconds selected from the policy.

    Returns
    -------
    float
        The larger of the configured minimum and the first valid server hint,
        plus uniform jitter from zero to 10% of that value. Positive finite
        ``retry-after-ms`` values take precedence over ``Retry-After`` seconds
        or dates. Invalid and expired hints are ignored. This function does
        not check the elapsed budget or perform the wait.
    """
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
class _RequestState:
    """Mutable recovery state belonging to one execute call.

    Parameters
    ----------
    texts : list[str]
        Current input payload, replaced when truncation repairs it.
    timeout : httpx.Timeout
        Current timeout limits, before the per-attempt remaining-budget cap.
    retries : int, default=0
        Number of transient-error waits already consumed.
    truncations : int, default=0
        Number of input repairs already applied.
    """

    texts: list[str]
    timeout: httpx.Timeout
    retries: int = 0
    truncations: int = 0


class EmbeddingRequestExecutor:
    """Execute an embedding batch with bounded recovery and isolated attempt state.

    Parameters
    ----------
    request : Callable[[list[str], httpx.Timeout], list[list[float]]]
        Callback accepting the current input texts and timeout limits. It must
        make one API call with SDK retries disabled and validate the response.
        The executor returns successful callback output without inspecting it.
    policy : RequestPolicy
        Recovery settings for this workload.
    timeout : httpx.Timeout | float | None
        Initial per-phase limits or a timeout in seconds for all phases. None
        disables initial phase limits. Each execution copies these settings;
        every attempt caps even unlimited phases to the remaining elapsed budget.
    context_length : int | None
        Known model context length for character-based input repairs.
    model_id : str
        Model identifier included in recovery logs.

    Notes
    -----
    Payload and timeout adjustments, counters, and elapsed time are local to
    each execute call. Concurrent use still requires a thread-safe request
    callback. The executor does not own or close the callback's client.
    """

    def __init__(
        self,
        *,
        request: Callable[[list[str], httpx.Timeout], list[list[float]]],
        policy: RequestPolicy,
        timeout: httpx.Timeout | float | None,
        context_length: int | None,
        model_id: str,
    ) -> None:
        """Store the request configuration and copy the initial timeout limits."""
        self._request = request
        self._policy = policy
        self._timeout = httpx.Timeout(timeout)
        self._context_length = context_length
        self._model_id = model_id

    def execute(self, texts: list[str]) -> list[list[float]]:
        """Run one batch, preserving repaired input and timeout across attempts.

        Each invocation creates its own recovery state. Only SDK status and
        connection errors enter recovery. Other callback exceptions, including
        response-validation errors, propagate immediately.

        Parameters
        ----------
        texts : list[str]
            Texts to embed; the caller's list is not modified.

        Returns
        -------
        list[list[float]]
            The successful callback result, returned without further validation.

        Raises
        ------
        APIStatusError
            If a status error is not recoverable, recovery limits are exhausted,
            or its required retry wait would reach the elapsed budget.
        APIConnectionError
            If a transport error cannot be recovered under the same limits.
        TimeoutError
            If the elapsed budget expires before a failed request is recorded.
            After a failure, budget exhaustion re-raises that last SDK error.
        AssertionError
            If the policy's max_attempts is nonpositive.

        Notes
        -----
        The elapsed budget is checked before requests and waits, and each
        timeout phase is capped to the remaining budget. It does not cancel
        an in-flight request or provide a hard wall-clock deadline.
        """
        original = list(texts)
        state = _RequestState(texts=original, timeout=httpx.Timeout(self._timeout))
        started = monotonic()
        last_error: APIStatusError | APIConnectionError | None = None
        for attempt in range(self._policy.max_attempts):
            remaining = self._policy.max_elapsed - (monotonic() - started)
            if remaining <= 0:
                raise last_error or TimeoutError("Embedding recovery time budget exhausted.")
            try:
                return self._request(state.texts, self._bounded_timeout(state.timeout, remaining))
            except (APIStatusError, APIConnectionError) as exc:
                last_error = exc
                if attempt + 1 == self._policy.max_attempts:
                    raise
                remaining = self._policy.max_elapsed - (monotonic() - started)
                if not self._recover(state, original, exc, remaining):
                    raise
        raise AssertionError("Embedding attempt budget must be positive.")

    @staticmethod
    def _bounded_timeout(timeout: httpx.Timeout, remaining: float) -> httpx.Timeout:
        """Copy timeout limits, capping each phase to the remaining budget.

        Parameters
        ----------
        timeout : httpx.Timeout
            Current phase limits; this object is not modified.
        remaining : float
            Positive seconds remaining before this request attempt.

        Returns
        -------
        httpx.Timeout
            Limits no greater than remaining. An unlimited phase becomes
            limited to remaining seconds for this attempt.
        """
        return httpx.Timeout(
            **{
                key: min(value, remaining) if value is not None else remaining
                for key, value in timeout.as_dict().items()
            }
        )

    def _recover(
        self, state: _RequestState, original: list[str], error: APIStatusError | APIConnectionError, remaining: float
    ) -> bool:
        """Dispatch a failure to one recovery action.

        Parameters
        ----------
        state : _RequestState
            Execution state that the selected action may update.
        original : list[str]
            Original batch used to calculate each truncation independently.
        error : APIStatusError | APIConnectionError
            Failure from the latest request attempt.
        remaining : float
            Remaining budget in seconds, used to decide whether a retry wait fits.

        Returns
        -------
        bool
            True if an action prepares another attempt; False if the caller
            should propagate the error. A prepared attempt still needs the
            executor's next elapsed-budget check.
        """
        match _classify_error(error):
            case "truncate":
                return self._truncate(state, original)
            case "timeout":
                return self._extend_read_timeout(state)
            case "retry":
                return self._wait_for_retry(state, error, remaining)
            case _:
                return False

    def _truncate(self, state: _RequestState, original: list[str]) -> bool:
        """Apply the next character-based input repair after confirmed overflow.

        Parameters
        ----------
        state : _RequestState
            Current payload and count of previously applied truncations.
        original : list[str]
            Unrepaired texts. Each margin is applied to this original batch.

        Returns
        -------
        bool
            True after replacing state.texts and incrementing state.truncations.
            False if repair is disabled, context length is unknown, both margins
            are exhausted, or the payload would be unchanged.

        Notes
        -----
        Repairs target 95%, then 90% of the configured context length using
        CHARS_PER_TOKEN. Shorter texts remain intact; the character estimate
        does not guarantee that the service will accept the repaired batch.
        """
        if (
            not self._policy.allow_truncation
            or self._context_length is None
            or state.truncations == len(_TRUNCATION_MARGINS)
        ):
            return False
        margin = _TRUNCATION_MARGINS[state.truncations]
        max_chars = int(self._context_length * (1 - margin) * TokenEstimation.CHARS_PER_TOKEN)
        repaired = [text[:max_chars] for text in original]
        if repaired == state.texts:
            return False
        state.texts = repaired
        state.truncations += 1
        logger.warning(
            "Truncating embedding input by %.0f%% of context length. Model: %s.", margin * 100, self._model_id
        )
        return True

    def _extend_read_timeout(self, state: _RequestState) -> bool:
        """Apply the policy's read-timeout fallback when it extends a finite limit.

        Parameters
        ----------
        state : _RequestState
            Execution state whose timeout may be replaced.

        Returns
        -------
        bool
            True after extending the read limit. False if fallback is disabled
            or the current read limit is unlimited or already at least as long.
            Connect, write, and pool limits are preserved.

        Notes
        -----
        Keeping the extended read limit in state prevents a second extension.
        The request attempt may still cap it to the remaining elapsed budget.
        """
        fallback = self._policy.fallback_read_timeout
        if fallback is None or state.timeout.read is None or state.timeout.read >= fallback:
            return False
        state.timeout = httpx.Timeout(**(state.timeout.as_dict() | {"read": fallback}))
        logger.warning("Extending embedding read timeout to %.0fs once. Model: %s.", fallback, self._model_id)
        return True

    def _wait_for_retry(
        self, state: _RequestState, error: APIStatusError | APIConnectionError, remaining: float
    ) -> bool:
        """Consume one transient-error retry and wait if its delay fits the budget.

        Parameters
        ----------
        state : _RequestState
            Execution state whose retry counter may be incremented.
        error : APIStatusError | APIConnectionError
            Latest failure, used for server delay hints and recovery logging.
        remaining : float
            Remaining elapsed budget in seconds when recovery is considered.

        Returns
        -------
        bool
            True after incrementing the counter and sleeping. False if no
            retry delays remain or the delay would reach or exceed remaining.
            The executor checks elapsed time again before issuing the request.
        """
        if state.retries == len(self._policy.retry_delays):
            return False
        delay = _retry_delay(error, self._policy.retry_delays[state.retries])
        if delay >= remaining:
            return False
        state.retries += 1
        logger.warning(
            "Embedding request failed (%s, status=%s); retry %d/%d in %.1fs. Model: %s.",
            type(error).__name__,
            getattr(error, "status_code", None),
            state.retries,
            len(self._policy.retry_delays),
            delay,
            self._model_id,
        )
        sleep(delay)
        return True
