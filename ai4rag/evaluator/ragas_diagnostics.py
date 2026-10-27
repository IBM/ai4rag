# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Capture RAGAS job failures before its executor replaces them with NaN."""

from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from openai import BadRequestError


def failure_reason(error: BaseException) -> str:
    """Classify an evaluation failure without retaining endpoint response content."""
    if isinstance(error, BadRequestError):
        message = str(error).lower()
        if error.code == "context_length_exceeded" or "maximum context length" in message:
            return "context_overflow"
        return "bad_request"
    if isinstance(error, TimeoutError):
        return "timeout"
    return {
        "RagasOutputParserException": "parse_error",
        "OutputParserException": "parse_error",
        "LLMDidNotFinishException": "incomplete_completion",
        "APITimeoutError": "timeout",
    }.get(type(error).__name__, "evaluation_error")


class RagasFailureTrace(BaseCallbackHandler):
    """Associate suppressed failures with their row and metric callback ancestors."""

    run_inline = True

    def __init__(self) -> None:
        self.failures: list[dict[str, Any]] = []
        self._runs: dict[UUID, tuple[UUID | None, int | None, str | None]] = {}
        self._lock = Lock()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Keep only the callback ancestry needed to identify a failed metric."""
        metadata = metadata or {}
        chain_type = getattr(metadata.get("type"), "value", metadata.get("type"))
        row = metadata.get("row_index") if chain_type == "row" else None
        metric = serialized.get("name") if chain_type == "metric" else None
        with self._lock:
            self._runs[run_id] = (parent_run_id, row, metric)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Retain structured failure reasons even with ``raise_exceptions=False``."""
        with self._lock:
            row = metric = None
            current = run_id
            while current in self._runs:
                current, run_row, run_metric = self._runs[current]
                row = run_row if run_row is not None else row
                metric = run_metric or metric
            if row is not None and metric:
                failure = {
                    "row_index": row,
                    "metric": metric,
                    "reason": failure_reason(error),
                    "error_type": type(error).__name__,
                }
                if failure not in self.failures:
                    self.failures.append(failure)
