# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Invocation-scoped, content-free diagnostics emitted only on agent failure."""

import json
from collections import Counter, deque
from threading import Lock
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler

from ai4rag import logger

_ERROR_TYPES = {
    "GraphRecursionError",
    "InternalServerError",
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "ValueError",
    "RuntimeError",
    "TypeError",
}
_OBSERVATIONS = {
    "Retrieval limit reached. Answer using the documents already found.": "retrieval_limit_reached",
    "No new relevant documents were found for this query.": "no_new_documents",
    "No new relevant excerpts were found for this query.": "no_new_documents",
}


class AgentTrace(BaseCallbackHandler):
    """Count calls and retain bounded events without storing model or document content."""

    run_inline = True

    def __init__(self) -> None:
        self.compression_retry = False
        self._events: deque[dict] = deque(maxlen=32)
        self._retrievals: deque[dict] = deque(maxlen=32)
        self._counts: Counter = Counter()
        self._lock = Lock()

    def _record(self, event: str, **labels: str) -> None:
        with self._lock:
            self._counts[event] += 1
            self._events.append({"event": event, **labels})

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        """Count model calls, including nested query rewrites."""
        self._record("model_start")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record completion without inspecting the response content."""
        self._record("model_end")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Record a model failure without its message or response body."""
        self._record("model_error")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        """Record only the two known tool names; ignore their arguments."""
        name = serialized.get("name")
        name = name if name in {"retriever", "rewrite_query"} else "other"
        with self._lock:
            self._counts[name] += 1
        self._record("tool_start", tool=name)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Recognize fixed budget/empty-result observations without retaining text."""
        content = getattr(output, "content", output)
        outcome = _OBSERVATIONS.get(content, "returned") if isinstance(content, str) else "returned"
        self._record("tool_end", outcome=outcome)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Record a tool failure without its exception text."""
        self._record("tool_error")

    def record_retrieval(self, retrieved: int, retained: int, new: int) -> None:
        """Record counts before compression, after compression and after deduplication."""
        with self._lock:
            self._retrievals.append(
                {
                    "retrieval_index": self._counts["retrievals"],
                    "retrieved_chunk_count": retrieved,
                    "retained_chunk_count": retained,
                    "new_chunk_count": new,
                }
            )
            self._counts["retrievals"] += 1
            self._counts["new_chunks"] += new

    def log_failure(self, error: Exception, initial_chunk_count: int) -> None:
        """Emit one safe record when the agent cannot recover from a failure."""
        name = type(error).__name__
        with self._lock:
            summary = {
                "diagnostic_id": uuid4().hex,
                "error_type": name if name in _ERROR_TYPES else "OtherError",
                "initial_document_count": initial_chunk_count,
                "model_calls": self._counts["model_start"],
                "tool_calls": {tool: self._counts[tool] for tool in ("retriever", "rewrite_query", "other")},
                "context_compression_retry": self.compression_retry,
                "new_chunk_count": self._counts["new_chunks"],
                "retrieval_results": list(self._retrievals),
                "retrieval_results_omitted": max(0, self._counts["retrievals"] - len(self._retrievals)),
                "events": list(self._events),
                "events_omitted": sum(
                    self._counts[event]
                    for event in ("model_start", "model_end", "model_error", "tool_start", "tool_end", "tool_error")
                )
                - len(self._events),
            }
        logger.warning("RAG_FAILURE_TRACE %s", json.dumps(summary))
