# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Classify RAGAS judge failures without retaining endpoint response content."""

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
