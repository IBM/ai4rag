# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Ensure failure diagnostics remain bounded and contain no free-form content."""

import json

from ai4rag.rag.template.agent_trace import AgentTrace


def test_content_and_unknown_identifiers_are_never_retained_or_logged(caplog):
    secret = "SECRET_TOKEN user@example.com\nforged log"
    trace = AgentTrace()
    trace.on_chat_model_start({"name": secret}, [[secret]], metadata={"token": secret})
    trace.on_llm_end(secret)
    trace.on_tool_start({"name": secret}, secret)
    trace.on_tool_end(secret)
    trace.on_llm_error(RuntimeError(secret))
    trace.on_tool_error(RuntimeError(secret))
    trace.record_retrieval(5, 2, 1)
    trace.log_failure(type(secret, (Exception,), {})(secret), 3)

    record = caplog.records[-1]
    summary = json.loads(record.getMessage().split(" ", 1)[1])
    assert summary["error_type"] == "OtherError"
    assert summary["model_calls"] == 1
    assert summary["tool_calls"] == {"retriever": 0, "rewrite_query": 0, "other": 1}
    assert summary["events"] == [
        {"event": "model_start"},
        {"event": "model_end"},
        {"event": "tool_start", "tool": "other"},
        {"event": "tool_end", "outcome": "returned"},
        {"event": "model_error"},
        {"event": "tool_error"},
    ]
    assert summary["retrieval_results"] == [
        {"retrieval_index": 0, "retrieved_chunk_count": 5, "retained_chunk_count": 2, "new_chunk_count": 1}
    ]
    assert "SECRET" not in caplog.text
    assert "user@example.com" not in caplog.text
    assert record.exc_info is None
    assert secret not in repr(vars(trace))


def test_events_and_retrieval_results_are_bounded_but_totals_cover_all_calls(caplog):
    trace = AgentTrace()
    for _ in range(40):
        trace.on_chat_model_start({}, [])
        trace.record_retrieval(2, 2, 1)
    trace.on_tool_end("No new relevant documents were found for this query.")
    trace.log_failure(RuntimeError("not logged"), 1)

    summary = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert summary["model_calls"] == 40
    assert summary["new_chunk_count"] == 40
    assert len(summary["events"]) == len(summary["retrieval_results"]) == 32
    assert summary["events_omitted"] == 9
    assert summary["retrieval_results_omitted"] == 8
    assert summary["retrieval_results"][0]["retrieval_index"] == 8
    assert summary["events"][-1] == {"event": "tool_end", "outcome": "no_new_documents"}
