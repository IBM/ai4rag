# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Generation evidence reuse and context-overflow reporting without judge compression."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pandas as pd
import pytest
from langchain_openai import ChatOpenAI
from openai import BadRequestError, OpenAI
from ragas.callbacks import ChainType

from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.utils import build_evaluation_data
from ai4rag.evaluator.base_evaluator import EvaluationData
from ai4rag.evaluator.metric import Metrics
from ai4rag.evaluator.ragas_diagnostics import RagasFailureTrace, failure_reason
from ai4rag.evaluator.ragas_evaluator import RagasEvaluator
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.retrieval.document_compressor import LLMDocumentCompressor
from ai4rag.rag.template.agentic_rag_template import AgenticRAG

METRICS = [Metrics.RAGAS_FAITHFULNESS, Metrics.RAGAS_CONTEXT_RECALL]


def scores(values, reason=None):
    frame = pd.DataFrame(values)
    frame.attrs["failures"] = (
        [{"row_index": 0, "metric": "faithfulness", "reason": reason, "error_type": "BadRequestError"}]
        if reason
        else []
    )
    return frame


@pytest.mark.parametrize("reason", ["context_overflow", "bad_request", "parse_error", "timeout"])
def test_failed_metrics_are_reported_without_compression_or_sample_retries(mocker, reason):
    evaluator = RagasEvaluator(MagicMock(), MagicMock())
    samples = [
        EvaluationData(question="q", answer="a", contexts=["evidence " * 100], question_id="q1"),
        EvaluationData(question="q2", answer="a2", contexts=["other"], question_id="q2"),
    ]
    run = mocker.patch.object(
        evaluator,
        "_run_ragas",
        return_value=scores({"faithfulness": [float("nan"), 0.9], "context_recall": [0.8, 0.7]}, reason),
    )
    compress = mocker.spy(LLMDocumentCompressor, "compress")

    result = evaluator.evaluate_metrics(samples, METRICS)

    run.assert_called_once()
    compress.assert_not_called()
    assert result["question_scores"][0]["metrics"] == [{"name": "context_recall", "evaluator": "ragas", "value": 0.8}]
    assert [m["value"] for m in result["question_scores"][1]["metrics"]] == [0.9, 0.7]
    detail = result["diagnostics"][0]
    assert "evaluated_contexts" not in detail
    assert detail["failures"][0]["reason"] == reason
    assert detail["missing_metrics"] == ["faithfulness"]
    assert detail["complete"] is False


def test_failure_trace_attributes_suppressed_errors_without_content():
    trace = RagasFailureTrace()
    row, metric, prompt = uuid4(), uuid4(), uuid4()
    trace.on_chain_start({"name": "row 2"}, {}, run_id=row, metadata={"type": ChainType.ROW, "row_index": 2})
    trace.on_chain_start(
        {"name": "faithfulness"}, {}, run_id=metric, parent_run_id=row, metadata={"type": ChainType.METRIC}
    )
    trace.on_chain_start({"name": "prompt"}, {}, run_id=prompt, parent_run_id=metric)
    error = BadRequestError(
        "maximum context length: private response",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.test")),
        body=None,
    )
    trace.on_chain_error(error, run_id=prompt)
    trace.on_chain_error(error, run_id=metric)
    assert trace.failures == [
        {"row_index": 2, "metric": "faithfulness", "reason": "context_overflow", "error_type": "BadRequestError"}
    ]
    assert "private" not in json.dumps(trace.failures)
    unrelated = BadRequestError("unsupported max_tokens", response=error.response, body=None)
    assert failure_reason(unrelated) == "bad_request"


@pytest.mark.parametrize("generation_compressed", [True, False])
def test_single_32k_endpoint_reuses_generation_evidence_without_judge_compression(generation_compressed):
    """Reuse generation excerpts; report judge-only overflow without another extraction pass."""
    calls = []
    evaluating = False
    evidence = "The claim is supported. " * (178 if generation_compressed else 5050)
    documents = [
        AI4RAGChunk(text=evidence, metadata={"document_id": f"source{i}"})
        for i in range(33 if generation_compressed else 1)
    ]

    def respond(request):
        payload = json.loads(request.content)
        text = "\n".join(m.get("content") or "" for m in payload["messages"])
        kind = (
            "compression"
            if text.startswith("Identify one contiguous excerpt")
            else (
                "generation"
                if payload["messages"][0]["role"] == "system"
                else (
                    "statements"
                    if "Break down each sentence" in text
                    else "faithfulness" if "judge the faithfulness" in text else "context_recall"
                )
            )
        )
        input_tokens = (len(text) + 3) // 4
        overflow = input_tokens + payload["max_completion_tokens"] > 32000
        calls.append((kind, overflow, text, evaluating))
        if overflow:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "maximum context length is 32000",
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                    }
                },
            )
        content = {
            "generation": "The claim is supported.",
            "compression": json.dumps({"excerpt": "The claim is supported."}),
            "statements": json.dumps({"statements": ["The claim is supported."]}),
            "faithfulness": json.dumps(
                {
                    "statements": [
                        {"statement": "The claim is supported.", "reason": "Present in evidence", "verdict": 1}
                    ]
                }
            ),
            "context_recall": json.dumps(
                {
                    "classifications": [
                        {"statement": "The claim is supported.", "reason": "Present in evidence", "attributed": 1}
                    ]
                }
            ),
        }[kind]
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen-test",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
                ],
            },
        )

    client = OpenAI(
        api_key="test",
        base_url="https://endpoint.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    model = OpenAIFoundationModel(
        client=client,
        model_id="qwen-test",
        params={"max_completion_tokens": 1024},
        system_message_text="Answer from context.",
        user_message_text="{question}\n{reference_documents}",
        context_template_text="{document}",
    )
    # Simulate legacy generation with no knowledge of the judge's different prompt shape.
    retriever = SimpleNamespace(retrieve=lambda *_args, **_kwargs: documents)
    chat_model = ChatOpenAI(
        model="qwen-test",
        api_key="test",
        base_url="https://endpoint.test/v1",
        http_client=client._client,
        max_completion_tokens=1024,
        use_responses_api=False,
    )
    generation = AgenticRAG(
        model,
        retriever,
        max_retrieval_steps=1,
        chat_model=chat_model,
    ).generate("What is supported?")
    assert generation["answer"] == "The claim is supported."
    assert generation["conversation"]["context_compression_fallback"] is generation_compressed
    benchmark = BenchmarkData(
        pd.DataFrame(
            {
                "question": ["What is supported?"],
                "correct_answers": [["The claim is supported."]],
                "correct_answer_document_keys": [["source0"]],
            }
        )
    )
    samples = build_evaluation_data(benchmark, [generation])
    contexts = [chunk.text for chunk in generation["reference_documents"]]
    assert samples[0].contexts == contexts
    evaluating = True
    result = RagasEvaluator(model, MagicMock(), max_workers=1).evaluate_metrics(samples, METRICS)

    assert not any(kind == "compression" and judging for kind, _, _, judging in calls)
    for metric in ("faithfulness", "context_recall"):
        metric_calls = [(overflow, text) for kind, overflow, text, _ in calls if kind == metric]
        assert len(metric_calls) == 1
        assert metric_calls[0][0] is not generation_compressed
        assert all(context in metric_calls[0][1] for context in contexts)
    if generation_compressed:
        assert contexts == ["The claim is supported."] * len(documents)
        assert sum(kind == "compression" for kind, _, _, _ in calls) == len(documents)
        assert result["diagnostics"] == []
        assert [metric["value"] for metric in result["question_scores"][0]["metrics"]] == [1.0, 1.0]
    else:
        assert contexts == [evidence]
        assert not any(kind == "compression" for kind, _, _, _ in calls)
        assert result["question_scores"][0]["metrics"] == []
        detail = result["diagnostics"][0]
        assert "evaluated_contexts" not in detail
        assert detail["missing_metrics"] == ["faithfulness", "context_recall"]
        assert {failure["reason"] for failure in detail["failures"]} == {"context_overflow"}


def test_parser_failure_keeps_existing_repairs_and_does_not_trigger_compression():
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "invalid JSON"}}
                ],
            },
        )

    model = OpenAIFoundationModel(
        client=OpenAI(
            api_key="test",
            base_url="https://endpoint.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ),
        model_id="test",
    )
    result = RagasEvaluator(model, MagicMock()).evaluate_metrics(
        [EvaluationData(question="q", answer="a", contexts=["evidence"], question_id="q")], [Metrics.RAGAS_FAITHFULNESS]
    )
    assert len(calls) > 1
    assert result["diagnostics"][0]["failures"][0]["reason"] == "parse_error"
    assert result["diagnostics"][0]["complete"] is False
