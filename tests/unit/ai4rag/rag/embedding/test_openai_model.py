# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Embedding recovery, capability detection, and response-contract regressions."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, InternalServerError, OpenAI

from ai4rag.rag.embedding.openai_model import (
    MIN_CONTEXT_LENGTH,
    OpenAIEmbeddingModel,
    OpenAIEmbeddingParams,
)
from ai4rag.rag.embedding.request_executor import INDEXING_POLICY, QUERY_POLICY, RequestPolicy

REQUEST = httpx.Request("POST", "https://test/v1/embeddings")
VECTOR = [0.1, 0.2, 0.3]


def response(vectors, indices=None):
    indices = range(len(vectors)) if indices is None else indices
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in zip(indices, vectors)]
    )


def status_error(status=500, message="Unavailable", headers=None, code=None):
    error_type = BadRequestError if status == 400 else InternalServerError if status >= 500 else APIStatusError
    return error_type(message, response=httpx.Response(status, request=REQUEST, headers=headers), body={"code": code})


@pytest.fixture(autouse=True)
def no_wait(mocker):
    mocker.patch("ai4rag.rag.embedding.request_executor.uniform", return_value=0.0)
    mocker.patch("ai4rag.rag.embedding.request_executor.monotonic", return_value=0.0)
    return mocker.patch("ai4rag.rag.embedding.request_executor.sleep")


@pytest.fixture
def client(mocker):
    instance = mocker.MagicMock()
    instance.timeout = httpx.Timeout(600.0, connect=5.0, write=20.0, pool=10.0)
    instance.with_options.return_value = instance
    instance.embeddings.create.return_value = response([VECTOR])
    return instance


@pytest.fixture
def model(client):
    return OpenAIEmbeddingModel(client, "test-model", {"embedding_dimension": 3, "context_length": 1024})


def sent_inputs(client):
    return [call.kwargs["input"] for call in client.embeddings.create.call_args_list]


def test_explicit_capabilities_do_not_call_service(client, model):
    assert model.params == OpenAIEmbeddingParams(embedding_dimension=3, context_length=1024)
    assert str(model) == "test-model"
    assert repr(model) == "test-model"
    client.embeddings.create.assert_not_called()


@pytest.mark.parametrize("params", [None, OpenAIEmbeddingParams(), {}])
def test_detects_missing_capabilities(client, params):
    detected = OpenAIEmbeddingModel(client, "test-model", params)
    assert detected.params.embedding_dimension == 3
    assert detected.params.context_length == 8129
    assert all(
        call.kwargs["extra_body"] == {"truncate_prompt_tokens": None}
        for call in client.embeddings.create.call_args_list
    )


def test_invalid_parameter_type(client):
    with pytest.raises(TypeError, match="Incorrect type"):
        OpenAIEmbeddingModel(client, "test-model", "invalid")


@pytest.mark.parametrize("length", [MIN_CONTEXT_LENGTH - 1, 0])
def test_rejects_too_small_context(client, length):
    with pytest.raises(ValueError, match="Minimal required value"):
        OpenAIEmbeddingModel(client, "test-model", {"embedding_dimension": 3, "context_length": length})


def test_accepts_minimum_context(client):
    assert (
        OpenAIEmbeddingModel(client, "test-model", OpenAIEmbeddingParams(3, MIN_CONTEXT_LENGTH)).params.context_length
        == MIN_CONTEXT_LENGTH
    )


@pytest.mark.parametrize("status", [400, 401, 500])
def test_dimension_probe_preserves_error_before_context_is_known(client, status, no_wait):
    error = status_error(status, "Unknown model")
    client.embeddings.create.side_effect = error
    with pytest.raises(RuntimeError, match="Failed to auto-detect embedding dimension") as caught:
        OpenAIEmbeddingModel(client, "test-model")
    assert caught.value.__cause__ is error
    assert client.embeddings.create.call_count == (3 if status == 500 else 1)
    assert all(item == ["test"] for item in sent_inputs(client))


def test_context_probe_only_narrows_after_length_rejection(client):
    def respond(**kwargs):
        if len(kwargs["input"][0]) > 2048 * 5:
            raise status_error(400, "Input too long")
        return response([VECTOR])

    client.embeddings.create.side_effect = respond
    detected = OpenAIEmbeddingModel(client, "test-model", {"embedding_dimension": 3})
    assert 1792 <= detected.params.context_length <= 2048


def test_transient_probe_failure_does_not_change_detected_limit(client, no_wait):
    client.embeddings.create.side_effect = [status_error()] + [response([VECTOR])] * 20
    detected = OpenAIEmbeddingModel(client, "test-model", {"embedding_dimension": 3})
    assert detected.params.context_length == 8129
    assert sent_inputs(client)[0] == sent_inputs(client)[1]
    no_wait.assert_called_once_with(0.5)


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_context_probe_outage_propagates_instead_of_inventing_limit(client, status):
    error = status_error(status, "Unavailable")
    client.embeddings.create.side_effect = error
    with pytest.raises(RuntimeError, match="Failed to auto-detect 'context_length'") as caught:
        OpenAIEmbeddingModel(client, "test-model", {"embedding_dimension": 3})
    assert caught.value.__cause__ is error
    assert len({len(item[0]) for item in sent_inputs(client)}) == 1


def test_no_accepted_context_probe_has_clear_error(client):
    client.embeddings.create.side_effect = status_error(400, "Input too long")
    with pytest.raises(RuntimeError, match="context_length") as caught:
        OpenAIEmbeddingModel(client, "test-model", {"embedding_dimension": 3})
    assert isinstance(caught.value.__cause__, ValueError)


def test_context_probe_honors_backend_stop_instruction(client):
    error = status_error(400, "Input too long", headers={"x-should-retry": "false"})
    client.embeddings.create.side_effect = error
    with pytest.raises(RuntimeError, match="context_length") as caught:
        OpenAIEmbeddingModel(client, "test-model", {"embedding_dimension": 3})
    assert caught.value.__cause__ is error
    client.embeddings.create.assert_called_once()


def test_empty_documents_make_no_requests(model, client, no_wait):
    assert model.embed_documents([]) == []
    client.embeddings.create.assert_not_called()
    no_wait.assert_not_called()


def test_batches_128_and_keeps_completed_results(model, client, no_wait):
    texts = [str(i) for i in range(257)]
    vectors = [[float(i)] * 3 for i in range(257)]
    client.embeddings.create.side_effect = [
        response(vectors[:128]),
        status_error(),
        status_error(),
        response(vectors[128:256]),
        status_error(),
        response(vectors[256:]),
    ]
    assert model.embed_documents(texts) == vectors
    assert sent_inputs(client) == [
        texts[:128],
        texts[128:256],
        texts[128:256],
        texts[128:256],
        texts[256:],
        texts[256:],
    ]
    assert [call.args[0] for call in no_wait.call_args_list] == [30.0, 60.0, 30.0]


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_status_has_two_retries(model, client, no_wait, status):
    errors = [status_error(status) for _ in range(3)]
    client.embeddings.create.side_effect = errors
    with pytest.raises(APIStatusError) as caught:
        model.embed_documents(["text"])
    assert caught.value is errors[-1]
    assert client.embeddings.create.call_count == 3
    assert [call.args[0] for call in no_wait.call_args_list] == [30.0, 60.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501, 505])
def test_permanent_errors_fail_immediately(model, client, no_wait, status):
    error = status_error(status, "Unsupported parameter")
    client.embeddings.create.side_effect = error
    with pytest.raises(APIStatusError) as caught:
        model.embed_documents(["text"])
    assert caught.value is error
    client.embeddings.create.assert_called_once()
    no_wait.assert_not_called()


@pytest.mark.parametrize("policy", [INDEXING_POLICY, QUERY_POLICY])
def test_backend_stop_instruction_is_honored(model, client, no_wait, policy):
    error = status_error(503, headers={"x-should-retry": "false"})
    client.embeddings.create.side_effect = error
    with pytest.raises(InternalServerError) as caught:
        model.embed_documents(["text"], policy=policy)
    assert caught.value is error
    client.embeddings.create.assert_called_once()
    no_wait.assert_not_called()


@pytest.mark.parametrize("operation", ["query", "evaluation"])
def test_short_policy_for_query_and_evaluation(model, client, no_wait, operation):
    client.embeddings.create.side_effect = [status_error(), response([VECTOR])]
    result = model.embed_query("text") if operation == "query" else model.embed_documents(["text"], policy=QUERY_POLICY)
    assert result == (VECTOR if operation == "query" else [VECTOR])
    no_wait.assert_called_once_with(0.5)


def test_connection_failure_recovers(model, client, no_wait):
    client.embeddings.create.side_effect = [APIConnectionError(request=REQUEST), response([VECTOR])]
    assert model.embed_documents(["text"]) == [VECTOR]
    no_wait.assert_called_once_with(30.0)


@pytest.mark.parametrize(
    "message,code",
    [("Input too long", None), ("maximum context length exceeded", None), ("Rejected", "context_length_exceeded")],
)
def test_confirmed_overflow_truncates(model, client, message, code, no_wait):
    original = "x" * 6000
    client.embeddings.create.side_effect = [status_error(400, message, code=code), response([VECTOR])]
    assert model.embed_query(original) == VECTOR
    assert [len(item[0]) for item in sent_inputs(client)] == [6000, int(1024 * 0.95 * 4)]
    no_wait.assert_not_called()


def test_mixed_overflow_and_server_errors_keep_repaired_payload(model, client, no_wait):
    original = "x" * 6000
    client.embeddings.create.side_effect = [
        status_error(400, "Input too long"),
        status_error(),
        status_error(400, "Input too long"),
        response([VECTOR]),
    ]
    assert model.embed_documents([original]) == [VECTOR]
    assert [len(item[0]) for item in sent_inputs(client)] == [6000, 3891, 3891, 3686]
    no_wait.assert_called_once_with(30.0)


def test_truncation_exhaustion_raises_latest_error(model, client):
    errors = [status_error(400, "Input too long") for _ in range(3)]
    client.embeddings.create.side_effect = errors
    with pytest.raises(BadRequestError) as caught:
        model.embed_query("x" * 6000)
    assert caught.value is errors[-1]
    assert client.embeddings.create.call_count == 3


def test_overflow_does_not_retry_unchanged_input(model, client):
    client.embeddings.create.side_effect = status_error(400, "Input too long")
    with pytest.raises(BadRequestError):
        model.embed_query("short")
    client.embeddings.create.assert_called_once()


def test_timeout_extends_only_read_once_and_retains_other_phase_limits(model, client):
    error = APITimeoutError(request=REQUEST)
    client.embeddings.create.side_effect = [error, status_error(), error]
    with pytest.raises(APITimeoutError) as caught:
        model.embed_documents(["text"])
    assert caught.value is error
    timeouts = [call.kwargs["timeout"] for call in client.with_options.call_args_list]
    assert [timeout.read for timeout in timeouts] == [600.0, 1200.0, 1200.0]
    assert all(timeout.connect == 5.0 and timeout.write == 20.0 and timeout.pool == 10.0 for timeout in timeouts)
    assert all(call.kwargs["max_retries"] == 0 for call in client.with_options.call_args_list)
    assert client.timeout.read == 600.0


@pytest.mark.parametrize("timeout", [None, 1500.0])
def test_timeout_fallback_does_not_shorten_existing_read_limit(model, client, timeout):
    client.timeout = httpx.Timeout(timeout)
    client.embeddings.create.side_effect = APITimeoutError(request=REQUEST)
    with pytest.raises(APITimeoutError):
        model.embed_query("text")
    client.embeddings.create.assert_called_once()


@pytest.mark.parametrize("cause", [httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout])
def test_non_read_timeouts_use_normal_retry_without_extending_read(model, client, cause, no_wait):
    error = APITimeoutError(request=REQUEST)
    error.__cause__ = cause("Timed out", request=REQUEST)
    client.embeddings.create.side_effect = [error, response([VECTOR])]
    assert model.embed_documents(["text"]) == [VECTOR]
    assert all(call.kwargs["timeout"].read == 600.0 for call in client.with_options.call_args_list)
    no_wait.assert_called_once_with(30.0)


def test_successful_read_timeout_fallback(model, client, no_wait):
    client.embeddings.create.side_effect = [APITimeoutError(request=REQUEST), response([VECTOR])]
    assert model.embed_query("text") == VECTOR
    no_wait.assert_not_called()


def test_total_attempt_budget_covers_mixed_failures(model, client, no_wait):
    errors = [
        status_error(400, "Input too long"),
        APITimeoutError(request=REQUEST),
        status_error(),
        status_error(400, "Input too long"),
        status_error(),
        status_error(),
    ]
    client.embeddings.create.side_effect = errors
    with pytest.raises(InternalServerError) as caught:
        model.embed_documents(["x" * 6000])
    assert caught.value is errors[-1]
    assert client.embeddings.create.call_count == 6
    assert [call.args[0] for call in no_wait.call_args_list] == [30.0, 60.0]


def test_elapsed_budget_does_not_wait_past_deadline(model, client, no_wait):
    error = status_error(503, headers={"Retry-After": "2500"})
    client.embeddings.create.side_effect = error
    with pytest.raises(InternalServerError) as caught:
        model.embed_documents(["text"])
    assert caught.value is error
    client.embeddings.create.assert_called_once()
    no_wait.assert_not_called()


def test_request_timeout_is_capped_to_remaining_budget(model, client):
    assert model._embed_text("text", policy=RequestPolicy(max_elapsed=3.0)) == [VECTOR]
    assert client.with_options.call_args.kwargs["timeout"].as_dict() == dict(connect=3.0, read=3.0, write=3.0, pool=3.0)


def test_elapsed_budget_preserves_last_error(model, client, mocker, no_wait):
    error = status_error()
    client.embeddings.create.side_effect = error
    mocker.patch("ai4rag.rag.embedding.request_executor.monotonic", side_effect=[0.0, 0.0, 0.0, 2500.0])
    with pytest.raises(InternalServerError) as caught:
        model.embed_documents(["text"])
    assert caught.value is error
    client.embeddings.create.assert_called_once()
    no_wait.assert_called_once_with(30.0)


def test_response_indices_restore_input_order(model, client):
    client.embeddings.create.return_value = response([[2.0] * 3, [1.0] * 3], [1, 0])
    assert model.embed_documents(["first", "second"]) == [[1.0] * 3, [2.0] * 3]


@pytest.mark.parametrize(
    "data,texts,match",
    [
        (response([VECTOR]), ["a", "b"], "count"),
        (response([VECTOR, VECTOR], [0, 0]), ["a", "b"], "indices"),
        (response([VECTOR], [1]), ["a"], "indices"),
        (response([VECTOR], [False]), ["a"], "indices"),
        (response([VECTOR], ["0"]), ["a"], "indices"),
        (SimpleNamespace(data=[SimpleNamespace(embedding=VECTOR)]), ["a"], "indices"),
        (SimpleNamespace(data=[SimpleNamespace(index=0)]), ["a"], "numeric"),
        (response(["encoded"]), ["a"], "numeric"),
        (response([[]]), ["a"], "numeric"),
        (response([[True, 0.0, 0.0]]), ["a"], "numeric"),
        (response([[float("nan"), 0.0, 0.0]]), ["a"], "numeric"),
        (response([[float("inf"), 0.0, 0.0]]), ["a"], "numeric"),
        (response([["bad", 0.0, 0.0]]), ["a"], "numeric"),
        (response([[0.0]]), ["a"], "dimension"),
        (SimpleNamespace(data=None), ["a"], "count"),
    ],
)
def test_malformed_responses_fail_without_silent_loss(model, client, no_wait, data, texts, match):
    client.embeddings.create.return_value = data
    with pytest.raises(ValueError, match=match):
        model.embed_documents(texts)
    client.embeddings.create.assert_called_once()
    no_wait.assert_not_called()


@pytest.mark.parametrize("failures", [1, 2, 3])
def test_real_sdk_makes_exact_attempts_without_nested_retries(failures, no_wait):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) <= failures:
            return httpx.Response(500, json={"error": {"message": "Unavailable"}})
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-model",
                "data": [{"object": "embedding", "index": 0, "embedding": VECTOR}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    with OpenAI(
        api_key="test",
        base_url="https://test/v1",
        max_retries=8,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as sdk:
        embedding = OpenAIEmbeddingModel(sdk, "test-model", {"embedding_dimension": 3, "context_length": 1024})
        if failures == 3:
            with pytest.raises(InternalServerError):
                embedding.embed_documents(["text"])
        else:
            assert embedding.embed_documents(["text"]) == [VECTOR]
        assert sdk.max_retries == 8
    assert len(requests) == min(failures + 1, 3)
    assert all(request["encoding_format"] == "float" for request in requests)
    assert [call.args[0] for call in no_wait.call_args_list] == [30.0, 60.0][:failures]
