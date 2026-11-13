# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Recovery state isolation independently of the embedding model and SDK client."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from unittest.mock import Mock

import httpx
import pytest
from openai import APITimeoutError, BadRequestError, InternalServerError

from ai4rag.rag.embedding.request_executor import INDEXING_POLICY, EmbeddingRequestExecutor, _retry_delay

REQUEST = httpx.Request("POST", "https://test/v1/embeddings")


def status_error(status=500, headers=None):
    return InternalServerError(
        "Unavailable", response=httpx.Response(status, request=REQUEST, headers=headers), body=None
    )


def overflow_error():
    return BadRequestError("Input too long", response=httpx.Response(400, request=REQUEST), body=None)


@pytest.fixture(autouse=True)
def no_wait(mocker):
    mocker.patch("ai4rag.rag.embedding.request_executor.uniform", return_value=0.0)
    mocker.patch("ai4rag.rag.embedding.request_executor.monotonic", return_value=0.0)
    return mocker.patch("ai4rag.rag.embedding.request_executor.sleep")


def executor(request, timeout=None):
    return EmbeddingRequestExecutor(
        request=request,
        policy=INDEXING_POLICY,
        timeout=httpx.Timeout(600.0) if timeout is None else timeout,
        context_length=1024,
        model_id="test",
    )


def test_repeated_execution_resets_input_timeout_and_retry_budget(no_wait):
    texts = ["x" * 6000]
    timeout = httpx.Timeout(600.0, connect=5.0, write=20.0, pool=10.0)
    request = Mock(
        side_effect=[
            overflow_error(),
            status_error(),
            status_error(),
            APITimeoutError(request=REQUEST),
            [[0.1]],
            status_error(),
            [[0.2]],
        ]
    )
    runner = executor(request, timeout)

    assert runner.execute(texts) == [[0.1]]
    assert runner.execute(texts) == [[0.2]]

    assert [len(call.args[0][0]) for call in request.call_args_list] == [6000, 3891, 3891, 3891, 3891, 6000, 6000]
    assert [call.args[1].read for call in request.call_args_list] == [600.0] * 4 + [1200.0, 600.0, 600.0]
    assert [call.args[0] for call in no_wait.call_args_list] == [30.0, 60.0, 30.0]
    assert texts == ["x" * 6000]
    assert timeout.as_dict() == dict(connect=5.0, read=600.0, write=20.0, pool=10.0)


def test_failed_execution_does_not_poison_later_execution(no_wait):
    error = status_error()
    request = Mock(side_effect=[error, error, error, error, [[0.1]]])
    runner = executor(request)

    with pytest.raises(InternalServerError) as caught:
        runner.execute(["first"])
    assert caught.value is error
    assert runner.execute(["second"]) == [[0.1]]

    assert [call.args[0] for call in no_wait.call_args_list] == [30.0, 60.0, 30.0]
    assert [call.args[0] for call in request.call_args_list] == [["first"]] * 3 + [["second"]] * 2


def test_concurrent_executions_keep_recovery_state_independent(no_wait):
    barrier = Barrier(2, timeout=5.0)
    lock = Lock()
    counts = {"a": 0, "b": 0}
    attempts = []

    def request(texts, timeout):
        key = texts[0][0]
        with lock:
            counts[key] += 1
            first_attempt = counts[key] == 1
            attempts.append((key, len(texts[0]), timeout.read))
        if first_attempt:
            barrier.wait()
            raise overflow_error() if key == "a" else APITimeoutError(request=REQUEST)
        return [[1.0 if key == "a" else 2.0]]

    runner = executor(request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runner.execute, ["a" * 6000])
        second = pool.submit(runner.execute, ["b" * 6000])
        assert first.result(timeout=10) == [[1.0]]
        assert second.result(timeout=10) == [[2.0]]

    assert sorted(attempts) == [("a", 3891, 600.0), ("a", 6000, 600.0), ("b", 6000, 600.0), ("b", 6000, 1200.0)]
    no_wait.assert_not_called()


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"Retry-After": "90"}, 90.0),
        ({"retry-after-ms": "90000"}, 90.0),
        ({"Retry-After": "-1"}, 30.0),
        ({"Retry-After": "nan"}, 30.0),
        ({"Retry-After": "inf"}, 30.0),
        ({"Retry-After": "invalid"}, 30.0),
        ({"Retry-After": "Thu, 01 Jan 99999999999999999999999 00:00:00 GMT"}, 30.0),
        ({"retry-after-ms": "bad", "Retry-After": "90"}, 90.0),
        ({"Retry-After": "Thu, 01 Jan 1970 00:01:30 GMT"}, 90.0),
    ],
)
def test_server_retry_delays(headers, expected, mocker):
    mocker.patch("ai4rag.rag.embedding.request_executor.time", return_value=0.0)
    assert _retry_delay(status_error(503, headers=headers), 30.0) == expected


def test_jitter_only_adds_to_server_delay(mocker):
    mocker.patch("ai4rag.rag.embedding.request_executor.uniform", side_effect=lambda low, high: high)
    assert _retry_delay(status_error(503, headers={"Retry-After": "90"}), 30.0) == 99.0
