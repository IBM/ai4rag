# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for the ai4rag <-> RAGAS adapters.

RAGAS is a regular dependency, so the adapters subclass the real
``BaseRagasLLM`` / ``BaseRagasEmbeddings`` base classes.  The tests exercise the
adapters' *delegation* logic (prompt rendering, ``max_completion_tokens``
passthrough, the ``n`` loop, ``stop`` handling and the async wrappers) against a
mocked ai4rag model.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import BadRequestError, InternalServerError
from ragas.run_config import RunConfig

from ai4rag.evaluator.ragas_adapters import AI4RAGRagasEmbeddings, AI4RAGRagasLLM
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel


def _make_chat_model(content: str = "hello") -> MagicMock:
    model = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    model.chat.return_value = [choice]
    return model


class TestAI4RAGRagasLLM:
    @pytest.mark.parametrize("overflow", [True, False])
    def test_context_overflow_bypasses_identical_request_retries(self, overflow):
        model = _make_chat_model()
        error = BadRequestError(
            "maximum context length exceeded" if overflow else "unsupported parameter",
            response=httpx.Response(400, request=httpx.Request("POST", "https://example.test")),
            body=None,
        )
        model.chat.side_effect = error
        llm = AI4RAGRagasLLM(model)
        llm.set_run_config(RunConfig(max_retries=2, max_wait=0))

        with pytest.raises(BadRequestError):
            asyncio.run(llm.generate(MagicMock()))

        assert model.chat.call_count == (1 if overflow else 2)

    def test_transient_failure_keeps_existing_retries(self):
        model = _make_chat_model("recovered")
        model.chat.side_effect = [ConnectionError("temporary"), model.chat.return_value]
        llm = AI4RAGRagasLLM(model)
        llm.set_run_config(RunConfig(max_retries=2, max_wait=0))

        result = asyncio.run(llm.generate(MagicMock(), temperature=None))

        assert result.generations[0][0].text == "recovered"
        assert model.chat.call_count == 2

    def test_generate_text_delegates_with_token_cap(self):
        model = _make_chat_model("hi there")
        llm = AI4RAGRagasLLM(model, max_completion_tokens=512)
        prompt = MagicMock()
        prompt.to_string.return_value = "the prompt"

        result = llm.generate_text(prompt, n=1, temperature=0.3)

        model.chat.assert_called_once_with(
            [{"role": "user", "content": "the prompt"}], temperature=0.3, max_completion_tokens=512
        )
        assert result.generations[0][0].text == "hi there"

    def test_n_loop_and_stop_passthrough(self):
        model = _make_chat_model()
        llm = AI4RAGRagasLLM(model)
        prompt = MagicMock()
        prompt.to_string.return_value = "p"

        result = llm.generate_text(prompt, n=3, stop=["\n"])

        assert model.chat.call_count == 3
        assert model.chat.call_args.kwargs["stop"] == ["\n"]
        assert len(result.generations[0]) == 3

    def test_none_content_becomes_empty_string(self):
        model = _make_chat_model(content=None)
        llm = AI4RAGRagasLLM(model)
        prompt = MagicMock()
        prompt.to_string.return_value = "p"

        result = llm.generate_text(prompt)
        assert result.generations[0][0].text == ""

    def test_agenerate_text_delegates(self):
        model = _make_chat_model("async")
        llm = AI4RAGRagasLLM(model)
        prompt = MagicMock()
        prompt.to_string.return_value = "p"

        result = asyncio.run(llm.agenerate_text(prompt, temperature=None))

        assert result.generations[0][0].text == "async"
        # temperature=None must fall back to the default rather than being forwarded as None.
        assert model.chat.call_args.kwargs["temperature"] == pytest.approx(1e-2)

    def test_is_finished_true(self):
        llm = AI4RAGRagasLLM(_make_chat_model())
        assert llm.is_finished(None) is True


class TestAI4RAGRagasEmbeddings:
    def test_sync_delegation(self):
        model = MagicMock()
        model.embed_query.return_value = [0.1, 0.2]
        model.embed_documents.return_value = [[0.1], [0.2]]
        emb = AI4RAGRagasEmbeddings(model)

        assert emb.embed_query("x") == [0.1, 0.2]
        assert emb.embed_documents(["a", "b"]) == [[0.1], [0.2]]
        model.embed_query.assert_called_once_with("x")
        model.embed_documents.assert_called_once_with(["a", "b"])

    def test_async_delegation(self):
        model = MagicMock()
        model.embed_query.return_value = [0.3]
        model.embed_documents.return_value = [[0.3]]
        emb = AI4RAGRagasEmbeddings(model)

        assert asyncio.run(emb.aembed_query("x")) == [0.3]
        assert asyncio.run(emb.aembed_documents(["a"])) == [[0.3]]

    @pytest.mark.parametrize("is_async", [True, False])
    def test_other_providers_keep_ragas_retries(self, is_async):
        model = MagicMock()
        model.embed_documents.side_effect = [ConnectionError("temporary"), [[0.3]]]
        emb = AI4RAGRagasEmbeddings(model)
        emb.set_run_config(RunConfig(max_retries=2, max_wait=0))

        assert asyncio.run(emb.embed_texts(["a"], is_async=is_async)) == [[0.3]]
        assert model.embed_documents.call_count == 2


@pytest.fixture
def openai_embeddings(mocker):
    client = MagicMock()
    client.timeout = httpx.Timeout(600)
    client.with_options.return_value = client
    client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.3])])
    model = OpenAIEmbeddingModel(client, "test", {"embedding_dimension": 1, "context_length": 1024})
    sleep = mocker.patch("ai4rag.rag.embedding.openai_model.sleep")
    mocker.patch("ai4rag.rag.embedding.openai_model.uniform", return_value=0.0)
    emb = AI4RAGRagasEmbeddings(model)
    emb.set_run_config(RunConfig(max_retries=10, max_wait=0))
    return emb, client, sleep


@pytest.mark.parametrize("entrypoint", ["sync", "async", "texts-sync", "texts-async"])
def test_openai_evaluation_selects_short_recovery(openai_embeddings, entrypoint):
    emb, client, sleep = openai_embeddings
    error = InternalServerError(
        "Unavailable",
        response=httpx.Response(500, request=httpx.Request("POST", "https://example.test")),
        body=None,
    )
    client.embeddings.create.side_effect = [error, client.embeddings.create.return_value]

    if entrypoint == "sync":
        result = emb.embed_documents(["a"])
    elif entrypoint == "async":
        result = asyncio.run(emb.aembed_documents(["a"]))
    else:
        result = asyncio.run(emb.embed_texts(["a"], is_async=entrypoint == "texts-async"))

    assert result == [[0.3]]
    assert client.embeddings.create.call_count == 2
    sleep.assert_called_once_with(0.5)


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("status,headers,attempts", [(400, {}, 1), (500, {}, 3), (503, {"x-should-retry": "false"}, 1)])
def test_openai_evaluation_has_one_retry_owner(openai_embeddings, is_async, status, headers, attempts):
    emb, client, sleep = openai_embeddings
    error_type = BadRequestError if status == 400 else InternalServerError
    error = error_type(
        "Unsupported parameter" if status == 400 else "Unavailable",
        response=httpx.Response(status, request=httpx.Request("POST", "https://example.test"), headers=headers),
        body=None,
    )
    client.embeddings.create.side_effect = error

    with pytest.raises(error_type) as caught:
        asyncio.run(emb.embed_texts(["a"], is_async=is_async))

    assert caught.value is error
    assert client.embeddings.create.call_count == attempts
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0][: attempts - 1]
