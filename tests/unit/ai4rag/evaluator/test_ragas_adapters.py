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
from unittest.mock import MagicMock

import pytest

from ai4rag.evaluator.ragas_adapters import AI4RAGRagasEmbeddings, AI4RAGRagasLLM


def _make_chat_model(content: str = "hello") -> MagicMock:
    model = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    model.chat.return_value = [choice]
    return model


class TestAI4RAGRagasLLM:
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
