# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def agent_module(monkeypatch, mocker):
    """Import the agent module from the starter kit source tree."""
    template_src = Path(__file__).resolve().parents[4] / "ai4rag/assets_generator/starter_kit_templates/agentic_rag/src"
    monkeypatch.syspath_prepend(str(template_src))
    module = importlib.import_module("agentic_rag.agent")
    config = SimpleNamespace(
        model_id="test-model",
        temperature=0.2,
        max_completion_tokens=128,
        system_message="Answer using context.",
        user_message_template="{question}: {reference_documents}",
        context_template="{document}",
        language_code="en",
        language_name="English",
        embedding_model_id="embedding-model",
        embedding_dimension=768,
        collection_name="test-collection",
        provider_type="milvus",
        retrieval_method="simple",
        number_of_chunks=5,
        search_mode="vector",
        ranker_strategy="",
        ranker_alpha=None,
    )
    mocker.patch.object(module.AgentConfig, "from_env", return_value=config)
    return module


def test_create_rag_reuses_one_maas_client(agent_module, mocker):
    """The foundation model and retriever share the same MaaS client."""
    client = mocker.sentinel.client
    rag = mocker.sentinel.rag
    retriever = mocker.sentinel.retriever
    openai = mocker.patch.object(agent_module, "OpenAI", return_value=client)
    foundation_model = mocker.patch.object(agent_module, "OpenAIFoundationModel")
    initialize_retriever = mocker.patch.object(agent_module, "initialize_retriever", return_value=retriever)
    rag_class = mocker.patch.object(agent_module, "AgenticRAG", return_value=rag)

    result = agent_module.create_rag(base_url="https://maas.example/v1/", api_key="secret")

    assert result is rag
    openai.assert_called_once_with(api_key="secret", base_url="https://maas.example/v1")
    assert foundation_model.call_args.kwargs["client"] is client
    assert initialize_retriever.call_args.kwargs["client"] is client
    assert initialize_retriever.call_args.kwargs["config"].model_id == "test-model"
    assert rag_class.call_args.kwargs["retriever"] is retriever


def test_create_rag_uses_maas_base_url_even_with_legacy_chat_url(agent_module, monkeypatch, mocker):
    """The legacy chat URL cannot select a different MaaS endpoint."""
    monkeypatch.setenv("MAAS_BASE_URL", "https://shared-maas.example")
    monkeypatch.setenv("CHAT_BASE_URL", "https://chat-only.example")
    monkeypatch.setenv("MAAS_API_KEY", "secret")
    openai = mocker.patch.object(agent_module, "OpenAI", return_value=mocker.sentinel.client)
    mocker.patch.object(agent_module, "OpenAIFoundationModel")
    mocker.patch.object(agent_module, "initialize_retriever")
    mocker.patch.object(agent_module, "AgenticRAG")

    agent_module.create_rag()

    openai.assert_called_once_with(api_key="secret", base_url="https://shared-maas.example/v1")


def test_create_rag_rejects_remote_http(agent_module, mocker):
    """Credentials must not be sent to a remote plaintext MaaS endpoint."""
    openai = mocker.patch.object(agent_module, "OpenAI")

    with pytest.raises(ValueError, match="HTTPS"):
        agent_module.create_rag(base_url="http://maas.example", api_key="secret")

    openai.assert_not_called()


def test_create_rag_rejects_url_without_host(agent_module, mocker):
    """A URL scheme without a host cannot identify a MaaS endpoint."""
    openai = mocker.patch.object(agent_module, "OpenAI")

    with pytest.raises(ValueError, match="host"):
        agent_module.create_rag(base_url="https://", api_key="secret")

    openai.assert_not_called()


def test_create_rag_allows_local_http(agent_module, mocker):
    """Local development can use HTTP without an API key."""
    openai = mocker.patch.object(agent_module, "OpenAI", return_value=mocker.sentinel.client)
    mocker.patch.object(agent_module, "OpenAIFoundationModel")
    mocker.patch.object(agent_module, "initialize_retriever")
    mocker.patch.object(agent_module, "AgenticRAG")

    agent_module.create_rag(base_url="http://localhost:8000")

    openai.assert_called_once_with(api_key="not-needed-for-local-development", base_url="http://localhost:8000/v1")
