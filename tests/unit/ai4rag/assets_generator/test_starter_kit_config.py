# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import base64
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def config_module(monkeypatch):
    """Import the starter kit configuration from its source tree."""
    template_src = Path(__file__).resolve().parents[4] / "ai4rag/assets_generator/starter_kit_templates/agentic_rag/src"
    monkeypatch.syspath_prepend(str(template_src))
    return importlib.import_module("agentic_rag.config")


@pytest.fixture
def generated_config():
    """A complete generated agent configuration."""
    return {
        "template_id": "agentic_rag",
        "generation": {
            "model_id": "chat-model",
            "temperature": 0.2,
            "max_completion_tokens": 128,
            "language_code": "en",
            "language_name": "English",
        },
        "prompts": {
            "system_message": "Use context.",
            "user_message_template": "{question}: {reference_documents}",
            "context_template": "{document}",
        },
        "embedding": {"model_id": "embedding-model", "dimension": 768},
        "retrieval": {
            "method": "simple",
            "number_of_chunks": 5,
            "search_mode": "vector",
            "ranker_strategy": "",
            "ranker_k": None,
            "ranker_alpha": None,
        },
        "vector_store": {"provider_type": "milvus", "collection_name": "documents"},
    }


def _inject_config(monkeypatch, config):
    encoded = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
    monkeypatch.setenv("AGENT_CONFIG_B64", encoded)


def test_generated_settings(config_module, generated_config, monkeypatch):
    """Load optimized RAG settings without deployment-specific runtime settings."""
    _inject_config(monkeypatch, generated_config)

    config = config_module.AgentConfig.from_env()

    assert config.model_id == "chat-model"
    assert config.embedding_model_id == "embedding-model"


def test_rrf_ranker_k_is_loaded(config_module, generated_config, monkeypatch):
    generated_config["retrieval"].update(ranker_strategy="rrf", ranker_k=42)
    _inject_config(monkeypatch, generated_config)

    assert config_module.AgentConfig.from_env().ranker_k == 42


def test_invalid_base64_is_rejected(config_module, monkeypatch):
    """A malformed injected configuration fails before model initialization."""
    monkeypatch.setenv("AGENT_CONFIG_B64", "%%%")

    with pytest.raises(ValueError, match="valid base64-encoded JSON"):
        config_module.AgentConfig.from_env()


def test_missing_required_config_is_rejected(config_module, generated_config, monkeypatch):
    """Missing optimized model settings fail explicitly."""
    del generated_config["embedding"]["model_id"]
    _inject_config(monkeypatch, generated_config)

    with pytest.raises(ValueError, match="model_id"):
        config_module.AgentConfig.from_env()


@pytest.mark.parametrize(
    "graph_settings", [{}, {"route_k": 4, "entity_relationship_hops": 2, "include_entity_neighbors": False}]
)
def test_graph_settings_reach_store_search(config_module, generated_config, monkeypatch, mocker, graph_settings):
    """Generated graph options survive config loading and actual retriever calls."""
    generated_config["template_id"] = "agentic_graph_rag"
    generated_config["vector_store"]["provider_type"] = "neo4j"
    generated_config["retrieval"].update(search_mode="graph", **graph_settings)
    _inject_config(monkeypatch, generated_config)
    config = config_module.AgentConfig.from_env()

    tools_module = importlib.import_module("agentic_rag.tools")
    mocker.patch.object(tools_module, "OpenAIEmbeddingModel")
    mocker.patch.object(tools_module, "get_vector_store_config")
    store = mocker.patch.object(tools_module, "get_vector_store").return_value
    retriever = tools_module.initialize_retriever(mocker.sentinel.client, config)
    retriever.retrieve("What is connected?")

    options = store.search.call_args.kwargs
    assert options["search_mode"] == "graph"
    assert options["entity_neighbor_limit"] == 5
    assert options["entity_relationship_hops"] == graph_settings.get("entity_relationship_hops", 1)
    assert options["include_entity_neighbors"] == graph_settings.get("include_entity_neighbors", True)
    if "route_k" in graph_settings:
        assert options["route_k"] == 4
    else:
        assert "route_k" not in options  # Leave the store's k-dependent route limit in effect.
