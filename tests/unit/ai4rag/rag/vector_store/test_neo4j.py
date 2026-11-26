# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from httpx import Request
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship
from neo4j_graphrag.llm.base import LLMInterfaceV2
from openai import APIConnectionError, OpenAI

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.neo4j import (
    Neo4jGraphStore,
    Neo4jMemoryError,
    _KGExtractionDegradedError,
    _KGExtractionRateLimitedError,
)
from ai4rag.rag.vector_store.neo4j_utils import (
    _KG_LEXICAL_GRAPH_CONFIG,
    DEFAULT_KG_LLM_MAX_RETRIES,
    DEFAULT_KG_LLM_RETRY_BASE_SECONDS,
    DEFAULT_KG_LLM_SDK_MAX_RETRIES,
    DEFAULT_KG_RATE_LIMIT_BREAKER_THRESHOLD,
    KG_BATCH_JSON_SCHEMA_RESPONSE_FORMAT,
    KG_BATCH_RESPONSE_FORMAT,
    KG_LLM_RETRY_MAX_BACKOFF_SECONDS,
    KG_MAX_ENTITIES_PER_CHUNK,
    KG_MAX_RELATIONSHIPS_PER_CHUNK,
    _build_graph_retrieval_query,
    _CanonicalKGWriter,
    _EmbedderAdapter,
    _LLMAdapter,
    _model_with_sdk_retries,
    _validate_kg_extraction_config,
    _validate_neo4j_search_params,
    is_rate_limit_llm_error,
    is_retryable_llm_error,
    kg_retry_delay_seconds,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockEmbeddingModel:
    model_id = "test-embedding"
    params = {"embedding_dimension": 128}

    def embed_documents(self, texts):
        return [[float(i % 10) / 10] * 128 for i in range(len(texts))]

    def embed_query(self, query):
        return [0.1] * 128


@pytest.fixture
def mock_embedding():
    return _MockEmbeddingModel()


@pytest.fixture
def neo4j_config():
    return Neo4jConfig(uri="neo4j://localhost:7687", username="neo4j", password="test")


@pytest.fixture(autouse=True)
def mock_batched_kg_requests():
    """Avoid external model calls while preserving the batch response contract."""

    async def respond(_, prompt, _completion_limit):
        payload = json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1])
        return SimpleNamespace(
            content=json.dumps(
                {"chunks": [{"chunk_id": item["chunk_id"], "nodes": [], "relationships": []} for item in payload]}
            )
        )

    with (
        patch.object(_LLMAdapter, "ainvoke_with_completion_limit", respond),
        patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)),
    ):
        yield


def _make_store(mock_driver_cls, mock_embedding, neo4j_config, collection_name="ai4rag_test"):
    """Construct a Neo4jGraphStore with a fully mocked neo4j driver."""
    store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name=collection_name)
    return store


def test_llm_adapter_accepts_base_foundation_model_message_response():
    """The adapter accepts the message mapping promised by BaseFoundationModel."""
    model = MagicMock(model_id="test-model")
    model.chat.return_value = [{"role": "assistant", "content": '{"nodes": [], "relationships": []}'}]

    adapter = _LLMAdapter(model)
    response = adapter.invoke([{"role": "user", "content": "Extract entities."}])

    assert response.content == '{"nodes": [], "relationships": []}'
    assert isinstance(adapter, LLMInterfaceV2)
    model.chat.assert_called_once_with([{"role": "user", "content": "Extract entities."}])


def test_llm_adapter_can_override_the_completion_limit_for_entity_extraction():
    """The KG-specific token cap must not mutate the model's general parameters."""
    model = MagicMock(model_id="test-model")
    model.chat.return_value = [{"role": "assistant", "content": '{"nodes": [], "relationships": []}'}]

    _LLMAdapter(model, max_completion_tokens=1024).invoke([{"role": "user", "content": "Extract entities."}])

    model.chat.assert_called_once_with([{"role": "user", "content": "Extract entities."}], max_completion_tokens=1024)


def test_llm_adapter_requests_a_json_object_response():
    """KG extraction requests the MaaS-compatible JSON-object response mode."""
    model = MagicMock(model_id="test-model")
    model.chat.return_value = [{"role": "assistant", "content": '{"chunks": []}'}]

    _LLMAdapter(model, response_format=KG_BATCH_RESPONSE_FORMAT).invoke(
        [{"role": "user", "content": "Extract entities."}]
    )

    model.chat.assert_called_once_with(
        [{"role": "user", "content": "Extract entities."}], response_format=KG_BATCH_RESPONSE_FORMAT
    )


def test_llm_adapter_logs_the_kg_response_stop_reason(caplog):
    model = MagicMock(model_id="test-model")
    model.chat.return_value = [
        SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"chunks": []}'))
    ]

    with caplog.at_level(logging.INFO):
        _LLMAdapter(model, response_format=KG_BATCH_RESPONSE_FORMAT).invoke(
            [{"role": "user", "content": "Extract entities."}]
        )

    assert "KG extraction LLM response received (model=test-model, stop_reason=length)." in caplog.messages


def test_llm_adapter_repairs_fenced_singleton_kg_response():
    model = MagicMock(model_id="test-model")
    model.chat.return_value = [
        {
            "role": "assistant",
            "content": '```json\n[{"entities": [], "relations": []}]\n```',
        }
    ]
    adapter = _LLMAdapter(model)

    response = adapter.invoke([{"role": "user", "content": "Extract entities."}])

    assert json.loads(response.content) == {"nodes": [], "relationships": []}
    assert adapter.metrics[2] == 1


def test_llm_adapter_repairs_a_valid_json_object_with_a_short_preamble():
    """Cheap local repair must recover valid JSON without a second MaaS call."""
    model = MagicMock(model_id="test-model")
    model.chat.return_value = [{"role": "assistant", "content": 'Result: {"chunks": []}'}]

    response = _LLMAdapter(model, response_format=KG_BATCH_RESPONSE_FORMAT).invoke(
        [{"role": "user", "content": "Extract entities."}]
    )

    assert json.loads(response.content) == {"chunks": []}
    assert model.chat.call_count == 1


def test_kg_embedder_reuses_canonical_embedding(mock_embedding):
    adapter = _EmbedderAdapter(mock_embedding, {"canonical chunk": [0.9, 0.8]})

    assert adapter.embed_query("canonical chunk") == [0.9, 0.8]
    assert adapter.embed_query("entity name") == [0.1] * 128
    model_calls, cache_hits, elapsed_seconds = adapter.metrics
    assert (model_calls, cache_hits) == (1, 1)
    assert elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# Neo4jConfig
# ---------------------------------------------------------------------------


class TestNeo4jConfig:
    def test_from_env_reads_required_vars(self):
        env = {
            "NEO4J_URI": "neo4j://localhost:7687",
            "NEO4J_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Neo4jConfig.from_env()
        assert cfg.uri == "neo4j://localhost:7687"
        assert cfg.password == "secret"
        assert cfg.username == "neo4j"
        assert cfg.database == "neo4j"

    def test_from_env_reads_optional_vars(self):
        env = {
            "NEO4J_URI": "neo4j://localhost:7687",
            "NEO4J_PASSWORD": "pw",
            "NEO4J_USERNAME": "admin",
            "NEO4J_DATABASE": "mydb",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Neo4jConfig.from_env()
        assert cfg.username == "admin"
        assert cfg.database == "mydb"

    def test_from_env_reads_ca_cert(self):
        cert = "-----BEGIN CERTIFICATE-----\ncertificate\n-----END CERTIFICATE-----"
        env = {
            "NEO4J_URI": "neo4j://host:7687",
            "NEO4J_PASSWORD": "pw",
            "NEO4J_CA_CERT": cert,
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Neo4jConfig.from_env()
        assert cfg.ca_cert == cert

    @pytest.mark.parametrize("uri", ["neo4j+s://host:7687", "bolt+s://host:7687"])
    def test_ca_cert_accepts_tls_uri(self, uri):
        cfg = Neo4jConfig(uri=uri, password="pw", ca_cert="CERT")

        assert cfg.uri == uri

    def test_ca_cert_rejects_unverified_tls_uri(self):
        with pytest.raises(ValueError, match=r"\+ssc"):
            Neo4jConfig(uri="neo4j+ssc://host:7687", password="pw", ca_cert="CERT")

    def test_from_env_missing_uri_raises(self):
        env = {"NEO4J_PASSWORD": "pw"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(KeyError):
                # Remove NEO4J_URI if it somehow leaks in
                os.environ.pop("NEO4J_URI", None)
                Neo4jConfig.from_env()

    def test_from_env_missing_password_raises(self):
        env = {"NEO4J_URI": "neo4j://localhost:7687"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(KeyError):
                os.environ.pop("NEO4J_PASSWORD", None)
                Neo4jConfig.from_env()

    def test_provider_is_neo4j(self):
        cfg = Neo4jConfig(uri="neo4j://localhost:7687", password="pw")
        assert cfg.provider == "neo4j"

    def test_rejects_plain_neo4j_scheme_to_remote_host(self):
        """A plaintext neo4j:// URI against a non-local host must be rejected."""
        with pytest.raises(ValueError, match="Neo4jConfig.uri"):
            Neo4jConfig(uri="neo4j://evil.example.com:7687", password="pw")

    def test_allows_plain_bolt_scheme_to_cluster_local_host(self):
        """A plaintext bolt:// URI against an in-cluster host is allowed."""
        cfg = Neo4jConfig(uri="bolt://neo4j.svc.cluster.local:7687", password="pw")

        assert cfg.uri == "bolt://neo4j.svc.cluster.local:7687"

    def test_allows_encrypted_scheme_to_remote_host(self):
        """neo4j+s:// uses verified TLS and is allowed for remote hosts."""
        cfg = Neo4jConfig(uri="neo4j+s://remote.example.com:7687", password="pw")

        assert cfg.uri == "neo4j+s://remote.example.com:7687"

    def test_repr_redacts_password_and_ca_cert(self):
        cfg = Neo4jConfig(
            uri="neo4j://host:7687",
            password="do-not-log-password",
            ca_cert="do-not-log-certificate",
        )

        config_repr = repr(cfg)

        assert "do-not-log-password" not in config_repr
        assert "do-not-log-certificate" not in config_repr


class TestKGExtractionConfig:
    def test_default_completion_limit_is_1024(self):
        assert _validate_kg_extraction_config(None)["max_completion_tokens"] == 1024

    def test_default_retry_settings_are_rate_limit_tolerant(self):
        config = _validate_kg_extraction_config(None)

        assert config["max_retries"] == DEFAULT_KG_LLM_MAX_RETRIES
        assert config["retry_base_seconds"] == DEFAULT_KG_LLM_RETRY_BASE_SECONDS
        assert config["sdk_max_retries"] == DEFAULT_KG_LLM_SDK_MAX_RETRIES
        assert config["rate_limit_breaker_threshold"] == DEFAULT_KG_RATE_LIMIT_BREAKER_THRESHOLD
        assert config["fail_on_empty_kg"] is True

    def test_accepts_custom_retry_settings(self):
        config = _validate_kg_extraction_config(
            {
                "max_retries": 0,
                "retry_base_seconds": 0.5,
                "sdk_max_retries": 2,
                "rate_limit_breaker_threshold": 3,
            }
        )

        assert config["max_retries"] == 0
        assert config["retry_base_seconds"] == 0.5
        assert config["sdk_max_retries"] == 2
        assert config["rate_limit_breaker_threshold"] == 3

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({"max_retries": -1}, "max_retries must be a non-negative integer"),
            ({"max_retries": True}, "max_retries must be a non-negative integer"),
            ({"retry_base_seconds": 0}, "retry_base_seconds must be a positive number"),
            ({"sdk_max_retries": -1}, "sdk_max_retries must be a non-negative integer"),
            ({"rate_limit_breaker_threshold": 0}, "rate_limit_breaker_threshold must be a positive integer"),
            ({"fail_on_empty_kg": "yes"}, "fail_on_empty_kg must be a boolean"),
        ],
    )
    def test_rejects_invalid_retry_settings(self, config, message):
        with pytest.raises(ValueError, match=message):
            _validate_kg_extraction_config(config)

    def test_rejects_boolean_completion_limit(self):
        with pytest.raises(ValueError, match="max_completion_tokens must be a positive integer"):
            _validate_kg_extraction_config({"max_completion_tokens": True})

    def test_rejects_boolean_free_mode_limits(self):
        with pytest.raises(ValueError, match="max_entities_per_chunk must be a positive integer"):
            _validate_kg_extraction_config(
                {"mode": "free", "max_entities_per_chunk": True, "max_relationships_per_chunk": 5}
            )

    def test_json_schema_is_the_default_response_format(self):
        assert _validate_kg_extraction_config(None)["response_format"] == "json_schema"
        assert _validate_kg_extraction_config({})["response_format"] == "json_schema"

    def test_accepts_the_json_object_response_format(self):
        config = _validate_kg_extraction_config({"response_format": "json_object"})

        assert config["response_format"] == "json_object"

    def test_rejects_an_unknown_response_format(self):
        with pytest.raises(ValueError, match="response_format must be 'json_object' or 'json_schema'"):
            _validate_kg_extraction_config({"response_format": "tools"})


def test_json_schema_response_format_requires_every_field():
    """Structured output must be strict so the endpoint enforces the payload."""
    schema = KG_BATCH_JSON_SCHEMA_RESPONSE_FORMAT["json_schema"]
    assert schema["strict"] is True
    chunk_schema = schema["schema"]["properties"]["chunks"]["items"]
    assert schema["schema"]["required"] == ["chunks"]
    assert chunk_schema["required"] == ["chunk_id", "nodes", "relationships"]
    assert chunk_schema["additionalProperties"] is False
    node_schema = chunk_schema["properties"]["nodes"]["items"]
    assert node_schema["required"] == ["id", "label", "properties"]
    assert node_schema["additionalProperties"] is False
    relationship_schema = chunk_schema["properties"]["relationships"]["items"]
    assert relationship_schema["required"] == ["start_node_id", "end_node_id", "type"]
    assert relationship_schema["additionalProperties"] is False


def test_balanced_graph_query_limits_pivots_hops_and_related_chunks():
    """Balanced graph retrieval must bound relationship traversal per seed."""
    query = _build_graph_retrieval_query(
        include_entity_neighbors=True,
        entity_neighbor_limit=5,
        entity_pivot_limit=3,
        entity_relationship_hops=2,
        relationship_neighbor_limit=5,
    )

    assert "count(DISTINCT entity) AS overlap" in query
    assert "ORDER BY similarity DESC, overlap DESC, ent_nb.id ASC LIMIT 5" in query
    assert "count(DISTINCT pivot_chunk) AS degree" in query
    assert "ORDER BY degree DESC, pivot.name ASC, pivot.id ASC LIMIT 3" in query
    assert "[rels*1..2]-(related:__Entity__)" in query
    assert "count(DISTINCT related) AS entity_count, min(size(rels)) AS hops" in query
    assert "ORDER BY similarity DESC, entity_count DESC, hops ASC, rel_nb.id ASC LIMIT 5" in query
    assert "count(DISTINCT path)" not in query
    assert "MATCH path =" not in query
    assert "$col IN COALESCE(related.ai4rag_kg_collections, [])" in query
    assert "type(rel) <> 'FROM_CHUNK'" in query
    assert "$col IN COALESCE(rel.ai4rag_kg_collections, [])" in query
    assert "UNWIND hits AS hit" in query
    assert "sum(hit.strength) AS graph_strength" in query
    assert "vector.similarity.cosine(candidate.embedding, $query_vector)" in query
    assert "0.8 * semantic_score + 0.2 * graph_strength / (1.0 + graph_strength)" in query
    assert "LIMIT $top_k" in query
    assert "ORDER BY score DESC, graph_strength DESC, candidate.id ASC" in query
    assert "RETURN candidate.text AS text" in query
    assert "elementId(candidate)" not in query
    assert "collect(DISTINCT ent_nb)[.." not in query


def test_graph_query_uses_relationship_expansion_defaults():
    """Relationship expansion defaults to three pivots, one hop, and five chunks."""
    query = _build_graph_retrieval_query(
        include_entity_neighbors=True,
        entity_neighbor_limit=5,
    )

    assert "ORDER BY degree DESC, pivot.name ASC, pivot.id ASC LIMIT 3" in query
    assert "[rels*1..1]-(related:__Entity__)" in query
    assert "rel_nb.id ASC LIMIT 5" in query


def test_graph_query_returns_single_chunks_and_falls_back_to_seed():
    query = _build_graph_retrieval_query(
        include_entity_neighbors=False,
        entity_neighbor_limit=0,
        entity_pivot_limit=0,
    )

    assert "[] AS ent_hits" in query
    assert "[] AS rel_hits" in query
    assert "THEN [{chunk: node, strength: 0.0}] ELSE graph_hits END" in query
    assert "RETURN candidate.text AS text" in query
    assert "node.text +" not in query


def test_graph_route_fusion_deduplicates_evidence_and_preserves_routes():
    chunk = AI4RAGChunk(text="shared evidence")
    local_only = AI4RAGChunk(text="local evidence")

    fused = Neo4jGraphStore._fuse_graph_routes(
        {"vector": [(chunk, 0.9)], "local": [(AI4RAGChunk(text="shared evidence"), 0.8), (local_only, 0.7)]},
        k=2,
    )

    assert len(fused) == 2
    assert fused[0][0].text == "shared evidence"
    assert fused[0][0].metadata["routes"] == ["vector", "local"]


# ---------------------------------------------------------------------------
# Neo4jGraphStore.__init__
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestNeo4jGraphStoreInit:
    def test_defers_collection_vector_index_until_chunks_are_written(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        cypher_calls = [str(c) for c in session.run.call_args_list]
        joined = " ".join(cypher_calls)
        assert "VECTOR INDEX" not in joined

    def test_verifies_connectivity(self, mock_driver_cls, mock_embedding, neo4j_config):
        Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        mock_driver_cls.return_value.verify_connectivity.assert_called_once()

    def test_uses_default_ca_bundle_when_ca_cert_is_omitted(self, mock_driver_cls, mock_embedding, monkeypatch):
        """Leave TLS trust to the driver when its default CA bundle is configured."""
        monkeypatch.setenv("SSL_CERT_FILE", "/custom/default-ca-bundle.pem")
        config = Neo4jConfig(uri="neo4j+s://host:7687", username="neo4j", password="test")

        Neo4jGraphStore(mock_embedding, config, collection_name="ai4rag_col")

        assert config.ca_cert is None
        assert mock_driver_cls.call_args.args[0] == "neo4j+s://host:7687"
        assert mock_driver_cls.call_args.kwargs == {
            "auth": ("neo4j", "test"),
            "liveness_check_timeout": 30.0,
            "max_connection_lifetime": 300.0,
            "max_transaction_retry_time": 30.0,
        }

    @pytest.mark.parametrize(
        ("configured_uri", "expected_driver_uri"),
        [("neo4j+s://host:7687", "neo4j://host:7687"), ("bolt+s://host:7687", "bolt://host:7687")],
    )
    def test_trusts_configured_ca_cert(self, mock_driver_cls, mock_embedding, configured_uri, expected_driver_uri):
        cert = "-----BEGIN CERTIFICATE-----\ncertificate\n-----END CERTIFICATE-----"
        config = Neo4jConfig(
            uri=configured_uri,
            username="neo4j",
            password="test",
            ca_cert=cert,
        )

        Neo4jGraphStore(mock_embedding, config, collection_name="ai4rag_col")

        trusted_certificates = mock_driver_cls.call_args.kwargs["trusted_certificates"]
        assert isinstance(trusted_certificates, neo4j.TrustCustomCAs)
        assert mock_driver_cls.call_args.kwargs["encrypted"] is True
        assert mock_driver_cls.call_args.args[0] == expected_driver_uri
        cert_path = trusted_certificates.certs[0]
        assert Path(cert_path).read_text() == cert

    def test_collection_name_prefix_guard(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="ai4rag"):
            Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="bad_name")

    def test_reuses_supplied_collection_name(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_my_col")
        assert store.collection_name == "ai4rag_my_col"

    def test_generates_collection_name_when_none(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config)
        assert store.collection_name.startswith("ai4rag_")


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestAddDocuments:
    def _make_chunks(self, n=3, doc_id="doc1"):
        return [
            AI4RAGChunk(
                text=f"chunk {i}",
                metadata={"document_id": doc_id, "sequence_number": i},
            )
            for i in range(n)
        ]

    def test_empty_list_is_noop(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        mock_driver_cls.reset_mock()
        store.add_documents([])
        mock_driver_cls.return_value.session.assert_not_called()

    def test_merges_document_and_chunk_nodes(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunks = self._make_chunks(2)

        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(MagicMock(), *args, **kwargs)

        store.add_documents(chunks)
        session.execute_write.assert_called()
        index_query = session.run.call_args.args[0]
        assert "CREATE VECTOR INDEX `ai4rag_col__embedding`" in index_query

    def test_no_sequential_chunk_links_are_written(self, mock_driver_cls, mock_embedding, neo4j_config):
        """Chunk order lives in ``sequence_number``; no chunk-to-chunk relationship is stored."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunks = self._make_chunks(3)

        tx = MagicMock()
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)

        store.add_documents(chunks)

        all_cypher = " ".join(str(c) for c in tx.run.call_args_list)
        assert "UNWIND $rows AS row" in all_cypher
        assert "NEXT_CHUNK" not in all_cypher
        assert len(tx.run.call_args_list) == 2

    def test_deduplicates_chunks(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunk = AI4RAGChunk(text="same text", metadata={"document_id": "d", "sequence_number": 0})
        duplicates = [chunk, chunk]

        tx = MagicMock()
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)

        store.add_documents(duplicates)
        # Only one MERGE for the Chunk node (after dedup)
        merge_chunk_calls = [c for c in tx.run.call_args_list if "Chunk" in str(c) and "MERGE (c:" in str(c)]
        assert len(merge_chunk_calls) == 1

    def test_upserts_each_write_batch_with_set_based_cypher(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunks = self._make_chunks(3)
        tx = MagicMock()
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)

        store.add_documents(chunks)

        assert tx.run.call_count == 2
        assert all("UNWIND $rows AS row" in call.args[0] for call in tx.run.call_args_list)
        assert len(tx.run.call_args_list[1].kwargs["rows"]) == 3
        assert not any("NEXT_CHUNK" in call.args[0] for call in tx.run.call_args_list)

    def test_forwards_kg_concurrency_override(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunks = self._make_chunks(1)

        with patch.object(store, "_run_kg_pipeline") as run_pipeline:
            store.add_documents(chunks, model=MagicMock(), kg_max_concurrent=4)

        assert run_pipeline.call_args.kwargs["max_concurrent"] == 4

    def test_pipeline_reuses_canonical_chunk_id(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunk = self._make_chunks(1)[0]
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        store.add_documents([chunk], model=MagicMock())
        assert not any("SET kc:ai4rag_col" in call.args[0] for call in session.run.call_args_list)
        assert not any("apoc.refactor.mergeNodes" in call.args[0] for call in session.run.call_args_list)
        constraint_queries = [call.args[0] for call in session.run.call_args_list]
        assert any("ai4rag_entity_identity" in query for query in constraint_queries)

    def test_kg_prompt_includes_synthetic_format_examples(self, mock_driver_cls, mock_embedding, neo4j_config):
        """Few-shot examples specify both populated and empty chunk output."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        prompt = store._build_kg_batch_prompt(self._make_chunks(1))

        assert "Examples (format only; do not copy their facts or IDs):" in prompt
        assert '"chunk_id":"example-relationship"' in prompt
        assert '"chunk_id":"example-empty","nodes":[],"relationships":[]' in prompt

    def test_creates_composite_canonical_lookup_constraints(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        store._ensure_canonical_schema()
        store._ensure_canonical_schema()

        queries = [call.args[0] for call in session.run.call_args_list]
        assert queries.count(next(query for query in queries if "ai4rag_chunk_collection_id" in query)) == 1
        assert "REQUIRE (c.collection, c.id) IS UNIQUE" in " ".join(queries)
        assert "REQUIRE (d.collection, d.id) IS UNIQUE" in " ".join(queries)

    def test_kg_pipeline_prompt_preserves_canonical_chunk_text(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        text = "A long canonical chunk. " * 200
        chunk = AI4RAGChunk(text=text, metadata={"document_id": "document.md"})

        prompt = store._build_kg_batch_prompt([chunk])
        assert "at most 5 entities and at most 5 relationships" in prompt
        assert f"at most {KG_MAX_ENTITIES_PER_CHUNK} entities" in prompt
        assert f"{KG_MAX_RELATIONSHIPS_PER_CHUNK} relationships" in prompt
        assert "Never exceed these limits" in prompt
        assert "generate a concise, factual description of no more than 8 words" in prompt
        assert json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1]) == [
            {"chunk_id": chunk.chunk_id, "text": text}
        ]

    def test_kg_response_uses_maas_compatible_json_object_mode(self, _mock_driver_cls):
        assert KG_BATCH_RESPONSE_FORMAT == {"type": "json_object"}

    def test_free_mode_prompt_honors_configured_graph_limits(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
            kg_extraction_config={"mode": "free", "max_entities_per_chunk": 9, "max_relationships_per_chunk": 11},
        )

        prompt = store._build_kg_batch_prompt(self._make_chunks(1))

        assert "at most 9 entities and at most 11 relationships" in prompt

    def test_kg_pipeline_does_not_log_chunk_text(self, mock_driver_cls, mock_embedding, neo4j_config, caplog):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        secret_text = "this document text must not appear in logs"
        chunk = AI4RAGChunk(text=secret_text, metadata={"document_id": "document.md"})
        graphrag_logger = logging.getLogger("neo4j_graphrag.experimental.pipeline.config.runner")
        original_level = graphrag_logger.level

        try:
            with (
                caplog.at_level(logging.INFO),
                patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)),
            ):
                store.add_documents([chunk], model=MagicMock())
        finally:
            assert graphrag_logger.level == original_level

        assert secret_text not in caplog.text

    def test_kg_pipeline_logs_entity_extraction_lifecycle(self, mock_driver_cls, mock_embedding, neo4j_config, caplog):
        """Entity extraction logs its start and successful completion without document content."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunks = self._make_chunks(2)

        with (
            caplog.at_level(logging.INFO),
            patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)),
        ):
            store._run_kg_pipeline(chunks, MagicMock(), perform_entity_resolution=False)

        assert "Starting entity extraction for 2 chunks (max_concurrent=4, collection=ai4rag_col)." in caplog.messages
        assert "Finished entity extraction for 2 chunks (collection=ai4rag_col)." in caplog.messages

    def test_kg_pipeline_sends_one_chunk_per_request(self, mock_driver_cls, mock_embedding, neo4j_config):
        """Each chunk is extracted in its own request, whatever the collection size."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        request_sizes: list[int] = []
        completion_limits: list[int] = []

        async def run_pipeline(_, prompt, completion_limit):
            payload = json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1])
            request_sizes.append(len(payload))
            completion_limits.append(completion_limit)
            return SimpleNamespace(
                content=json.dumps({"chunks": [{"chunk_id": payload[0]["chunk_id"], "nodes": [], "relationships": []}]})
            )

        with patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline):
            store._run_kg_pipeline(self._make_chunks(3), MagicMock(), perform_entity_resolution=False)

        assert request_sizes == [1, 1, 1]
        assert completion_limits == [1024, 1024, 1024]

    def test_kg_pipeline_skips_a_chunk_after_an_sdk_exhausted_request(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        """A transport failure is not retried outside the foundation-model SDK."""
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
            kg_extraction_config={"partial_failure_policy": "degrade", "max_skipped_chunks": 1},
        )
        completion_limits: list[int] = []

        class GatewayError(Exception):
            status_code = 502

        async def run_pipeline(_, prompt, completion_limit):
            payload = json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1])
            failed_request = not completion_limits
            completion_limits.append(completion_limit)
            if failed_request:
                raise GatewayError("gateway reset")
            return SimpleNamespace(
                content=json.dumps({"chunks": [{"chunk_id": payload[0]["chunk_id"], "nodes": [], "relationships": []}]})
            )

        with patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline):
            store._run_kg_pipeline(self._make_chunks(2), MagicMock(), perform_entity_resolution=False)

        assert completion_limits == [1024, 1024]
        assert not store.is_kg_extraction_degraded

    def test_kg_pipeline_skips_a_chunk_with_a_malformed_response(self, mock_driver_cls, mock_embedding, neo4j_config):
        """Malformed output is not retried; the affected chunk is skipped alone."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        request_sizes: list[int] = []

        async def run_pipeline(_, prompt, _completion_limit):
            payload = json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1])
            malformed_request = not request_sizes
            request_sizes.append(len(payload))
            if malformed_request:
                return SimpleNamespace(content='{"chunks":')
            return SimpleNamespace(
                content=json.dumps({"chunks": [{"chunk_id": payload[0]["chunk_id"], "nodes": [], "relationships": []}]})
            )

        with patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline):
            store._run_kg_pipeline(self._make_chunks(2), MagicMock(), perform_entity_resolution=False)

        assert request_sizes == [1, 1]

    def test_kg_pipeline_skips_a_chunk_omitted_from_a_valid_response(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        """A response that omits its own chunk is skipped without a second request."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        request_sizes: list[int] = []

        async def run_pipeline(_, prompt, _completion_limit):
            payload = json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1])
            omitted_request = not request_sizes
            request_sizes.append(len(payload))
            if omitted_request:
                return SimpleNamespace(content=json.dumps({"chunks": []}))
            return SimpleNamespace(
                content=json.dumps({"chunks": [{"chunk_id": payload[0]["chunk_id"], "nodes": [], "relationships": []}]})
            )

        with patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline):
            store._run_kg_pipeline(self._make_chunks(2), MagicMock(), perform_entity_resolution=False)

        assert request_sizes == [1, 1]

    def test_kg_pipeline_configures_the_strict_json_schema_response_mode(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        """The default extraction mode asks the endpoint for a required JSON Schema."""
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
        )
        captured: dict = {}

        class RecordingAdapter(_LLMAdapter):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch("ai4rag.rag.vector_store.neo4j._LLMAdapter", RecordingAdapter):
            store._run_kg_pipeline(self._make_chunks(2), MagicMock(), perform_entity_resolution=False)

        assert captured["response_format"] == KG_BATCH_JSON_SCHEMA_RESPONSE_FORMAT

    def test_kg_pipeline_supports_the_json_object_fallback(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
            kg_extraction_config={"response_format": "json_object"},
        )
        captured: dict = {}

        class RecordingAdapter(_LLMAdapter):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch("ai4rag.rag.vector_store.neo4j._LLMAdapter", RecordingAdapter):
            store._run_kg_pipeline(self._make_chunks(1), MagicMock(), perform_entity_resolution=False)

        assert captured["response_format"] == KG_BATCH_RESPONSE_FORMAT

    def test_kg_batch_prompt_includes_the_required_json_contract(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        prompt = store._build_kg_batch_prompt(self._make_chunks(1))

        assert "Return only this JSON object" in prompt
        assert "Examples (format only" in prompt

    def test_kg_pipeline_raises_when_every_chunk_is_skipped(
        self, mock_driver_cls, mock_embedding, neo4j_config, caplog
    ):
        """A completely empty extraction fails instead of leaving a silently empty graph."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        class GatewayError(Exception):
            status_code = 502

        async def run_pipeline(*_):
            raise GatewayError("gateway reset")

        with (
            caplog.at_level(logging.ERROR),
            patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline),
            pytest.raises(_KGExtractionDegradedError, match="produced no entities"),
        ):
            store._run_kg_pipeline([self._make_chunks(1)[0]], MagicMock(), perform_entity_resolution=False)

        assert "Skipping KG extraction" in caplog.text
        assert store.is_kg_extraction_degraded

    def test_all_skipped_chunks_degrade_when_fail_on_empty_kg_is_disabled(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        """Operators can trade a hard failure for chunks-only indexing."""
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
            kg_extraction_config={"fail_on_empty_kg": False, "max_skipped_chunks": 5},
        )

        class BadRequest(Exception):
            status_code = 400

        async def run_pipeline(*_):
            raise BadRequest("invalid payload")

        with patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline):
            store._run_kg_pipeline(self._make_chunks(2), MagicMock(), perform_entity_resolution=False)

        assert store.is_kg_extraction_degraded

    def test_breaker_persists_graphs_extracted_before_it_trips(self, mock_driver_cls, mock_embedding, neo4j_config):
        """A retry must not re-pay for chunks that were already extracted."""
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
            kg_extraction_config={"rate_limit_breaker_threshold": 2},
        )
        attempts = 0

        async def run_pipeline(_, prompt, _completion_limit):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                payload = json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1])
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "chunks": [
                                {
                                    "chunk_id": payload[0]["chunk_id"],
                                    "nodes": [{"id": "n1", "label": "Person", "properties": {"name": "Ada"}}],
                                    "relationships": [],
                                }
                            ]
                        }
                    )
                )
            raise _HttpStatusError(429)

        writer = MagicMock()
        with (
            patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline),
            patch("ai4rag.rag.vector_store.neo4j._CanonicalKGWriter", return_value=writer),
            pytest.raises(_KGExtractionRateLimitedError),
        ):
            store._run_kg_pipeline(self._make_chunks(4), MagicMock(), max_concurrent=1, perform_entity_resolution=False)

        written = writer.write_graphs.call_args.args[0]
        assert len(written) == 1

    def test_pipeline_can_skip_scoped_entity_resolution(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        store._run_kg_pipeline([self._make_chunks(1)[0]], MagicMock(), perform_entity_resolution=False)
        assert not any("apoc.refactor.mergeNodes" in call.args[0] for call in session.run.call_args_list)


# ---------------------------------------------------------------------------
# search — vector mode
# ---------------------------------------------------------------------------

_CYPHER_RETRIEVER_PATH = "ai4rag.rag.vector_store.neo4j.VectorCypherRetriever"
_VECTOR_RETRIEVER_PATH = "ai4rag.rag.vector_store.neo4j.VectorRetriever"


@pytest.fixture(autouse=True)
def mock_vector_retriever():
    """Avoid GraphRAG's concrete-driver validation in Neo4j unit tests."""
    with patch(_VECTOR_RETRIEVER_PATH) as retriever:
        retriever.return_value.search.return_value = _make_retriever_result([])
        yield retriever


def _make_retriever_result(items):
    """Build a mock RetrieverResult with the given items."""
    result = MagicMock()
    result.items = items
    return result


class _HttpStatusError(Exception):
    """Provider failure carrying an HTTP status and optional Retry-After header."""

    def __init__(self, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(status_code=status_code, headers=headers)


def _chat_choice(content: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))


# ---------------------------------------------------------------------------
# KG extraction resilience
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestKGExtractionResilience:
    def _make_chunks(self, n=3, doc_id="doc1"):
        return [
            AI4RAGChunk(text=f"chunk {i}", metadata={"document_id": doc_id, "sequence_number": i}) for i in range(n)
        ]

    def test_classifies_retryable_and_permanent_failures(self, _mock_driver_cls):
        assert is_retryable_llm_error(_HttpStatusError(429))
        assert is_retryable_llm_error(_HttpStatusError(500))
        assert is_retryable_llm_error(APIConnectionError(request=Request("POST", "http://localhost")))
        assert not is_retryable_llm_error(_HttpStatusError(400))
        assert not is_retryable_llm_error(ValueError("bad payload"))

    def test_detects_rate_limit_errors_only(self, _mock_driver_cls):
        assert is_rate_limit_llm_error(_HttpStatusError(429))
        assert not is_rate_limit_llm_error(_HttpStatusError(500))

    def test_backoff_honours_retry_after_and_caps_it(self, _mock_driver_cls):
        assert kg_retry_delay_seconds(0, _HttpStatusError(429, retry_after="7"), 1.0) == 7.0
        assert (
            kg_retry_delay_seconds(3, _HttpStatusError(429, retry_after="9999"), 1.0)
            == KG_LLM_RETRY_MAX_BACKOFF_SECONDS
        )

    def test_backoff_grows_exponentially_with_jitter(self, _mock_driver_cls):
        assert 0.5 <= kg_retry_delay_seconds(0, _HttpStatusError(503), 1.0) <= 1.0
        assert 0.5 <= kg_retry_delay_seconds(1, _HttpStatusError(503), 1.0) <= 2.0
        assert 0.5 <= kg_retry_delay_seconds(2, _HttpStatusError(503), 1.0) <= 4.0

    def test_adapter_retries_a_rate_limited_call(self, _mock_driver_cls):
        model = MagicMock(model_id="test-model")
        model.chat.side_effect = [_HttpStatusError(429), [_chat_choice()]]
        adapter = _LLMAdapter(model, max_retries=2, retry_base_seconds=0.01)

        with patch("ai4rag.rag.vector_store.neo4j_utils.sleep"):
            response = adapter.invoke([{"role": "user", "content": "hello"}])

        assert json.loads(response.content) == {"nodes": [], "relationships": []}
        assert model.chat.call_count == 2
        assert adapter.retry_metrics == (1, 0)

    def test_adapter_surfaces_permanent_errors_without_retrying(self, _mock_driver_cls):
        model = MagicMock(model_id="test-model")
        model.chat.side_effect = _HttpStatusError(400)
        adapter = _LLMAdapter(model, max_retries=2, retry_base_seconds=0.01)

        with (
            patch("ai4rag.rag.vector_store.neo4j_utils.sleep") as sleep,
            pytest.raises(_HttpStatusError),
        ):
            adapter.invoke([{"role": "user", "content": "hello"}])

        assert model.chat.call_count == 1
        sleep.assert_not_called()
        assert adapter.retry_metrics == (0, 1)

    def test_adapter_retries_connection_errors(self, _mock_driver_cls):
        model = MagicMock(model_id="test-model")
        model.chat.side_effect = [
            APIConnectionError(request=Request("POST", "http://localhost")),
            [_chat_choice()],
        ]
        adapter = _LLMAdapter(model, max_retries=1, retry_base_seconds=0.01)

        with patch("ai4rag.rag.vector_store.neo4j_utils.sleep"):
            adapter.invoke([{"role": "user", "content": "hello"}])

        assert model.chat.call_count == 2
        assert adapter.retry_metrics == (1, 0)

    def test_adapter_counts_every_attempt(self, _mock_driver_cls):
        model = MagicMock(model_id="test-model")
        model.chat.side_effect = [_HttpStatusError(429), [_chat_choice()]]
        adapter = _LLMAdapter(model, max_retries=2, retry_base_seconds=0.01)

        with patch("ai4rag.rag.vector_store.neo4j_utils.sleep"):
            adapter.invoke([{"role": "user", "content": "hello"}])

        assert adapter.metrics[0] == 2

    def test_adapter_reports_exhausted_retries(self, _mock_driver_cls):
        model = MagicMock(model_id="test-model")
        model.chat.side_effect = _HttpStatusError(429)
        adapter = _LLMAdapter(model, max_retries=2, retry_base_seconds=0.01)

        with (
            patch("ai4rag.rag.vector_store.neo4j_utils.sleep"),
            pytest.raises(_HttpStatusError),
        ):
            adapter.invoke([{"role": "user", "content": "hello"}])

        assert model.chat.call_count == 3
        assert adapter.retry_metrics == (2, 1)

    def test_rejects_invalid_adapter_retry_settings(self, _mock_driver_cls):
        with pytest.raises(ValueError, match="max_retries"):
            _LLMAdapter(MagicMock(model_id="m"), max_retries=-1)
        with pytest.raises(ValueError, match="retry_base_seconds"):
            _LLMAdapter(MagicMock(model_id="m"), retry_base_seconds=0)
        with pytest.raises(ValueError, match="sdk_max_retries"):
            _LLMAdapter(MagicMock(model_id="m"), sdk_max_retries=-1)

    def test_model_view_disables_sdk_retries_without_mutating_the_original(self, _mock_driver_cls):
        client = OpenAI(api_key="test-key", base_url="http://localhost:1234/v1")
        model = OpenAIFoundationModel(client=client, model_id="test-model")

        view = _model_with_sdk_retries(model, 0)

        assert view is not model
        assert view.model_id == model.model_id
        assert view.client.max_retries == 0
        assert model.client is client
        assert client.max_retries > 0

    def test_model_view_passes_through_models_without_an_openai_client(self, _mock_driver_cls):
        model = MagicMock(model_id="test-model")

        assert _model_with_sdk_retries(model, 0) is model

    def test_model_view_wraps_a_duck_typed_client(self, _mock_driver_cls):
        class WrappedClient:
            def with_options(self, **kwargs):
                self.kwargs = kwargs
                return self

        model = SimpleNamespace(model_id="test-model", client=WrappedClient())

        view = _model_with_sdk_retries(model, 0)

        assert view is not model
        assert view.client.kwargs == {"max_retries": 0}

    def test_adapter_applies_the_configured_sdk_retries(self, _mock_driver_cls):
        model = MagicMock(model_id="test-model")
        model.chat.return_value = [_chat_choice()]

        with patch("ai4rag.rag.vector_store.neo4j_utils._model_with_sdk_retries", return_value=model) as view:
            _LLMAdapter(model, sdk_max_retries=1)

        view.assert_called_once_with(model, 1)

    def test_breaker_aborts_extraction_after_repeated_rate_limits(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
            kg_extraction_config={"rate_limit_breaker_threshold": 2},
        )
        calls = 0

        async def run_pipeline(*_):
            nonlocal calls
            calls += 1
            raise _HttpStatusError(429)

        with (
            patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline),
            pytest.raises(_KGExtractionRateLimitedError),
        ):
            store._run_kg_pipeline(self._make_chunks(8), MagicMock(), perform_entity_resolution=False)

        assert calls <= 4
        assert store.is_kg_extraction_degraded

    def test_breaker_does_not_trip_for_occasional_rate_limits(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(
            mock_embedding,
            neo4j_config,
            collection_name="ai4rag_col",
            kg_extraction_config={"rate_limit_breaker_threshold": 2, "max_skipped_chunks": 1},
        )
        attempts = 0

        async def run_pipeline(_, prompt, _completion_limit):
            nonlocal attempts
            attempts += 1
            payload = json.loads(prompt.rsplit("Input chunks:\n", maxsplit=1)[1])
            if attempts == 1:
                raise _HttpStatusError(429)
            return SimpleNamespace(
                content=json.dumps({"chunks": [{"chunk_id": payload[0]["chunk_id"], "nodes": [], "relationships": []}]})
            )

        with patch.object(_LLMAdapter, "ainvoke_with_completion_limit", run_pipeline):
            store._run_kg_pipeline(self._make_chunks(3), MagicMock(), perform_entity_resolution=False)

        assert attempts == 3
        assert not store.is_kg_extraction_degraded


def _make_retriever_item(text, score=0.9, meta=None):
    item = MagicMock()
    item.content = text
    item.metadata = {"score": score, "_meta": meta or {}}
    return item


# ---------------------------------------------------------------------------
# search — graph mode
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestSearchGraph:
    def test_waits_for_vector_index_once_across_concurrent_searches(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda _: store._ensure_kg_schema(wait_until_online=True), range(10)))

        create_calls = [call for call in session.run.call_args_list if "CREATE VECTOR INDEX" in call.args[0]]
        wait_calls = [call for call in session.run.call_args_list if "CALL db.awaitIndex" in call.args[0]]
        assert len(create_calls) == 1
        assert len(wait_calls) == 1

    def test_uses_cypher_retriever_with_collection_index(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=2, search_mode="graph")

        _, init_kwargs = mock_cr_cls.call_args
        assert init_kwargs["index_name"] == "ai4rag_col__embedding"
        mock_cr_cls.return_value.search.assert_called_once_with(
            query_vector=[0.1] * 128, top_k=10, query_params={"col": "ai4rag_col"}
        )

    def test_retrieval_query_does_not_expand_sequential_neighbors(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=1, search_mode="graph")

        _, init_kwargs = mock_cr_cls.call_args
        assert "NEXT_CHUNK" not in init_kwargs["retrieval_query"]
        assert "node.collection = $col" in init_kwargs["retrieval_query"]

    def test_graph_search_passes_collection_param(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_kg2")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=1, search_mode="graph")

        mock_cr_cls.return_value.search.assert_called_once_with(
            query_vector=[0.1] * 128, top_k=10, query_params={"col": "ai4rag_kg2"}
        )

    def test_graph_expansion_maps_heap_exhaustion_to_memory_error(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        class _HeapExhausted(Exception):
            code = "Neo4j.TransientError.General.OutOfMemoryError"

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.side_effect = _HeapExhausted("Java heap space")
            with pytest.raises(Neo4jMemoryError, match="entity_relationship_hops"):
                store.search("q", k=1, search_mode="graph", entity_relationship_hops=2)

    def test_graph_expansion_propagates_other_failures(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                store.search("q", k=1, search_mode="graph")

    def test_graph_search_embeds_query_once_for_both_routes(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        vector = [0.25] * 8

        with (
            patch.object(mock_embedding, "embed_query", return_value=vector) as embed_query,
            patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls,
            patch(_VECTOR_RETRIEVER_PATH) as mock_vr_cls,
        ):
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            mock_vr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=1, search_mode="graph")

        embed_query.assert_called_once_with("q")
        assert mock_cr_cls.return_value.search.call_args.kwargs["query_vector"] == vector
        assert mock_vr_cls.return_value.search.call_args.kwargs["query_vector"] == vector

    def test_retrieval_query_contains_entity_expansion(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=1, search_mode="graph", include_entity_neighbors=True)

        _, init_kwargs = mock_cr_cls.call_args
        assert "__Entity__" in init_kwargs["retrieval_query"]
        assert "FROM_CHUNK" in init_kwargs["retrieval_query"]
        assert "ent_nb.collection = $col" in init_kwargs["retrieval_query"]
        assert "NEXT_CHUNK" not in init_kwargs["retrieval_query"]

    def test_returns_chunks_from_graph_search(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        item = MagicMock()
        item.content = "seed text with context"
        item.metadata = {"score": 0.88}
        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([item])
            results = store.search("q", k=1, search_mode="graph")

        assert len(results) == 1
        assert isinstance(results[0], AI4RAGChunk)
        assert results[0].text == "seed text with context"

    def test_vector_route_preserves_chunk_document_metadata(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_VECTOR_RETRIEVER_PATH) as mock_vr_cls:
            mock_vr_cls.return_value.search.return_value = _make_retriever_result([])
            store._search_vector_route([0.1] * 128, k=1)

        formatter = mock_vr_cls.call_args.kwargs["result_formatter"]
        item = formatter(
            {
                "node": {
                    "text": "chunk text",
                    "document_id": "datasets/rag/william/william.md",
                    "metadata": json.dumps({"source": "datasets/rag/william/william.md", "sequence_number": 2}),
                },
                "score": 0.9,
            }
        )

        assert item.content == "chunk text"
        assert item.metadata["_meta"] == {
            "document_id": "datasets/rag/william/william.md",
            "source": "datasets/rag/william/william.md",
            "sequence_number": 2,
            "route": "vector",
        }

    def test_nonzero_graph_hops_raises(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", graph_hops=1)

    def test_invalid_entity_neighbor_limit_raises(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="entity_neighbor_limit"):
            _validate_neo4j_search_params("graph", entity_neighbor_limit=-1)


# ---------------------------------------------------------------------------
# build_knowledge_graph_from_documents
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestBuildKnowledgeGraphFromDocuments:
    """Docling input must become canonical chunks before KG extraction."""

    def _make_docling_doc(self, text="Document text content.", name="document.md"):
        doc = MagicMock()
        doc.export_to_markdown.return_value = text
        doc.name = name
        return doc

    def test_uses_one_document_id_and_canonical_chunks_per_input(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        model = MagicMock()
        doc1 = self._make_docling_doc("Text one.", "one.md")
        doc2 = self._make_docling_doc("Text two.", "two.md")

        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(documents=[doc1, doc2], model=model)

        doc1.export_to_markdown.assert_called_once()
        doc2.export_to_markdown.assert_called_once()
        chunks = add_documents.call_args.args[0]
        assert len(chunks) == 2
        assert [chunk.metadata["document_id"] for chunk in chunks] == ["one.md", "two.md"]
        assert all(isinstance(chunk, AI4RAGChunk) for chunk in chunks)
        assert add_documents.call_args.kwargs["model"] is model

    def test_empty_text_skips_pipeline(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        model = MagicMock()
        doc = self._make_docling_doc("")  # export produces empty string

        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(documents=[doc], model=model)

        add_documents.assert_not_called()

    def test_passes_extraction_options_to_shared_indexer(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(
                documents=[self._make_docling_doc()],
                model=MagicMock(),
                chunk_size=500,
                chunk_overlap=20,
                on_error="RAISE",
                perform_entity_resolution=False,
            )

        assert "kg_chunk_size" not in add_documents.call_args.kwargs
        assert "kg_chunk_overlap" not in add_documents.call_args.kwargs
        assert add_documents.call_args.kwargs["on_error"] == "RAISE"
        assert add_documents.call_args.kwargs["perform_entity_resolution"] is False

    def test_repeated_document_produces_the_same_chunk_ids(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        doc = self._make_docling_doc("Alice works at Acme.", "people.md")
        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(documents=[doc], model=MagicMock())
            first_ids = [chunk.chunk_id for chunk in add_documents.call_args.args[0]]
            store.build_knowledge_graph_from_documents(documents=[doc], model=MagicMock())
            second_ids = [chunk.chunk_id for chunk in add_documents.call_args.args[0]]

        assert first_ids == second_ids

    def test_writes_one_document_and_chunk_set(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        tx = MagicMock()
        session.execute_write.side_effect = lambda fn, *args: fn(tx, *args)

        store.build_knowledge_graph_from_documents(
            documents=[self._make_docling_doc("Alice works at Acme.")], model=MagicMock()
        )

        queries = [call.args[0] for call in tx.run.call_args_list]
        assert sum("MERGE (d:ai4rag_col:Document" in query for query in queries) == 1
        assert sum("MERGE (c:ai4rag_col:Chunk" in query for query in queries) == 1
        assert not any("FROM_DOCUMENT" in query for query in queries)


def _make_canonical_writer(driver):
    return _CanonicalKGWriter(driver, "tenant_graph", "ai4rag_col")


def _executed_rows(driver):
    """Map writer transaction function name to the rows it was given."""
    session = driver.session.return_value.__enter__.return_value
    return {call.args[0].__name__: call.args[1] for call in session.execute_write.call_args_list}


def test_canonical_writer_skips_lexical_graph_nodes():
    driver = MagicMock()
    writer = _make_canonical_writer(driver)
    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(
                id="pipeline_doc",
                label=_KG_LEXICAL_GRAPH_CONFIG.document_node_label,
                properties={"ai4rag_chunk_id": "c1"},
            ),
            Neo4jNode(id="pipeline_chunk", label=_KG_LEXICAL_GRAPH_CONFIG.chunk_node_label),
            Neo4jNode(id="alice", label="Person", properties={"name": "Alice"}),
        ],
        relationships=[Neo4jRelationship(start_node_id="alice", end_node_id="pipeline_chunk", type="FROM_CHUNK")],
    )

    writer.write_graphs([("c1", graph)])

    rows = _executed_rows(driver)
    node_rows = rows["_write_entity_batch_tx"]
    assert [row["labels"] for row in node_rows] == [["Person", "__Entity__"]]
    assert node_rows[0]["chunk_ids"] == ["c1"]
    assert "_write_relationship_batch_tx" not in rows
    assert list(rows) == ["_write_entity_batch_tx"]


def test_canonical_writer_keeps_lexical_labels_as_entity_names():
    driver = MagicMock()
    writer = _make_canonical_writer(driver)
    graph = Neo4jGraph(nodes=[Neo4jNode(id="report", label="Document", properties={"name": "Report"})])

    writer.write_graphs([("c1", graph)])

    node_rows = _executed_rows(driver)["_write_entity_batch_tx"]
    assert node_rows == [
        {
            "key": node_rows[0]["key"],
            "labels": ["Document", "__Entity__"],
            "properties": {"name": "Report"},
            "chunk_ids": ["c1"],
        }
    ]


def test_canonical_writer_deduplicates_entities_shared_by_chunks():
    driver = MagicMock()
    writer = _make_canonical_writer(driver)
    alice = Neo4jNode(id="a", label="Person", properties={"name": "Alice"})
    acme = Neo4jNode(id="b", label="Organization", properties={"name": "Acme"})
    first = Neo4jGraph(
        nodes=[alice, acme],
        relationships=[Neo4jRelationship(start_node_id="a", end_node_id="b", type="WORKS_AT")],
    )
    second = Neo4jGraph(
        nodes=[
            Neo4jNode(id="a2", label="Person", properties={"name": "alice"}),
            Neo4jNode(id="b2", label="Organization", properties={"name": "ACME"}),
        ],
        relationships=[Neo4jRelationship(start_node_id="a2", end_node_id="b2", type="WORKS_AT")],
    )

    writer.write_graphs([("c1", first), ("c2", second)])

    rows = _executed_rows(driver)
    node_rows = rows["_write_entity_batch_tx"]
    assert len(node_rows) == 2
    assert len(rows["_write_relationship_batch_tx"]) == 1
    assert all(row["chunk_ids"] == ["c1", "c2"] for row in node_rows)


def test_canonical_writer_merges_duplicate_node_properties():
    driver = MagicMock()
    writer = _make_canonical_writer(driver)
    graphs = [
        ("c1", Neo4jGraph(nodes=[Neo4jNode(id="a", label="Person", properties={"name": "Alice", "role": "dev"})])),
        ("c2", Neo4jGraph(nodes=[Neo4jNode(id="b", label="Person", properties={"name": "alice", "role": "lead"})])),
    ]

    writer.write_graphs(graphs)

    rows = _executed_rows(driver)
    # Last extracted value wins, but the entity is still written exactly once.
    assert rows["_write_entity_batch_tx"][0]["properties"] == {"name": "alice", "role": "lead"}
    assert rows["_write_entity_batch_tx"][0]["chunk_ids"] == ["c1", "c2"]


def test_canonical_writer_keeps_unnamed_entities_chunk_local():
    driver = MagicMock()
    writer = _make_canonical_writer(driver)

    writer.write_graphs(
        [
            ("c1", Neo4jGraph(nodes=[Neo4jNode(id="n1", label="Person", properties={})])),
            ("c2", Neo4jGraph(nodes=[Neo4jNode(id="n1", label="Person", properties={})])),
        ]
    )

    rows = _executed_rows(driver)
    node_rows = rows["_write_entity_batch_tx"]
    assert len(node_rows) == 2
    assert node_rows[0]["key"] != node_rows[1]["key"]
    assert sorted(node_rows[0]["chunk_ids"] + node_rows[1]["chunk_ids"]) == ["c1", "c2"]


def test_canonical_writer_writes_entities_relationships_and_links():
    driver = MagicMock()
    writer = _make_canonical_writer(driver)
    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(id="a", label="Person", properties={"name": "Alice"}),
            Neo4jNode(id="b", label="Organization", properties={"name": "Acme"}),
        ],
        relationships=[Neo4jRelationship(start_node_id="a", end_node_id="b", type="WORKS_AT")],
    )

    writer.write_graphs([("c1", graph)])

    session = driver.session.return_value.__enter__.return_value
    calls = session.execute_write.call_args_list
    assert len(calls) == 2

    entity_tx = MagicMock()
    calls[0].args[0](entity_tx, calls[0].args[1], calls[0].args[2])
    entity_query = entity_tx.run.call_args.args[0]
    assert "apoc.merge.node" in entity_query
    assert "ai4rag_entity_key" in entity_query
    assert "UNWIND row.chunk_ids AS chunk_id" in entity_query
    assert "FROM_CHUNK" in entity_query

    relationship_tx = MagicMock()
    calls[1].args[0](relationship_tx, calls[1].args[1], calls[1].args[2])
    assert "apoc.merge.relationship" in relationship_tx.run.call_args.args[0]

    assert writer.write_elapsed_seconds >= 0.0


def test_canonical_writer_batches_large_flushes():
    driver = MagicMock()
    writer = _make_canonical_writer(driver)
    writer.batch_size = 2
    nodes = [Neo4jNode(id=f"n{i}", label="Person", properties={"name": f"Person {i}"}) for i in range(2)] + [
        Neo4jNode(id=f"m{i}", label="Organization", properties={"name": f"Org {i}"}) for i in range(2)
    ]
    graph = Neo4jGraph(nodes=nodes)

    writer.write_graphs([("c1", graph)])

    rows = driver.session.return_value.__enter__.return_value.execute_write.call_args_list
    assert [len(call.args[1]) for call in rows] == [2, 2]


# ---------------------------------------------------------------------------
# clean_collection / close
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestCleanAndClose:
    def test_clean_collection_drops_collection_vector_index(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.run.reset_mock()

        store.clean_collection()

        cypher_calls = " ".join(str(c) for c in session.run.call_args_list)
        assert "ai4rag_col__embedding" in cypher_calls
        assert "__Entity__" in cypher_calls
        assert "ai4rag_kg_collections" in cypher_calls
        assert "WHERE owner <> $col" in cypher_calls
        assert "MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk)" in cypher_calls
        assert "COALESCE(other.collection, '') <> $col" in cypher_calls
        assert "WHERE NOT (e)-[:FROM_CHUNK]-(:Chunk)" not in cypher_calls
        assert "DETACH DELETE" in cypher_calls

        queries = [call.args[0] for call in session.run.call_args_list]
        legacy_cleanup = next(i for i, query in enumerate(queries) if "MATCH (e:__Entity__)-[:FROM_CHUNK]" in query)
        chunk_cleanup = next(i for i, query in enumerate(queries) if "MATCH (n:ai4rag_col)" in query)
        assert legacy_cleanup < chunk_cleanup
        assert any("size(r.ai4rag_kg_collections) = 1 DELETE r" in query for query in queries)

    def test_close_closes_driver(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        store.close()
        mock_driver_cls.return_value.close.assert_called_once()

    def test_context_manager_calls_close(self, mock_driver_cls, mock_embedding, neo4j_config):
        with Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col"):
            pass
        mock_driver_cls.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# _validate_neo4j_search_params
# ---------------------------------------------------------------------------


class TestValidateNeo4jSearchParams:
    @pytest.mark.parametrize("search_mode", ["vector", "hybrid"])
    def test_non_graph_modes_rejected(self, search_mode):
        with pytest.raises(ValueError, match="Invalid search_mode"):
            _validate_neo4j_search_params(search_mode)

    def test_graph_mode_valid(self):
        _validate_neo4j_search_params("graph")

    def test_graph_mode_valid_with_zero_hops(self):
        _validate_neo4j_search_params("graph", graph_hops=0)

    def test_graph_hops_nonzero_raises(self):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", graph_hops=1)

    def test_graph_hops_non_int_raises(self):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", graph_hops=1.5)

    def test_entity_neighbor_limit_negative_raises(self):
        with pytest.raises(ValueError, match="entity_neighbor_limit"):
            _validate_neo4j_search_params("graph", entity_neighbor_limit=-1)

    def test_entity_neighbor_limit_zero_is_valid(self):
        _validate_neo4j_search_params("graph", entity_neighbor_limit=0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="search_mode"):
            _validate_neo4j_search_params("unknown")
