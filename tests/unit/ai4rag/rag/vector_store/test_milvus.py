# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import gc
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import MilvusConfig


class _MockEmbeddingModel:
    """Minimal mock for BaseEmbeddingModel."""

    model_id = "test-embedding"
    params = {"embedding_dimension": 128}

    def embed_documents(self, texts):
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, query):
        return [0.1] * 128


@pytest.fixture
def mock_embedding():
    return _MockEmbeddingModel()


@pytest.fixture
def milvus_config():
    return MilvusConfig(uri="http://localhost:19530")


@patch("ai4rag.rag.vector_store.milvus.MilvusClient")
class TestMilvusVectorStoreInit:

    def test_creates_new_collection(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = False
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        store = MilvusVectorStore(mock_embedding, milvus_config)

        client.has_collection.assert_called_once()
        client.create_collection.assert_called_once()
        assert store.collection_name.startswith("ai4rag_")

    def test_reuses_existing_collection(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        store = MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag_existing")

        client.has_collection.assert_called_once_with("ai4rag_existing")
        client.create_collection.assert_not_called()
        assert store.collection_name == "ai4rag_existing"

    def test_sanitizes_reuse_name(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        store = MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag-col.name")

        assert store.collection_name == "ai4rag_col_name"

    def test_passes_token(self, MockClient, mock_embedding):
        client = MockClient.return_value
        client.has_collection.return_value = True
        cfg = MilvusConfig(uri="http://host:19530", token="root:pw")
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        MilvusVectorStore(mock_embedding, cfg, collection_name="ai4rag_col")
        MockClient.assert_called_with(uri="http://host:19530", token="root:pw")

    def test_no_server_pem_path_without_cert(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag_col")

        _, kwargs = MockClient.call_args
        assert "server_pem_path" not in kwargs

    def test_writes_server_cert_to_tempfile(self, MockClient, mock_embedding):
        client = MockClient.return_value
        client.has_collection.return_value = True
        cert_pem = "-----BEGIN CERTIFICATE-----\nMIICert-unique-A\n-----END CERTIFICATE-----\n"
        cfg = MilvusConfig(uri="https://host:19530", token="root:pw", server_cert=cert_pem)
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        store = MilvusVectorStore(mock_embedding, cfg, collection_name="ai4rag_col")

        _, kwargs = MockClient.call_args
        assert kwargs["uri"] == "https://host:19530"
        assert kwargs["token"] == "root:pw"
        cert_path = Path(kwargs["server_pem_path"])
        assert cert_path.name.startswith("ai4rag-milvus-cert")
        assert cert_path.suffix == ".pem"
        assert cert_path.read_text() == cert_pem
        assert store.collection_name == "ai4rag_col"

    def test_cert_tempfile_survives_close_and_gc(self, MockClient, mock_embedding):
        """The cert file must outlive close()/GC: pymilvus may reconnect and re-read it."""
        client = MockClient.return_value
        client.has_collection.return_value = True
        cfg = MilvusConfig(uri="https://host:19530", server_cert="CERTDATA-survive")
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        store = MilvusVectorStore(mock_embedding, cfg, collection_name="ai4rag_col")
        cert_path = Path(MockClient.call_args[1]["server_pem_path"])
        assert cert_path.exists()

        store.close()
        assert cert_path.exists()  # close() must not remove it

        del store
        gc.collect()
        assert cert_path.exists()  # nor may garbage collection

    def test_identical_cert_reuses_single_tempfile(self, MockClient, mock_embedding):
        """Two stores sharing the same PEM must share one file (no per-store accumulation)."""
        client = MockClient.return_value
        client.has_collection.return_value = True
        cfg = MilvusConfig(uri="https://host:19530", server_cert="CERTDATA-shared")
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        MilvusVectorStore(mock_embedding, cfg, collection_name="ai4rag_a")
        path_a = MockClient.call_args[1]["server_pem_path"]
        MilvusVectorStore(mock_embedding, cfg, collection_name="ai4rag_b")
        path_b = MockClient.call_args[1]["server_pem_path"]

        assert path_a == path_b

    def test_cert_cache_cleaned_at_exit(self, MockClient, mock_embedding):
        """The atexit hook must remove every materialized cert file."""
        client = MockClient.return_value
        client.has_collection.return_value = True
        cfg = MilvusConfig(uri="https://host:19530", server_cert="CERTDATA-atexit")
        from ai4rag.rag.vector_store import milvus as milvus_mod

        milvus_mod.MilvusVectorStore(mock_embedding, cfg, collection_name="ai4rag_col")
        cert_path = Path(MockClient.call_args[1]["server_pem_path"])
        assert cert_path.exists()

        milvus_mod._cleanup_server_certs()
        assert not cert_path.exists()

    def test_close_without_cert_is_safe(self, MockClient, mock_embedding, milvus_config):
        """close() must not fail when no TLS certificate tempfile was created."""
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        store = MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag_col")
        store.close()
        client.close.assert_called_once()


@patch("ai4rag.rag.vector_store.milvus.MilvusClient")
class TestMilvusVectorStoreSearch:

    def _make_store(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        return MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag_test_col")

    def test_vector_search(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        client.search.return_value = [
            [
                {"entity": {"content": "hello", "metadata": {}}, "distance": 0.9},
            ]
        ]

        results = store.search("query", k=1)
        assert len(results) == 1
        assert isinstance(results[0], AI4RAGChunk)
        assert results[0].text == "hello"

    def test_vector_search_requests_strong_consistency(self, MockClient, mock_embedding, milvus_config):
        """Regression test: a query right after an upsert must not race Milvus's
        default Bounded-staleness consistency and see stale (zero) results."""
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value
        client.search.return_value = [[]]

        store.search("query", k=1)

        _, kwargs = client.search.call_args
        assert kwargs["consistency_level"] == "Strong"
        assert kwargs["output_fields"] == ["content", "metadata"]

    def test_vector_search_missing_metadata_defaults_to_empty_dict(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        client.search.return_value = [[{"entity": {"content": "hello", "metadata": None}, "distance": 0.9}]]

        results = store.search("query", k=1)
        assert results[0].metadata == {}

    def test_vector_search_with_scores(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        client.search.return_value = [
            [
                {"entity": {"content": "hello", "metadata": {}}, "distance": 0.9},
            ]
        ]

        results = store.search("query", k=1, include_scores=True)
        assert len(results) == 1
        chunk, score = results[0]
        assert chunk.text == "hello"
        assert score == 0.9

    def test_hybrid_search_rrf(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        client.hybrid_search.return_value = [
            [
                {"entity": {"content": "result", "metadata": {}}, "distance": 0.8},
            ]
        ]

        results = store.search("query", k=1, search_mode="hybrid", ranker_strategy="rrf")
        assert len(results) == 1

    def test_hybrid_search_weighted(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        client.hybrid_search.return_value = [
            [
                {"entity": {"content": "result", "metadata": {}}, "distance": 0.7},
            ]
        ]

        results = store.search("query", k=1, search_mode="hybrid", ranker_strategy="weighted", ranker_alpha=0.5)
        assert len(results) == 1

    def test_hybrid_search_requests_strong_consistency(self, MockClient, mock_embedding, milvus_config):
        """Same read-your-writes guard as vector search, for the hybrid_search path."""
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value
        client.hybrid_search.return_value = [[]]

        store.search("query", k=1, search_mode="hybrid", ranker_strategy="rrf")

        _, kwargs = client.hybrid_search.call_args
        assert kwargs["consistency_level"] == "Strong"
        assert kwargs["output_fields"] == ["content", "metadata"]


@patch("ai4rag.rag.vector_store.milvus.MilvusClient")
class TestMilvusVectorStoreValidation:

    def _make_store(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        return MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag_test_col")

    def test_invalid_search_mode(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        with pytest.raises(ValueError, match="Invalid search_mode"):
            store.search("q", k=1, search_mode="invalid")

    def test_ranker_strategy_on_vector_mode(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        with pytest.raises(ValueError, match="only valid when search_mode='hybrid'"):
            store.search("q", k=1, search_mode="vector", ranker_strategy="rrf")

    def test_hybrid_without_strategy(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        with pytest.raises(ValueError, match="ranker_strategy must be set"):
            store.search("q", k=1, search_mode="hybrid")

    def test_invalid_ranker_strategy(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        with pytest.raises(ValueError, match="Invalid ranker_strategy"):
            store.search("q", k=1, search_mode="hybrid", ranker_strategy="bogus")

    def test_ranker_k_with_wrong_strategy(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        with pytest.raises(ValueError, match="ranker_k"):
            store.search("q", k=1, search_mode="hybrid", ranker_strategy="weighted", ranker_k=60)

    def test_ranker_alpha_with_wrong_strategy(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        with pytest.raises(ValueError, match="ranker_alpha"):
            store.search("q", k=1, search_mode="hybrid", ranker_strategy="rrf", ranker_alpha=0.5)


@patch("ai4rag.rag.vector_store.milvus.MilvusClient")
class TestMilvusVectorStoreAddDocuments:

    def _make_store(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        return MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag_test_col")

    def test_add_documents(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        docs = [AI4RAGChunk(text="hello", metadata={"document_id": "d1"})]
        store.add_documents(docs)

        client.upsert.assert_called_once()
        call_args = client.upsert.call_args
        assert call_args[0][0] == "ai4rag_test_col"
        row = call_args[1]["data"][0]
        assert len(call_args[1]["data"]) == 1
        assert row["chunk_id"] == docs[0].chunk_id
        assert row["content"] == "hello"
        assert row["metadata"] == {"document_id": "d1"}
        assert "chunk_content" not in row

    def test_add_empty_documents(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        store.add_documents([])
        client.upsert.assert_not_called()

    def test_deduplicates_by_chunk_id(self, MockClient, mock_embedding, milvus_config):
        store = self._make_store(MockClient, mock_embedding, milvus_config)
        client = MockClient.return_value

        chunk = AI4RAGChunk(text="same text", metadata={"document_id": "d1"})
        store.add_documents([chunk, chunk])

        call_args = client.upsert.call_args
        assert len(call_args[1]["data"]) == 1


@patch("ai4rag.rag.vector_store.milvus.MilvusClient")
class TestMilvusVectorStoreCleanCollection:

    def test_clean_collection(self, MockClient, mock_embedding, milvus_config):
        client = MockClient.return_value
        client.has_collection.return_value = True
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        store = MilvusVectorStore(mock_embedding, milvus_config, collection_name="ai4rag_to_drop")
        store.clean_collection()

        client.drop_collection.assert_called_once_with("ai4rag_to_drop")
