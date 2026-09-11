# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.config import MilvusConfig, MilvusLiteConfig, PGVectorConfig
from ai4rag.rag.vector_store.get_vector_store import get_vector_store


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock BaseEmbeddingModel for testing."""

    def __init__(self):
        self.client = MagicMock()
        self.model_id = "test-model"
        self.params = {"embedding_dimension": 128}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mock embed_documents implementation."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        """Mock embed_query implementation."""
        return [0.1, 0.2, 0.3]


def _fake_config(provider: str) -> SimpleNamespace:
    """Build a minimal duck-typed config object exposing only ``.provider``."""
    return SimpleNamespace(provider=provider)


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    return MockEmbeddingModel()


class TestGetVectorStoreMilvus:
    """Test suite for get_vector_store with Milvus provider."""

    @patch("ai4rag.rag.vector_store.milvus.MilvusClient")
    def test_milvus_returns_vector_store(self, MockClient, mock_embedding_model):
        MockClient.return_value.has_collection.return_value = False
        config = MilvusConfig(uri="http://localhost:19530")

        vector_store = get_vector_store(
            embedding_model=mock_embedding_model,
            config=config,
        )

        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        assert isinstance(vector_store, MilvusVectorStore)

    def test_milvus_with_wrong_config_type_raises_type_error(self, mock_embedding_model):
        """A config whose provider claims 'milvus' but isn't a MilvusConfig must raise TypeError."""
        with pytest.raises(TypeError, match="MilvusConfig is required"):
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=PGVectorConfig(provider="milvus"),
            )


class TestGetVectorStoreMilvusLite:
    """Test suite for get_vector_store with the Milvus Lite provider."""

    @patch("ai4rag.rag.vector_store.milvus.MilvusClient")
    def test_milvus_lite_returns_vector_store(self, MockClient, mock_embedding_model):
        """A MilvusLiteConfig selects the embedded engine via the shared MilvusVectorStore."""
        MockClient.return_value.has_collection.return_value = False

        vector_store = get_vector_store(
            embedding_model=mock_embedding_model,
            config=MilvusLiteConfig(db_path="./ai4rag.db"),
            collection_name="ai4rag_my_collection",
        )

        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        assert isinstance(vector_store, MilvusVectorStore)
        assert vector_store.collection_name == "ai4rag_my_collection"
        # The embedded engine connects with the local db_path as its uri, and no
        # auth/TLS kwargs.
        MockClient.assert_called_once_with(uri="./ai4rag.db")

    def test_milvus_lite_with_wrong_config_type_raises_type_error(self, mock_embedding_model):
        """A config whose provider claims 'milvus_lite' but isn't a MilvusLiteConfig must raise TypeError."""
        with pytest.raises(TypeError, match="MilvusLiteConfig is required"):
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=PGVectorConfig(provider="milvus_lite"),
            )


class TestGetVectorStorePGVector:
    """Test suite for get_vector_store with PGVector provider."""

    @patch("ai4rag.rag.vector_store.pgvector.asyncpg.create_pool")
    def test_pgvector_returns_vector_store(self, mock_create_pool, mock_embedding_model):
        config = PGVectorConfig()

        vector_store = get_vector_store(
            embedding_model=mock_embedding_model,
            config=config,
        )

        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        assert isinstance(vector_store, PGVectorStore)

    def test_pgvector_with_wrong_config_type_raises_type_error(self, mock_embedding_model):
        """A config whose provider claims 'pgvector' but isn't a PGVectorConfig must raise TypeError."""
        with pytest.raises(TypeError, match="PGVectorConfig is required"):
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=MilvusConfig(uri="http://localhost:19530", provider="pgvector"),
            )


class TestGetVectorStoreInvalidProvider:
    """Test suite for get_vector_store with an unsupported provider."""

    def test_get_vector_store_invalid_provider_raises_error(self, mock_embedding_model):
        """Test that an unsupported config.provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=_fake_config("invalid_provider"),
            )

        assert "not supported" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)

    def test_get_vector_store_empty_provider_raises_error(self, mock_embedding_model):
        """Test that an empty provider string raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=_fake_config(""),
            )

    def test_get_vector_store_old_ogx_provider_no_longer_supported(self, mock_embedding_model):
        """Test that the retired 'ogx' provider is no longer supported."""
        with pytest.raises(ValueError, match="not supported"):
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=_fake_config("ogx"),
            )


class TestGetVectorStoreEdgeCases:
    """Test suite for edge cases in get_vector_store."""

    @patch("ai4rag.rag.vector_store.milvus.MilvusClient")
    def test_get_vector_store_case_sensitive(self, MockClient, mock_embedding_model):
        """Test that config.provider is case-sensitive."""
        MockClient.return_value.has_collection.return_value = False

        # "milvus" should work
        vector_store = get_vector_store(
            embedding_model=mock_embedding_model,
            config=MilvusConfig(uri="http://localhost:19530"),
        )
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        assert isinstance(vector_store, MilvusVectorStore)

        # "MILVUS" should not work
        with pytest.raises(ValueError):
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=_fake_config("MILVUS"),
            )

    def test_get_vector_store_whitespace_provider(self, mock_embedding_model):
        """Test that a provider string with whitespace raises error."""
        with pytest.raises(ValueError):
            get_vector_store(
                embedding_model=mock_embedding_model,
                config=_fake_config(" milvus "),
            )

    def test_get_vector_store_similar_provider_names(self, mock_embedding_model):
        """Test that similar but incorrect provider names raise errors."""
        invalid_providers = ["chromadb", "milvuslite", "ogx_milvus"]

        for invalid_provider in invalid_providers:
            with pytest.raises(ValueError):
                get_vector_store(
                    embedding_model=mock_embedding_model,
                    config=_fake_config(invalid_provider),
                )


class TestGetVectorStoreReturnTypes:
    """Test suite for verifying return types from get_vector_store."""

    @patch("ai4rag.rag.vector_store.milvus.MilvusClient")
    def test_milvus_returns_base_vector_store_interface(self, MockClient, mock_embedding_model):
        """Test that the Milvus vector store implements the BaseVectorStore interface."""
        MockClient.return_value.has_collection.return_value = False

        vector_store = get_vector_store(
            embedding_model=mock_embedding_model,
            config=MilvusConfig(uri="http://localhost:19530"),
        )

        assert hasattr(vector_store, "search")
        assert hasattr(vector_store, "add_documents")
        assert hasattr(vector_store, "collection_name")
        assert callable(vector_store.search)
        assert callable(vector_store.add_documents)
