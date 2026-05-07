# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest

from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.chroma import ChromaVectorStore
from ai4rag.rag.vector_store.get_vector_store import get_vector_store
from ai4rag.rag.vector_store.ogx import OGXVectorStore


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


class TestGetVectorStoreChroma:
    """Test suite for get_vector_store with Chroma type."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel()

    def test_get_vector_store_chroma_default(self, mocker, mock_embedding_model):
        """Test getting Chroma vector store with default parameters."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())

        vector_store = get_vector_store(
            vs_type="chroma",
            embedding_model=mock_embedding_model,
        )

        assert isinstance(vector_store, ChromaVectorStore)
        assert vector_store.embedding_model == mock_embedding_model

    def test_get_vector_store_chroma_with_collection_name(self, mocker, mock_embedding_model):
        """Test getting Chroma vector store with custom collection name."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())

        vector_store = get_vector_store(
            vs_type="chroma",
            embedding_model=mock_embedding_model,
            reuse_collection_name="my_collection",
        )

        assert isinstance(vector_store, ChromaVectorStore)
        assert vector_store.collection_name == "my_collection"

    def test_get_vector_store_chroma_ignores_client_param(self, mocker, mock_embedding_model):
        """Test that Chroma ignores client parameter."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        mock_client = MagicMock()

        vector_store = get_vector_store(
            vs_type="chroma",
            embedding_model=mock_embedding_model,
            client=mock_client,
        )

        assert isinstance(vector_store, ChromaVectorStore)


class TestGetVectorStoreOGX:
    """Test suite for get_vector_store with ogx type."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel()

    @pytest.fixture
    def mock_ogx_client(self):
        """Create a mock OgxClient."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs
        return mock_client

    def test_get_vector_store_ogx_milvus_default(self, mock_embedding_model, mock_ogx_client):
        """Test getting ogx vector store with milvus provider."""
        vector_store = get_vector_store(
            vs_type="ogx",
            embedding_model=mock_embedding_model,
            client=mock_ogx_client,
            ogx_vector_io_provider_id="milvus",
        )

        assert isinstance(vector_store, OGXVectorStore)
        assert vector_store.embedding_model == mock_embedding_model
        assert vector_store.client == mock_ogx_client
        assert vector_store.distance_metric == "cosine"

    def test_get_vector_store_ogx_with_collection_name(self, mock_embedding_model):
        """Test getting ogx vector store with collection name."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "existing-collection"
        mock_client.vector_stores.retrieve.return_value = mock_vs

        vector_store = get_vector_store(
            vs_type="ogx",
            embedding_model=mock_embedding_model,
            client=mock_client,
            ogx_vector_io_provider_id="milvus",
            reuse_collection_name="existing-collection",
        )

        assert isinstance(vector_store, OGXVectorStore)
        mock_client.vector_stores.retrieve.assert_called_once_with("existing-collection")

    def test_get_vector_store_ogx_passes_provider_id(self, mock_embedding_model, mock_ogx_client):
        """Test that ogx_vector_io_provider_id is passed to OGXVectorStore."""
        vector_store = get_vector_store(
            vs_type="ogx",
            embedding_model=mock_embedding_model,
            client=mock_ogx_client,
            ogx_vector_io_provider_id="milvus",
        )

        call_kwargs = mock_ogx_client.vector_stores.create.call_args.kwargs
        assert call_kwargs["extra_body"]["provider_id"] == "milvus"

    @pytest.mark.parametrize(
        "provider_id",
        ["qdrant", "faiss", "chromadb", "milvus", "milvus-lite"],
    )
    def test_get_vector_store_ogx_various_provider_ids(self, mock_embedding_model, mock_ogx_client, provider_id):
        """Test that various provider IDs are passed through correctly."""
        vector_store = get_vector_store(
            vs_type="ogx",
            embedding_model=mock_embedding_model,
            client=mock_ogx_client,
            ogx_vector_io_provider_id=provider_id,
        )

        assert isinstance(vector_store, OGXVectorStore)
        call_kwargs = mock_ogx_client.vector_stores.create.call_args.kwargs
        assert call_kwargs["extra_body"]["provider_id"] == provider_id

    def test_get_vector_store_ogx_requires_provider_id(self, mock_embedding_model, mock_ogx_client):
        """Test that ogx type without ogx_vector_io_provider_id raises ValueError."""
        with pytest.raises(ValueError, match="ogx_vector_io_provider_id is required"):
            get_vector_store(
                vs_type="ogx",
                embedding_model=mock_embedding_model,
                client=mock_ogx_client,
            )

    def test_get_vector_store_ogx_empty_provider_id_raises_error(self, mock_embedding_model, mock_ogx_client):
        """Test that ogx type with empty ogx_vector_io_provider_id raises ValueError."""
        with pytest.raises(ValueError, match="ogx_vector_io_provider_id is required"):
            get_vector_store(
                vs_type="ogx",
                embedding_model=mock_embedding_model,
                client=mock_ogx_client,
                ogx_vector_io_provider_id="",
            )

    def test_get_vector_store_ogx_requires_client(self, mock_embedding_model):
        """Test that ogx type requires a client parameter."""
        with pytest.raises(AttributeError):
            get_vector_store(
                vs_type="ogx",
                embedding_model=mock_embedding_model,
                client=None,
                ogx_vector_io_provider_id="milvus",
            )


class TestGetVectorStoreInvalidType:
    """Test suite for get_vector_store with invalid type."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel()

    def test_get_vector_store_invalid_type_raises_error(self, mock_embedding_model):
        """Test that invalid vs_type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_vector_store(
                vs_type="invalid_type",
                embedding_model=mock_embedding_model,
            )

        assert "not supported" in str(exc_info.value)
        assert "invalid_type" in str(exc_info.value)

    def test_get_vector_store_empty_type_raises_error(self, mock_embedding_model):
        """Test that empty vs_type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_vector_store(
                vs_type="",
                embedding_model=mock_embedding_model,
            )

        assert "not supported" in str(exc_info.value)

    def test_get_vector_store_none_type_raises_error(self, mock_embedding_model):
        """Test that None vs_type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_vector_store(
                vs_type=None,
                embedding_model=mock_embedding_model,
            )

        assert "not supported" in str(exc_info.value)

    def test_get_vector_store_old_prefix_format_raises_error(self, mock_embedding_model):
        """Test that the old ogx_ prefix format is no longer supported."""
        with pytest.raises(ValueError, match="not supported"):
            get_vector_store(
                vs_type="ogx_milvus",
                embedding_model=mock_embedding_model,
            )


class TestGetVectorStoreEdgeCases:
    """Test suite for edge cases in get_vector_store."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel()

    def test_get_vector_store_case_sensitive(self, mocker, mock_embedding_model):
        """Test that vs_type is case-sensitive."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())

        # "chroma" should work
        vector_store = get_vector_store(
            vs_type="chroma",
            embedding_model=mock_embedding_model,
        )
        assert isinstance(vector_store, ChromaVectorStore)

        # "CHROMA" should not work
        with pytest.raises(ValueError):
            get_vector_store(
                vs_type="CHROMA",
                embedding_model=mock_embedding_model,
            )

    def test_get_vector_store_whitespace_type(self, mock_embedding_model):
        """Test that vs_type with whitespace raises error."""
        with pytest.raises(ValueError):
            get_vector_store(
                vs_type=" chroma ",
                embedding_model=mock_embedding_model,
            )

    def test_get_vector_store_similar_type_names(self, mock_embedding_model):
        """Test that similar but incorrect type names raise errors."""
        invalid_types = ["chromadb", "chroma_db", "milvus", "ogx_milvus"]

        for invalid_type in invalid_types:
            with pytest.raises(ValueError):
                get_vector_store(
                    vs_type=invalid_type,
                    embedding_model=mock_embedding_model,
                )


class TestGetVectorStoreReturnTypes:
    """Test suite for verifying return types from get_vector_store."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel()

    def test_chroma_returns_base_vector_store_interface(self, mocker, mock_embedding_model):
        """Test that Chroma vector store implements BaseVectorStore interface."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())

        vector_store = get_vector_store(
            vs_type="chroma",
            embedding_model=mock_embedding_model,
        )

        # Check that it has required methods
        assert hasattr(vector_store, "search")
        assert hasattr(vector_store, "add_documents")
        assert hasattr(vector_store, "collection_name")
        assert callable(vector_store.search)
        assert callable(vector_store.add_documents)

    def test_ogx_returns_base_vector_store_interface(self, mock_embedding_model):
        """Test that ogx vector store implements BaseVectorStore interface."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-id"
        mock_client.vector_stores.create.return_value = mock_vs

        vector_store = get_vector_store(
            vs_type="ogx",
            embedding_model=mock_embedding_model,
            client=mock_client,
            ogx_vector_io_provider_id="milvus",
        )

        # Check that it has required methods
        assert hasattr(vector_store, "search")
        assert hasattr(vector_store, "add_documents")
        assert hasattr(vector_store, "collection_name")
        assert callable(vector_store.search)
        assert callable(vector_store.add_documents)


class TestGetVectorStoreWithNoneParameters:
    """Test suite for get_vector_store with None parameters."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel()

    def test_get_vector_store_chroma_with_none_collection_name(self, mocker, mock_embedding_model):
        """Test Chroma with None collection name uses default."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())

        vector_store = get_vector_store(
            vs_type="chroma",
            embedding_model=mock_embedding_model,
            reuse_collection_name=None,
        )

        assert isinstance(vector_store, ChromaVectorStore)
        # Should use auto-generated collection name
        assert vector_store.collection_name is not None

    def test_get_vector_store_ogx_with_none_collection_name(self, mock_embedding_model):
        """Test ogx type with None collection name creates new store."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "new-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs

        vector_store = get_vector_store(
            vs_type="ogx",
            embedding_model=mock_embedding_model,
            client=mock_client,
            ogx_vector_io_provider_id="milvus",
            reuse_collection_name=None,
        )

        # Should create new vector store
        mock_client.vector_stores.create.assert_called_once()
        mock_client.vector_stores.retrieve.assert_not_called()
