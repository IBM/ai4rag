# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest
from unittest.mock import MagicMock, patch

from ai4rag.rag.vector_store.get_vector_store import get_vector_store
from ai4rag.rag.vector_store.chroma import ChromaVectorStore
from ai4rag.rag.vector_store.llama_stack import LSVectorStore
from ai4rag.rag.embedding.base_model import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """Mock EmbeddingModel for testing."""

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


class TestGetVectorStoreLSMilvus:
    """Test suite for get_vector_store with ls_milvus type."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel()

    @pytest.fixture
    def mock_llama_stack_client(self):
        """Create a mock LlamaStackClient."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs
        return mock_client

    def test_get_vector_store_ls_milvus_default(self, mock_embedding_model, mock_llama_stack_client):
        """Test getting ls_milvus vector store with default parameters."""
        vector_store = get_vector_store(
            vs_type="ls_milvus",
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
        )

        assert isinstance(vector_store, LSVectorStore)
        assert vector_store.embedding_model == mock_embedding_model
        assert vector_store.client == mock_llama_stack_client
        assert vector_store.distance_metric == "cosine"

    def test_get_vector_store_ls_milvus_with_collection_name(self, mock_embedding_model):
        """Test getting ls_milvus vector store with collection name."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "existing-collection"
        mock_client.vector_stores.retrieve.return_value = mock_vs

        vector_store = get_vector_store(
            vs_type="ls_milvus",
            embedding_model=mock_embedding_model,
            client=mock_client,
            reuse_collection_name="existing-collection",
        )

        assert isinstance(vector_store, LSVectorStore)
        mock_client.vector_stores.retrieve.assert_called_once_with("existing-collection")

    def test_get_vector_store_ls_milvus_uses_milvus_provider(self, mock_embedding_model, mock_llama_stack_client):
        """Test that ls_milvus uses milvus provider_id."""
        vector_store = get_vector_store(
            vs_type="ls_milvus",
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
        )

        call_kwargs = mock_llama_stack_client.vector_stores.create.call_args.kwargs
        assert call_kwargs["extra_body"]["provider_id"] == "milvus"

    def test_get_vector_store_ls_milvus_requires_client(self, mock_embedding_model):
        """Test that ls_milvus requires a client parameter."""
        # This should raise an error or handle None client gracefully
        with pytest.raises(AttributeError):
            vector_store = get_vector_store(
                vs_type="ls_milvus",
                embedding_model=mock_embedding_model,
                client=None,
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
        invalid_types = ["chromadb", "chroma_db", "milvus", "ls_milvus_"]

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

    def test_ls_milvus_returns_base_vector_store_interface(self, mock_embedding_model):
        """Test that ls_milvus vector store implements BaseVectorStore interface."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-id"
        mock_client.vector_stores.create.return_value = mock_vs

        vector_store = get_vector_store(
            vs_type="ls_milvus",
            embedding_model=mock_embedding_model,
            client=mock_client,
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

    def test_get_vector_store_ls_milvus_with_none_collection_name(self, mock_embedding_model):
        """Test ls_milvus with None collection name creates new store."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "new-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs

        vector_store = get_vector_store(
            vs_type="ls_milvus",
            embedding_model=mock_embedding_model,
            client=mock_client,
            reuse_collection_name=None,
        )

        # Should create new vector store
        mock_client.vector_stores.create.assert_called_once()
        mock_client.vector_stores.retrieve.assert_not_called()
