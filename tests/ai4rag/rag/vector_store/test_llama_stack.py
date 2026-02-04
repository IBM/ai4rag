# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest
from unittest.mock import MagicMock, Mock
from langchain_core.documents import Document

from ai4rag.rag.vector_store.llama_stack import LSVectorStore
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel


class MockLSEmbeddingModel(LSEmbeddingModel):
    """Mock LSEmbeddingModel for testing."""

    def __init__(self):
        self.client = MagicMock()
        self.model_id = "test-embedding-model"
        self.params = {"embedding_dimension": 128}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mock embed_documents implementation."""
        return [[0.1, 0.2, 0.3] * 43 for _ in texts]  # 129 floats (close to 128)

    def embed_query(self, query: str) -> list[float]:
        """Mock embed_query implementation."""
        return [0.1, 0.2, 0.3] * 43


class TestLSVectorStoreInitialization:
    """Test suite for LSVectorStore initialization."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockLSEmbeddingModel()

    @pytest.fixture
    def mock_llama_stack_client(self):
        """Create a mock LlamaStackClient."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-vector-store-id"
        mock_client.vector_stores.create.return_value = mock_vs
        mock_client.vector_stores.retrieve.return_value = mock_vs
        return mock_client

    def test_init_creates_new_vector_store(self, mock_embedding_model, mock_llama_stack_client):
        """Test initialization creates a new vector store."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        assert vector_store.embedding_model == mock_embedding_model
        assert vector_store.client == mock_llama_stack_client
        assert vector_store._ls_vs is not None
        mock_llama_stack_client.vector_stores.create.assert_called_once()

    def test_init_with_reuse_collection_name(self, mock_embedding_model, mock_llama_stack_client):
        """Test initialization with reuse_collection_name retrieves existing store."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
            reuse_collection_name="existing-collection",
        )

        mock_llama_stack_client.vector_stores.retrieve.assert_called_once_with("existing-collection")
        mock_llama_stack_client.vector_stores.create.assert_not_called()

    def test_init_with_distance_metric(self, mock_embedding_model, mock_llama_stack_client):
        """Test initialization with custom distance metric."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
            distance_metric="cosine",
        )

        assert vector_store.distance_metric == "cosine"

    def test_init_passes_provider_id(self, mock_embedding_model, mock_llama_stack_client):
        """Test that provider_id is passed correctly during initialization."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="test-provider",
        )

        call_kwargs = mock_llama_stack_client.vector_stores.create.call_args.kwargs
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["provider_id"] == "test-provider"

    def test_init_passes_embedding_model_params(self, mock_embedding_model, mock_llama_stack_client):
        """Test that embedding model parameters are passed correctly."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        call_kwargs = mock_llama_stack_client.vector_stores.create.call_args.kwargs
        extra_body = call_kwargs["extra_body"]
        assert extra_body["embedding_model"] == "test-embedding-model"
        assert extra_body["embedding_dimension"] == 128


class TestLSVectorStoreCollectionName:
    """Test suite for LSVectorStore collection_name property."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockLSEmbeddingModel()

    @pytest.fixture
    def mock_llama_stack_client(self):
        """Create a mock LlamaStackClient."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-collection-id"
        mock_client.vector_stores.create.return_value = mock_vs
        return mock_client

    def test_collection_name_getter(self, mock_embedding_model, mock_llama_stack_client):
        """Test collection_name property getter."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        # Initially None
        assert vector_store.collection_name is None


class TestLSVectorStoreSearch:
    """Test suite for LSVectorStore.search method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockLSEmbeddingModel()

    @pytest.fixture
    def mock_llama_stack_client(self):
        """Create a mock LlamaStackClient with search results."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs

        # Mock search response
        mock_chunk = MagicMock()
        mock_chunk.content = "Test content"
        mock_chunk.chunk_metadata.to_dict.return_value = {"doc_id": "doc1", "seq": 1}

        mock_response = MagicMock()
        mock_response.chunks = [mock_chunk]
        mock_response.scores = [0.95]

        mock_client.vector_io.query.return_value = mock_response
        return mock_client

    def test_search_without_scores(self, mock_embedding_model, mock_llama_stack_client):
        """Test search without scores."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        result = vector_store.search("test query", k=5, include_scores=False)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].page_content == "Test content"
        assert result[0].metadata == {"doc_id": "doc1", "seq": 1}

    def test_search_with_scores(self, mock_embedding_model, mock_llama_stack_client):
        """Test search with scores."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        result = vector_store.search("test query", k=5, include_scores=True)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        assert isinstance(result[0][0], Document)
        assert isinstance(result[0][1], float)
        assert result[0][1] == 0.95

    def test_search_calls_client_with_correct_params(self, mock_embedding_model, mock_llama_stack_client):
        """Test that search calls client with correct parameters."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        vector_store.search("test query", k=10)

        call_kwargs = mock_llama_stack_client.vector_io.query.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["vector_store_id"] == "test-vs-id"
        assert call_kwargs["params"]["max_chunks"] == 10
        assert call_kwargs["params"]["mode"] == "vector"

    def test_search_with_multiple_results(self, mock_embedding_model):
        """Test search with multiple results."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs

        # Create multiple chunks
        chunks = []
        scores = []
        for i in range(3):
            mock_chunk = MagicMock()
            mock_chunk.content = f"Content {i}"
            mock_chunk.chunk_metadata.to_dict.return_value = {"id": i}
            chunks.append(mock_chunk)
            scores.append(0.9 - i * 0.1)

        mock_response = MagicMock()
        mock_response.chunks = chunks
        mock_response.scores = scores
        mock_client.vector_io.query.return_value = mock_response

        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_client,
            provider_id="milvus",
        )

        result = vector_store.search("query", k=3, include_scores=True)

        assert len(result) == 3
        for i, (doc, score) in enumerate(result):
            assert doc.page_content == f"Content {i}"
            assert score == 0.9 - i * 0.1


class TestLSVectorStoreAddDocuments:
    """Test suite for LSVectorStore.add_documents method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockLSEmbeddingModel()

    @pytest.fixture
    def mock_llama_stack_client(self):
        """Create a mock LlamaStackClient."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs
        return mock_client

    def test_add_documents_basic(self, mock_embedding_model, mock_llama_stack_client):
        """Test adding documents to vector store."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        docs = [
            Document(page_content="Doc 1", metadata={"document_id": "doc1"}),
            Document(page_content="Doc 2", metadata={"document_id": "doc2"}),
        ]

        vector_store.add_documents(docs)

        mock_llama_stack_client.vector_io.insert.assert_called_once()

    def test_add_documents_creates_chunks_with_embeddings(self, mock_embedding_model, mock_llama_stack_client):
        """Test that add_documents creates chunks with embeddings."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        docs = [Document(page_content="Test", metadata={"document_id": "doc1"})]

        vector_store.add_documents(docs)

        call_kwargs = mock_llama_stack_client.vector_io.insert.call_args.kwargs
        chunks = call_kwargs["chunks"]

        assert len(chunks) == 1
        assert "content" in chunks[0]
        assert "embedding" in chunks[0]
        assert "chunk_id" in chunks[0]
        assert chunks[0]["chunk_id"] == "doc1"

    def test_add_documents_multiple(self, mock_embedding_model, mock_llama_stack_client):
        """Test adding multiple documents."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        docs = [
            Document(page_content=f"Doc {i}", metadata={"document_id": f"doc{i}"}) for i in range(5)
        ]

        vector_store.add_documents(docs)

        call_kwargs = mock_llama_stack_client.vector_io.insert.call_args.kwargs
        chunks = call_kwargs["chunks"]

        assert len(chunks) == 5

    def test_add_documents_preserves_metadata(self, mock_embedding_model, mock_llama_stack_client):
        """Test that add_documents preserves metadata."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        docs = [
            Document(
                page_content="Test",
                metadata={"document_id": "doc1", "custom_field": "value"},
            )
        ]

        vector_store.add_documents(docs)

        call_kwargs = mock_llama_stack_client.vector_io.insert.call_args.kwargs
        chunks = call_kwargs["chunks"]

        assert chunks[0]["chunk_metadata"]["custom_field"] == "value"

    def test_add_documents_uses_embedding_model(self, mock_llama_stack_client):
        """Test that add_documents uses the embedding model."""
        mock_embed_model = MockLSEmbeddingModel()
        mock_embed_model.embed_documents = MagicMock(return_value=[[0.1, 0.2]])

        vector_store = LSVectorStore(
            embedding_model=mock_embed_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        docs = [Document(page_content="Test", metadata={"document_id": "doc1"})]

        vector_store.add_documents(docs)

        mock_embed_model.embed_documents.assert_called_once_with(["Test"])


class TestLSVectorStoreCleanCollection:
    """Test suite for LSVectorStore.clean_collection method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockLSEmbeddingModel()

    @pytest.fixture
    def mock_llama_stack_client(self):
        """Create a mock LlamaStackClient."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "test-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs
        return mock_client

    def test_clean_collection(self, mock_embedding_model, mock_llama_stack_client):
        """Test cleaning collection."""
        vector_store = LSVectorStore(
            embedding_model=mock_embedding_model,
            client=mock_llama_stack_client,
            provider_id="milvus",
        )

        vector_store.clean_collection()

        mock_llama_stack_client.vector_stores.delete.assert_called_once_with("test-vs-id")


class TestLSVectorStoreInitializeVectorStore:
    """Test suite for LSVectorStore._initialize_ls_vector_store static method."""

    def test_initialize_creates_new_store(self):
        """Test _initialize_ls_vector_store creates new store."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "new-vs-id"
        mock_client.vector_stores.create.return_value = mock_vs

        mock_embedding_model = MockLSEmbeddingModel()

        result = LSVectorStore._initialize_ls_vector_store(
            client=mock_client,
            embedding_model=mock_embedding_model,
            provider_id="test-provider",
            reuse_collection_name=None,
        )

        assert result == mock_vs
        mock_client.vector_stores.create.assert_called_once()

    def test_initialize_reuses_existing_store(self):
        """Test _initialize_ls_vector_store reuses existing store."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_vs.id = "existing-vs-id"
        mock_client.vector_stores.retrieve.return_value = mock_vs

        mock_embedding_model = MockLSEmbeddingModel()

        result = LSVectorStore._initialize_ls_vector_store(
            client=mock_client,
            embedding_model=mock_embedding_model,
            provider_id="test-provider",
            reuse_collection_name="existing-collection",
        )

        assert result == mock_vs
        mock_client.vector_stores.retrieve.assert_called_once_with("existing-collection")
        mock_client.vector_stores.create.assert_not_called()

    def test_initialize_includes_embedding_params(self):
        """Test that initialization includes embedding parameters."""
        mock_client = MagicMock()
        mock_vs = MagicMock()
        mock_client.vector_stores.create.return_value = mock_vs

        mock_embedding_model = MockLSEmbeddingModel()
        mock_embedding_model.model_id = "custom-model-id"
        mock_embedding_model.params = {"embedding_dimension": 256}

        LSVectorStore._initialize_ls_vector_store(
            client=mock_client,
            embedding_model=mock_embedding_model,
            provider_id="provider1",
            reuse_collection_name=None,
        )

        call_kwargs = mock_client.vector_stores.create.call_args.kwargs
        extra_body = call_kwargs["extra_body"]

        assert extra_body["embedding_model"] == "custom-model-id"
        assert extra_body["embedding_dimension"] == 256
        assert extra_body["provider_id"] == "provider1"
