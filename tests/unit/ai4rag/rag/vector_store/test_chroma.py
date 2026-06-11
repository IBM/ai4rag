# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.chroma import ChromaVectorStore


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock BaseEmbeddingModel for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mock embed_documents implementation."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        """Mock embed_query implementation."""
        return [0.1, 0.2, 0.3]


class TestChromaVectorStoreInitialization:
    """Test suite for ChromaVectorStore initialization."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    @pytest.fixture
    def mock_chroma_client(self, mocker):
        """Create a mock Chroma client."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": []}
        mock_client.add_documents.return_value = ["id1", "id2"]
        mock_client.similarity_search.return_value = [Document(page_content="test")]
        mock_client.similarity_search_with_score.return_value = [(Document(page_content="test"), 0.95)]
        return mock_client

    def test_init_with_defaults(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test initialization with default parameters."""
        mock_chroma_class = mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        assert vector_store.embedding_model == mock_embedding_model
        assert vector_store.collection_name.startswith("ai4rag_")
        assert vector_store.distance_metric == "cosine"
        assert vector_store._document_name_field == "document_id"
        assert vector_store._chunk_sequence_number_field == "sequence_number"

    def test_init_with_custom_parameters(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test initialization with custom parameters."""
        mock_chroma_class = mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(
            embedding_model=mock_embedding_model,
            reuse_collection_name="custom_collection",
            distance_metric="l2",
            document_name_field="doc_id",
            chunk_sequence_number_field="seq_num",
        )
        assert vector_store.collection_name == "custom_collection"
        assert vector_store.distance_metric == "l2"
        assert vector_store._document_name_field == "doc_id"
        assert vector_store._chunk_sequence_number_field == "seq_num"

    def test_init_creates_chroma_client(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test that Chroma client is created during initialization."""
        mock_chroma_class = mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        mock_chroma_class.assert_called_once()
        call_kwargs = mock_chroma_class.call_args.kwargs
        assert call_kwargs["collection_name"].startswith("ai4rag_")
        assert call_kwargs["embedding_function"] == mock_embedding_model
        assert call_kwargs["collection_metadata"]["hnsw:space"] == "cosine"

    def test_init_passes_kwargs_to_chroma(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test that additional kwargs are passed to Chroma client."""
        mock_chroma_class = mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model, persist_directory="/tmp/test")
        call_kwargs = mock_chroma_class.call_args.kwargs
        assert "persist_directory" in call_kwargs
        assert call_kwargs["persist_directory"] == "/tmp/test"


class TestChromaVectorStoreDistanceMetric:
    """Test suite for ChromaVectorStore distance_metric property."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_distance_metric_getter(self, mocker, mock_embedding_model):
        """Test distance_metric property getter."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model, distance_metric="l2")
        assert vector_store.distance_metric == "l2"

    def test_distance_metric_setter_valid(self, mocker, mock_embedding_model):
        """Test distance_metric setter with valid value."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model, distance_metric="cosine")
        vector_store.distance_metric = "l2"
        assert vector_store.distance_metric == "l2"

    def test_distance_metric_setter_invalid(self, mocker, mock_embedding_model):
        """Test distance_metric setter with invalid value raises ValueError."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        with pytest.raises(ValueError) as exc_info:
            vector_store.distance_metric = "invalid_metric"
        assert "Invalid distance metric" in str(exc_info.value)
        assert "cosine" in str(exc_info.value) or "l2" in str(exc_info.value)


class TestChromaVectorStoreToLangchainDocuments:
    """Test suite for ChromaVectorStore._to_langchain_documents method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_to_langchain_documents_basic(self, mocker, mock_embedding_model):
        """Test _to_langchain_documents converts AI4RAGChunks to Documents."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [
            AI4RAGChunk(text="First document"),
            AI4RAGChunk(text="Second document"),
            AI4RAGChunk(text="Third document"),
        ]
        result = vector_store._to_langchain_documents(chunks)
        assert len(result) == 3
        assert all(isinstance(doc, Document) for doc in result)
        assert result[0].page_content == "First document"
        assert result[1].page_content == "Second document"
        assert result[2].page_content == "Third document"

    def test_to_langchain_documents_with_metadata(self, mocker, mock_embedding_model):
        """Test _to_langchain_documents preserves metadata."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [
            AI4RAGChunk(text="First doc", metadata={"id": 1}),
            AI4RAGChunk(text="Second doc", metadata={"id": 2}),
        ]
        result = vector_store._to_langchain_documents(chunks)
        assert len(result) == 2
        assert result[0].page_content == "First doc"
        assert result[0].metadata == {"id": 1}
        assert result[1].page_content == "Second doc"
        assert result[1].metadata == {"id": 2}

    def test_to_langchain_documents_empty_list(self, mocker, mock_embedding_model):
        """Test _to_langchain_documents with empty list."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store._to_langchain_documents([])
        assert len(result) == 0


class TestChromaVectorStoreProcessDocuments:
    """Test suite for ChromaVectorStore._process_documents method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_process_documents_basic(self, mocker, mock_embedding_model):
        """Test _process_documents with basic AI4RAGChunks."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [AI4RAGChunk(text="Doc 1"), AI4RAGChunk(text="Doc 2")]
        ids, docs = vector_store._process_documents(chunks)
        assert len(ids) == len(docs)
        assert len(ids) == 2
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(isinstance(doc_id, str) for doc_id in ids)

    def test_process_documents_removes_duplicates(self, mocker, mock_embedding_model):
        """Test _process_documents removes duplicate chunks."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [AI4RAGChunk(text="Same doc"), AI4RAGChunk(text="Same doc"), AI4RAGChunk(text="Different doc")]
        ids, docs = vector_store._process_documents(chunks)
        # Should have 2 unique documents
        assert len(ids) == 2
        assert len(docs) == 2
        assert len(set(ids)) == 2  # All IDs should be unique

    def test_process_documents_empty_list(self, mocker, mock_embedding_model):
        """Test _process_documents with empty list."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        ids, docs = vector_store._process_documents([])
        assert ids == []
        assert docs == []


class TestChromaVectorStoreAddDocuments:
    """Test suite for ChromaVectorStore.add_documents method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock Chroma client."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": []}
        mock_client.add_documents.return_value = ["id1", "id2"]
        return mock_client

    def test_add_documents_basic(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test add_documents with AI4RAGChunks."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [AI4RAGChunk(text="Doc 1"), AI4RAGChunk(text="Doc 2")]
        result = vector_store.add_documents(chunks)
        assert isinstance(result, list)
        assert len(result) == 2
        mock_chroma_client.add_documents.assert_called_once()

    def test_add_documents_with_metadata(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test add_documents with AI4RAGChunks carrying metadata."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [
            AI4RAGChunk(text="Doc 1", metadata={"source": "a"}),
            AI4RAGChunk(text="Doc 2", metadata={"source": "b"}),
        ]
        result = vector_store.add_documents(chunks)
        assert isinstance(result, list)
        mock_chroma_client.add_documents.assert_called_once()

    def test_add_documents_with_batching(self, mocker, mock_embedding_model):
        """Test add_documents with batching when exceeding max_batch_size."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": []}
        mock_client.add_documents.return_value = ["id1", "id2"]
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [AI4RAGChunk(text=f"Doc {i}") for i in range(5)]
        result = vector_store.add_documents(chunks, max_batch_size=2)
        # Should be called multiple times due to batching
        assert mock_client.add_documents.call_count >= 2
        assert isinstance(result, list)

    def test_add_documents_with_custom_max_batch_size(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test add_documents with custom max_batch_size."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [AI4RAGChunk(text=f"Doc {i}") for i in range(3)]
        result = vector_store.add_documents(chunks, max_batch_size=2)
        # Should be called multiple times due to custom batch size
        assert mock_chroma_client.add_documents.call_count >= 2
        assert isinstance(result, list)

    def test_add_documents_with_default_max_batch_size(self, mocker, mock_embedding_model):
        """Test add_documents uses default max_batch_size of 2048."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": []}
        mock_client.add_documents.return_value = ["id1"]
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunks = [AI4RAGChunk(text="Doc 1")]
        result = vector_store.add_documents(chunks)
        # Should use default max_batch_size of 2048
        assert isinstance(result, list)
        mock_client.add_documents.assert_called_once()


class TestChromaVectorStoreSearch:
    """Test suite for ChromaVectorStore.search method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock Chroma client."""
        mock_client = MagicMock()
        mock_doc = Document(page_content="Test document", metadata={"id": 1})
        mock_client.similarity_search.return_value = [mock_doc]
        mock_client.similarity_search_with_score.return_value = [(mock_doc, 0.95)]
        return mock_client

    def test_search_basic(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test basic search without scores returns AI4RAGChunks."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store.search("test query", k=5)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], AI4RAGChunk)
        assert result[0].text == "Test document"
        mock_chroma_client.similarity_search.assert_called_once_with("test query", k=5)

    def test_search_with_scores(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test search with scores returns AI4RAGChunk tuples."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store.search("test query", k=5, include_scores=True)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        assert isinstance(result[0][0], AI4RAGChunk)
        assert result[0][0].text == "Test document"
        assert isinstance(result[0][1], float)
        mock_chroma_client.similarity_search_with_score.assert_called_once_with("test query", k=5)

    def test_search_with_kwargs(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test search with additional kwargs."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store.search("test query", k=3, filter={"key": "value"})
        mock_chroma_client.similarity_search.assert_called_once_with("test query", k=3, filter={"key": "value"})


class TestChromaVectorStoreWindowSearch:
    """Test suite for ChromaVectorStore.window_search method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock Chroma client."""
        mock_client = MagicMock()
        mock_doc = Document(
            page_content="Test document",
            metadata={"document_id": "doc1", "sequence_number": 5},
        )
        mock_client.similarity_search.return_value = [mock_doc]
        mock_client.similarity_search_with_score.return_value = [(mock_doc, 0.95)]
        return mock_client

    def test_window_search_with_zero_window_size(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test window_search with zero window_size returns original results."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store.window_search("test query", k=5, window_size=0)
        assert isinstance(result, list)
        assert len(result) > 0
        # Should return same as regular search
        mock_chroma_client.similarity_search.assert_called_once()

    def test_window_search_with_negative_window_size(self, mocker, mock_embedding_model, mock_chroma_client):
        """Test window_search with negative window_size returns original results."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_chroma_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store.window_search("test query", k=5, window_size=-1)
        assert isinstance(result, list)
        # Should return same as regular search (window_size <= 0)
        mock_chroma_client.similarity_search.assert_called_once()

    def test_window_search_without_scores(self, mocker, mock_embedding_model):
        """Test window_search without scores returns AI4RAGChunks."""
        mock_client = MagicMock()
        mock_doc = Document(
            page_content="Test document",
            metadata={"document_id": "doc1", "sequence_number": 5},
        )
        mock_client.similarity_search.return_value = [mock_doc]
        mock_client.get.return_value = {
            "documents": ["Window doc 1", "Window doc 2", "Window doc 3"],
            "metadatas": [
                {"sequence_number": 4, "document_id": "doc1"},
                {"sequence_number": 5, "document_id": "doc1"},
                {"sequence_number": 6, "document_id": "doc1"},
            ],
        }
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store.window_search("test query", k=5, window_size=1, include_scores=False)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], AI4RAGChunk)

    def test_window_search_with_scores(self, mocker, mock_embedding_model):
        """Test window_search with scores returns AI4RAGChunk tuples."""
        mock_client = MagicMock()
        mock_doc = Document(
            page_content="Test document",
            metadata={"document_id": "doc1", "sequence_number": 5},
        )
        mock_client.similarity_search_with_score.return_value = [(mock_doc, 0.95)]
        mock_client.get.return_value = {
            "documents": ["Window doc 1", "Window doc 2", "Window doc 3"],
            "metadatas": [
                {"sequence_number": 4, "document_id": "doc1"},
                {"sequence_number": 5, "document_id": "doc1"},
                {"sequence_number": 6, "document_id": "doc1"},
            ],
        }
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store.window_search("test query", k=5, window_size=1, include_scores=True)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], AI4RAGChunk)
        assert isinstance(result[0][1], float)


class TestChromaVectorStoreWindowExtendAndMerge:
    """Test suite for ChromaVectorStore._window_extend_and_merge method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_window_extend_and_merge_missing_document_id(self, mocker, mock_embedding_model):
        """Test _window_extend_and_merge raises ValueError when document_id is missing."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunk = AI4RAGChunk(text="Test", metadata={"sequence_number": 1})
        with pytest.raises(ValueError) as exc_info:
            vector_store._window_extend_and_merge(chunk, window_size=2)
        assert "document_id" in str(exc_info.value)

    def test_window_extend_and_merge_missing_sequence_number(self, mocker, mock_embedding_model):
        """Test _window_extend_and_merge raises ValueError when sequence_number is missing."""
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=MagicMock())
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunk = AI4RAGChunk(text="Test", metadata={"document_id": "doc1"})
        with pytest.raises(ValueError) as exc_info:
            vector_store._window_extend_and_merge(chunk, window_size=2)
        assert "sequence_number" in str(exc_info.value)

    def test_window_extend_and_merge_basic(self, mocker, mock_embedding_model):
        """Test _window_extend_and_merge with basic chunk."""
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "documents": ["Chunk 1", "Chunk 2", "Chunk 3"],
            "metadatas": [
                {"sequence_number": 1, "document_id": "doc1"},
                {"sequence_number": 2, "document_id": "doc1"},
                {"sequence_number": 3, "document_id": "doc1"},
            ],
        }
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        chunk = AI4RAGChunk(
            text="Chunk 2",
            metadata={"document_id": "doc1", "sequence_number": 2},
        )
        result = vector_store._window_extend_and_merge(chunk, window_size=1)
        assert isinstance(result, AI4RAGChunk)
        assert "Chunk" in result.text  # Should contain merged content


class TestChromaVectorStoreGetWindowDocuments:
    """Test suite for ChromaVectorStore._get_window_documents method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_get_window_documents_basic(self, mocker, mock_embedding_model):
        """Test _get_window_documents with basic parameters."""
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "documents": ["Doc 1", "Doc 2", "Doc 3"],
            "metadatas": [
                {"sequence_number": 1, "document_id": "doc1"},
                {"sequence_number": 2, "document_id": "doc1"},
                {"sequence_number": 3, "document_id": "doc1"},
            ],
        }
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        result = vector_store._get_window_documents("doc1", [1, 2, 3])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(doc, Document) for doc in result)
        # Verify the query expression was constructed correctly
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args.kwargs
        assert "where" in call_kwargs


class TestChromaVectorStoreCount:
    """Test suite for ChromaVectorStore.count method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_count_empty(self, mocker, mock_embedding_model):
        """Test count with empty vector store."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": []}
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        assert vector_store.count() == 0

    def test_count_with_documents(self, mocker, mock_embedding_model):
        """Test count with documents in vector store."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": ["id1", "id2", "id3"]}
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        assert vector_store.count() == 3


class TestChromaVectorStoreClear:
    """Test suite for ChromaVectorStore.clear method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_clear_with_documents(self, mocker, mock_embedding_model):
        """Test clear with documents in vector store."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": ["id1", "id2", "id3"]}
        mock_client.delete = MagicMock()
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        vector_store.clear()
        mock_client.delete.assert_called_once_with(["id1", "id2", "id3"])

    def test_clear_empty(self, mocker, mock_embedding_model):
        """Test clear with empty vector store."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"ids": []}
        mock_client.delete = MagicMock()
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        vector_store.clear()
        # Should not call delete if there are no documents
        mock_client.delete.assert_not_called()


class TestChromaVectorStoreDelete:
    """Test suite for ChromaVectorStore.delete method."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(client=MagicMock(), model_id="test-embedding-model")

    def test_delete_basic(self, mocker, mock_embedding_model):
        """Test delete with list of IDs."""
        mock_client = MagicMock()
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        vector_store.delete(["id1", "id2", "id3"])
        mock_client.delete.assert_called_once_with(["id1", "id2", "id3"])

    def test_delete_with_kwargs(self, mocker, mock_embedding_model):
        """Test delete with additional kwargs."""
        mock_client = MagicMock()
        mocker.patch("ai4rag.rag.vector_store.chroma.Chroma", return_value=mock_client)
        vector_store = ChromaVectorStore(embedding_model=mock_embedding_model)
        vector_store.delete(["id1"], filter={"key": "value"})
        mock_client.delete.assert_called_once_with(["id1"], filter={"key": "value"})
