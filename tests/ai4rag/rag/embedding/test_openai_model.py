# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel


def _make_mock_embedding_response(mocker, embeddings):
    """Helper to create a mock OpenAI embedding response."""
    mock_response = mocker.MagicMock()
    mock_response.data = []
    for emb in embeddings:
        mock_data = mocker.MagicMock()
        mock_data.embedding = emb
        mock_response.data.append(mock_data)
    return mock_response


class TestOpenAIEmbeddingModel:
    """Test suite for OpenAIEmbeddingModel class."""

    @pytest.fixture
    def mock_openai_client(self, mocker):
        """Create a mock OpenAI client with default embedding response for auto-detection."""
        mock_client = mocker.MagicMock()
        # Default response for embedding_dimension auto-detection
        mock_client.embeddings.create.return_value = _make_mock_embedding_response(mocker, [[0.1, 0.2, 0.3, 0.4, 0.5]])
        return mock_client

    @pytest.fixture
    def model_with_params(self, mock_openai_client):
        """Create an OpenAIEmbeddingModel with parameters including embedding_dimension and context_length."""
        return OpenAIEmbeddingModel(
            client=mock_openai_client,
            model_id="text-embedding-ada-002",
            params={"dimensions": 1536, "embedding_dimension": 1536, "context_length": 8191},
        )

    @pytest.fixture
    def model_without_params(self, mock_openai_client):
        """Create an OpenAIEmbeddingModel without parameters (auto-detects embedding_dimension and context_length)."""
        return OpenAIEmbeddingModel(
            client=mock_openai_client,
            model_id="text-embedding-3-small",
            params=None,
        )

    def test_init_with_params(self, model_with_params, mock_openai_client):
        """Test initialization with parameters does not trigger auto-detection."""
        assert model_with_params.model_id == "text-embedding-ada-002"
        assert model_with_params.params["dimensions"] == 1536
        assert model_with_params.params["embedding_dimension"] == 1536
        assert model_with_params.params["context_length"] == 8191
        assert model_with_params.client == mock_openai_client
        # No auto-detection call should have been made
        mock_openai_client.embeddings.create.assert_not_called()

    def test_init_without_params(self, model_without_params, mock_openai_client):
        """Test initialization without parameters triggers auto-detection of embedding_dimension and context_length."""
        assert model_without_params.model_id == "text-embedding-3-small"
        assert model_without_params.params["embedding_dimension"] == 5
        assert model_without_params.params["context_length"] == 4096
        assert model_without_params.client == mock_openai_client
        # 1 call for dimension + 1 call for context_length (first probe succeeds)
        assert mock_openai_client.embeddings.create.call_count == 2

    def test_embed_documents(self, model_with_params, mock_openai_client, mocker):
        """Test embed_documents method."""
        mock_openai_client.embeddings.create.return_value = _make_mock_embedding_response(
            mocker, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )

        texts = ["first document", "second document", "third document"]
        embeddings = model_with_params.embed_documents(texts)

        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=texts,
        )

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert embeddings[2] == [0.7, 0.8, 0.9]

    def test_embed_documents_empty_list(self, model_with_params, mock_openai_client, mocker):
        """Test embed_documents with empty list."""
        mock_openai_client.embeddings.create.return_value = _make_mock_embedding_response(mocker, [])

        embeddings = model_with_params.embed_documents([])

        assert embeddings == []
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=[],
        )

    def test_embed_documents_single_text(self, model_with_params, mock_openai_client, mocker):
        """Test embed_documents with single text."""
        mock_openai_client.embeddings.create.return_value = _make_mock_embedding_response(
            mocker, [[0.1, 0.2, 0.3, 0.4]]
        )

        embeddings = model_with_params.embed_documents(["single document"])

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4]

    def test_embed_query(self, model_with_params, mock_openai_client, mocker):
        """Test embed_query method."""
        mock_openai_client.embeddings.create.return_value = _make_mock_embedding_response(
            mocker, [[0.1, 0.2, 0.3, 0.4, 0.5]]
        )

        query = "What is machine learning?"
        embedding = model_with_params.embed_query(query)

        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=query,
        )

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_embed_query_empty_string(self, model_with_params, mock_openai_client, mocker):
        """Test embed_query with empty string."""
        mock_openai_client.embeddings.create.return_value = _make_mock_embedding_response(mocker, [[]])

        embedding = model_with_params.embed_query("")

        assert embedding == []

    def test_model_inherits_from_base(self, model_with_params):
        """Test that OpenAIEmbeddingModel inherits BaseEmbeddingModel methods."""
        assert repr(model_with_params) == "text-embedding-ada-002"
        assert str(model_with_params) == "text-embedding-ada-002"

    @pytest.mark.parametrize(
        "model_id",
        [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
    )
    def test_various_model_ids(self, mock_openai_client, model_id):
        """Test initialization with various OpenAI embedding model IDs."""
        model = OpenAIEmbeddingModel(
            client=mock_openai_client,
            model_id=model_id,
            params=None,
        )
        assert model.model_id == model_id

    def test_embed_documents_preserves_order(self, model_without_params, mock_openai_client, mocker):
        """Test that embed_documents preserves the order of input texts."""
        mock_openai_client.embeddings.create.return_value = _make_mock_embedding_response(
            mocker, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )

        texts = ["first", "second"]
        embeddings = model_without_params.embed_documents(texts)

        assert embeddings[0] == [1.0, 1.0, 1.0]
        assert embeddings[1] == [2.0, 2.0, 2.0]

    def test_multiple_calls_to_embed_query(self, model_with_params, mock_openai_client, mocker):
        """Test multiple sequential calls to embed_query."""
        mock_openai_client.embeddings.create.side_effect = [
            _make_mock_embedding_response(mocker, [[0.1, 0.2]]),
            _make_mock_embedding_response(mocker, [[0.3, 0.4]]),
        ]

        embedding_1 = model_with_params.embed_query("query 1")
        embedding_2 = model_with_params.embed_query("query 2")

        assert embedding_1 == [0.1, 0.2]
        assert embedding_2 == [0.3, 0.4]
        assert mock_openai_client.embeddings.create.call_count == 2

    def test_detect_embedding_dimension_api_failure(self, mocker):
        """Test that _detect_embedding_dimension raises RuntimeError on API failure."""
        mock_client = mocker.MagicMock()
        mock_client.embeddings.create.side_effect = ConnectionError("Service unavailable")

        with pytest.raises(RuntimeError, match="Failed to auto-detect embedding dimension"):
            OpenAIEmbeddingModel(client=mock_client, model_id="text-embedding-ada-002", params=None)

    def test_detect_context_length_succeeds_at_first_probe(self, mock_openai_client, mocker):
        """Test that context_length detection returns 4096 when the first probe succeeds."""
        model = OpenAIEmbeddingModel(
            client=mock_openai_client,
            model_id="text-embedding-3-small",
            params={"embedding_dimension": 1536},
        )

        assert model.params["context_length"] == 4096
        mock_openai_client.embeddings.create.assert_called_once()

    def test_detect_context_length_falls_back_to_smaller_probe(self, mocker):
        """Test that context_length detection falls back to smaller probe sizes."""
        mock_client = mocker.MagicMock()
        response = _make_mock_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        def side_effect(**kwargs):
            text = kwargs.get("input", "")
            if isinstance(text, str) and len(text) > 2048 * 5:
                raise ValueError("Input too long")
            return response

        mock_client.embeddings.create.side_effect = side_effect

        model = OpenAIEmbeddingModel(
            client=mock_client,
            model_id="test-model",
            params={"embedding_dimension": 384},
        )

        assert model.params["context_length"] == 2048

    def test_detect_context_length_all_probes_fail(self, mocker):
        """Test that RuntimeError is raised when all context_length probes fail."""
        mock_client = mocker.MagicMock()
        dim_response = _make_mock_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return dim_response  # dimension detection
            raise ValueError("Input too long")

        mock_client.embeddings.create.side_effect = side_effect

        with pytest.raises(RuntimeError, match="Failed to auto-detect 'context_length'"):
            OpenAIEmbeddingModel(client=mock_client, model_id="test-model", params=None)

    def test_detect_context_length_skipped_when_explicit(self, mock_openai_client):
        """Test that context_length detection is skipped when explicitly provided."""
        model = OpenAIEmbeddingModel(
            client=mock_openai_client,
            model_id="text-embedding-3-small",
            params={"embedding_dimension": 1536, "context_length": 8191},
        )

        assert model.params["context_length"] == 8191
        mock_openai_client.embeddings.create.assert_not_called()
