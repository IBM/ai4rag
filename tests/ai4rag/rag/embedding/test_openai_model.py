# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel


class TestOpenAIEmbeddingModel:
    """Test suite for OpenAIEmbeddingModel class."""

    @pytest.fixture
    def mock_openai_client(self, mocker):
        """Create a mock OpenAI client."""
        mock_client = mocker.MagicMock()
        return mock_client

    @pytest.fixture
    def mock_embedding_response(self, mocker):
        """Create a mock embedding response."""
        mock_data = mocker.MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        return mock_data

    @pytest.fixture
    def model_with_params(self, mock_openai_client):
        """Create an OpenAIEmbeddingModel with parameters."""
        return OpenAIEmbeddingModel(
            client=mock_openai_client,
            model_id="text-embedding-ada-002",
            params={"dimensions": 1536},
        )

    @pytest.fixture
    def model_without_params(self, mock_openai_client):
        """Create an OpenAIEmbeddingModel without parameters."""
        return OpenAIEmbeddingModel(
            client=mock_openai_client,
            model_id="text-embedding-3-small",
            params=None,
        )

    def test_init_with_params(self, model_with_params, mock_openai_client):
        """Test initialization with parameters."""
        assert model_with_params.model_id == "text-embedding-ada-002"
        assert model_with_params.params == {"dimensions": 1536}
        assert model_with_params.client == mock_openai_client

    def test_init_without_params(self, model_without_params, mock_openai_client):
        """Test initialization without parameters."""
        assert model_without_params.model_id == "text-embedding-3-small"
        assert model_without_params.params is None
        assert model_without_params.client == mock_openai_client

    def test_embed_documents(self, model_with_params, mock_openai_client, mocker):
        """Test embed_documents method."""
        # Setup mock response
        mock_response = mocker.MagicMock()
        mock_data_1 = mocker.MagicMock()
        mock_data_1.embedding = [0.1, 0.2, 0.3]
        mock_data_2 = mocker.MagicMock()
        mock_data_2.embedding = [0.4, 0.5, 0.6]
        mock_data_3 = mocker.MagicMock()
        mock_data_3.embedding = [0.7, 0.8, 0.9]
        mock_response.data = [mock_data_1, mock_data_2, mock_data_3]
        mock_openai_client.embeddings.create.return_value = mock_response

        texts = ["first document", "second document", "third document"]
        embeddings = model_with_params.embed_documents(texts)

        # Verify client was called correctly
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=texts,
        )

        # Verify returned embeddings
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert embeddings[2] == [0.7, 0.8, 0.9]

    def test_embed_documents_empty_list(self, model_with_params, mock_openai_client, mocker):
        """Test embed_documents with empty list."""
        mock_response = mocker.MagicMock()
        mock_response.data = []
        mock_openai_client.embeddings.create.return_value = mock_response

        embeddings = model_with_params.embed_documents([])

        assert embeddings == []
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=[],
        )

    def test_embed_documents_single_text(self, model_with_params, mock_openai_client, mocker):
        """Test embed_documents with single text."""
        mock_response = mocker.MagicMock()
        mock_data = mocker.MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3, 0.4]
        mock_response.data = [mock_data]
        mock_openai_client.embeddings.create.return_value = mock_response

        embeddings = model_with_params.embed_documents(["single document"])

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4]

    def test_embed_query(self, model_with_params, mock_openai_client, mocker):
        """Test embed_query method."""
        mock_response = mocker.MagicMock()
        mock_data = mocker.MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response.data = [mock_data]
        mock_openai_client.embeddings.create.return_value = mock_response

        query = "What is machine learning?"
        embedding = model_with_params.embed_query(query)

        # Verify client was called correctly
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=query,
        )

        # Verify returned embedding
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_embed_query_empty_string(self, model_with_params, mock_openai_client, mocker):
        """Test embed_query with empty string."""
        mock_response = mocker.MagicMock()
        mock_data = mocker.MagicMock()
        mock_data.embedding = []
        mock_response.data = [mock_data]
        mock_openai_client.embeddings.create.return_value = mock_response

        embedding = model_with_params.embed_query("")

        assert embedding == []

    def test_model_inherits_from_base(self, model_with_params):
        """Test that OpenAIEmbeddingModel inherits BaseEmbeddingModel methods."""
        # Test __repr__
        assert repr(model_with_params) == "text-embedding-ada-002"

        # Test __str__
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
        mock_response = mocker.MagicMock()
        mock_data_1 = mocker.MagicMock()
        mock_data_1.embedding = [1.0, 1.0, 1.0]
        mock_data_2 = mocker.MagicMock()
        mock_data_2.embedding = [2.0, 2.0, 2.0]
        mock_response.data = [mock_data_1, mock_data_2]
        mock_openai_client.embeddings.create.return_value = mock_response

        texts = ["first", "second"]
        embeddings = model_without_params.embed_documents(texts)

        assert embeddings[0] == [1.0, 1.0, 1.0]
        assert embeddings[1] == [2.0, 2.0, 2.0]

    def test_multiple_calls_to_embed_query(self, model_with_params, mock_openai_client, mocker):
        """Test multiple sequential calls to embed_query."""
        mock_response_1 = mocker.MagicMock()
        mock_data_1 = mocker.MagicMock()
        mock_data_1.embedding = [0.1, 0.2]
        mock_response_1.data = [mock_data_1]

        mock_response_2 = mocker.MagicMock()
        mock_data_2 = mocker.MagicMock()
        mock_data_2.embedding = [0.3, 0.4]
        mock_response_2.data = [mock_data_2]

        mock_openai_client.embeddings.create.side_effect = [mock_response_1, mock_response_2]

        embedding_1 = model_with_params.embed_query("query 1")
        embedding_2 = model_with_params.embed_query("query 2")

        assert embedding_1 == [0.1, 0.2]
        assert embedding_2 == [0.3, 0.4]
        assert mock_openai_client.embeddings.create.call_count == 2
