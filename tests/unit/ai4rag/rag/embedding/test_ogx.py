# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.rag.embedding.ogx import OGXEmbeddingModel, OGXEmbeddingParams


def _make_mock_ogx_embedding_response(mocker, embeddings):
    """Helper to create a mock OGX embedding response."""
    mock_response = mocker.MagicMock()
    mock_response.data = []
    for emb in embeddings:
        mock_data = mocker.MagicMock()
        mock_data.embedding = emb
        mock_response.data.append(mock_data)
    return mock_response


class TestOGXEmbeddingModel:
    """Test suite for OGXEmbeddingModel class."""

    @pytest.fixture
    def mock_ogx_client(self, mocker):
        """Create a mock OGX client with default embedding response for auto-detection."""
        mock_client = mocker.MagicMock()
        mock_client.embeddings.create.return_value = _make_mock_ogx_embedding_response(mocker, [[0.1, 0.2, 0.3, 0.4]])
        return mock_client

    def test_init_with_explicit_dimension(self, mock_ogx_client):
        """Test initialization with explicit embedding_dimension and context_length does not trigger auto-detection."""
        params = OGXEmbeddingParams(embedding_dimension=768, context_length=512)
        model = OGXEmbeddingModel(client=mock_ogx_client, model_id="all-MiniLM-L6-v2", params=params)

        assert model.params.embedding_dimension == 768
        assert model.params.context_length == 512
        mock_ogx_client.embeddings.create.assert_not_called()

    def test_init_with_dict_params_with_dimension(self, mock_ogx_client):
        """Test initialization with dict params containing embedding_dimension and context_length."""
        model = OGXEmbeddingModel(
            client=mock_ogx_client,
            model_id="all-MiniLM-L6-v2",
            params={"embedding_dimension": 384, "context_length": 512},
        )

        assert model.params.embedding_dimension == 384
        assert model.params.context_length == 512
        mock_ogx_client.embeddings.create.assert_not_called()

    def test_init_without_params_auto_detects_dimension(self, mock_ogx_client):
        """Test initialization without params triggers auto-detection of embedding_dimension and context_length."""
        model = OGXEmbeddingModel(client=mock_ogx_client, model_id="all-MiniLM-L6-v2")

        assert model.params.embedding_dimension == 4
        # Binary search converges near upper bound when all probes succeed
        assert model.params.context_length > 0

    def test_init_with_none_params_auto_detects_dimension(self, mock_ogx_client):
        """Test initialization with params=None triggers auto-detection."""
        model = OGXEmbeddingModel(client=mock_ogx_client, model_id="all-MiniLM-L6-v2", params=None)

        assert model.params.embedding_dimension == 4
        assert model.params.context_length > 0

    def test_init_with_params_missing_dimension_auto_detects(self, mock_ogx_client):
        """Test that auto-detection triggers when OGXEmbeddingParams has no dimension."""
        params = OGXEmbeddingParams()  # embedding_dimension and context_length default to None
        model = OGXEmbeddingModel(client=mock_ogx_client, model_id="all-MiniLM-L6-v2", params=params)

        assert model.params.embedding_dimension == 4
        assert model.params.context_length > 0

    def test_init_with_invalid_params_type(self, mock_ogx_client):
        """Test initialization with invalid params type raises TypeError."""
        with pytest.raises(TypeError, match="Incorrect type of 'params' parameter"):
            OGXEmbeddingModel(client=mock_ogx_client, model_id="all-MiniLM-L6-v2", params="invalid")

    def test_detect_embedding_dimension_api_failure(self, mocker):
        """Test that _detect_embedding_dimension raises RuntimeError on API failure."""
        mock_client = mocker.MagicMock()
        mock_client.embeddings.create.side_effect = ConnectionError("Service unavailable")

        with pytest.raises(RuntimeError, match="Failed to auto-detect embedding dimension"):
            OGXEmbeddingModel(client=mock_client, model_id="all-MiniLM-L6-v2", params=None)

    def test_detect_embedding_dimension_preserves_original_exception(self, mocker):
        """Test that the original exception is chained in the RuntimeError."""
        mock_client = mocker.MagicMock()
        original_error = ConnectionError("Connection refused")
        mock_client.embeddings.create.side_effect = original_error

        with pytest.raises(RuntimeError) as exc_info:
            OGXEmbeddingModel(client=mock_client, model_id="test-model", params=None)

        assert exc_info.value.__cause__ is original_error

    def test_detect_context_length_all_probes_succeed(self, mock_ogx_client):
        """Test that context_length detection converges near upper bound when all probes succeed."""
        params = OGXEmbeddingParams(embedding_dimension=384)
        model = OGXEmbeddingModel(client=mock_ogx_client, model_id="all-MiniLM-L6-v2", params=params)

        # Binary search converges near 8192 when all probes succeed
        assert model.params.context_length > 4096

    def test_detect_context_length_binary_search_finds_limit(self, mocker):
        """Test that binary search finds the correct context_length limit."""
        mock_client = mocker.MagicMock()
        response = _make_mock_ogx_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        def side_effect(**kwargs):
            text = kwargs.get("input", "")
            # Each "word " is 5 chars, so N words = N*5 chars
            # Accept up to 2048 words
            if isinstance(text, str) and len(text) > 2048 * 5:
                raise ValueError("Input too long")
            return response

        mock_client.embeddings.create.side_effect = side_effect

        params = OGXEmbeddingParams(embedding_dimension=384)
        model = OGXEmbeddingModel(client=mock_client, model_id="test-model", params=params)

        # Binary search should find a value close to 2048
        assert 1792 <= model.params.context_length <= 2048

    def test_detect_context_length_all_probes_fail(self, mocker):
        """Test that RuntimeError is raised when all context_length probes fail."""
        mock_client = mocker.MagicMock()
        dim_response = _make_mock_ogx_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return dim_response  # dimension detection
            raise ValueError("Input too long")

        mock_client.embeddings.create.side_effect = side_effect

        with pytest.raises(RuntimeError, match="Failed to auto-detect 'context_length'"):
            OGXEmbeddingModel(client=mock_client, model_id="test-model", params=None)

    def test_detect_context_length_skipped_when_explicit(self, mock_ogx_client):
        """Test that context_length detection is skipped when explicitly provided."""
        params = OGXEmbeddingParams(embedding_dimension=384, context_length=1024)
        model = OGXEmbeddingModel(client=mock_ogx_client, model_id="all-MiniLM-L6-v2", params=params)

        assert model.params.context_length == 1024
        mock_ogx_client.embeddings.create.assert_not_called()

    def test_embed_documents(self, mock_ogx_client, mocker):
        """Test embed_documents method."""
        model = OGXEmbeddingModel(
            client=mock_ogx_client,
            model_id="all-MiniLM-L6-v2",
            params=OGXEmbeddingParams(embedding_dimension=3, context_length=512),
        )
        mock_ogx_client.embeddings.create.return_value = _make_mock_ogx_embedding_response(
            mocker, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        embeddings = model.embed_documents(["text1", "text2"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_embed_documents_batches_large_input(self, mock_ogx_client, mocker):
        """Test embed_documents batches inputs exceeding _BATCH_SIZE texts."""
        model = OGXEmbeddingModel(
            client=mock_ogx_client,
            model_id="all-MiniLM-L6-v2",
            params=OGXEmbeddingParams(embedding_dimension=3, context_length=512),
        )

        batch_size = OGXEmbeddingModel._BATCH_SIZE
        remainder = 100
        total = 2 * batch_size + remainder

        batch1_response = _make_mock_ogx_embedding_response(mocker, [[0.1] for _ in range(batch_size)])
        batch2_response = _make_mock_ogx_embedding_response(mocker, [[0.2] for _ in range(batch_size)])
        batch3_response = _make_mock_ogx_embedding_response(mocker, [[0.3] for _ in range(remainder)])
        mock_ogx_client.embeddings.create.side_effect = [batch1_response, batch2_response, batch3_response]

        texts = [f"text{i}" for i in range(total)]
        embeddings = model.embed_documents(texts)

        assert len(embeddings) == total
        assert mock_ogx_client.embeddings.create.call_count == 3

    def test_embed_query(self, mock_ogx_client, mocker):
        """Test embed_query method."""
        model = OGXEmbeddingModel(
            client=mock_ogx_client,
            model_id="all-MiniLM-L6-v2",
            params=OGXEmbeddingParams(embedding_dimension=3, context_length=512),
        )
        mock_ogx_client.embeddings.create.return_value = _make_mock_ogx_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        embedding = model.embed_query("test query")

        assert embedding == [0.1, 0.2, 0.3]

    def test_model_repr(self, mock_ogx_client):
        """Test string representation."""
        model = OGXEmbeddingModel(
            client=mock_ogx_client,
            model_id="all-MiniLM-L6-v2",
            params=OGXEmbeddingParams(embedding_dimension=384, context_length=512),
        )

        assert repr(model) == "all-MiniLM-L6-v2"
        assert str(model) == "all-MiniLM-L6-v2"
