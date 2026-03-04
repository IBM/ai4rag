# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams


def _make_mock_ls_embedding_response(mocker, embeddings):
    """Helper to create a mock Llama Stack embedding response."""
    mock_response = mocker.MagicMock()
    mock_response.data = []
    for emb in embeddings:
        mock_data = mocker.MagicMock()
        mock_data.embedding = emb
        mock_response.data.append(mock_data)
    return mock_response


class TestLSEmbeddingModel:
    """Test suite for LSEmbeddingModel class."""

    @pytest.fixture
    def mock_ls_client(self, mocker):
        """Create a mock Llama Stack client with default embedding response for auto-detection."""
        mock_client = mocker.MagicMock()
        mock_client.embeddings.create.return_value = _make_mock_ls_embedding_response(mocker, [[0.1, 0.2, 0.3, 0.4]])
        return mock_client

    def test_init_with_explicit_dimension(self, mock_ls_client):
        """Test initialization with explicit embedding_dimension and context_length does not trigger auto-detection."""
        params = LSEmbeddingParams(embedding_dimension=768, context_length=512)
        model = LSEmbeddingModel(client=mock_ls_client, model_id="all-MiniLM-L6-v2", params=params)

        assert model.params.embedding_dimension == 768
        assert model.params.context_length == 512
        mock_ls_client.embeddings.create.assert_not_called()

    def test_init_with_dict_params_with_dimension(self, mock_ls_client):
        """Test initialization with dict params containing embedding_dimension and context_length."""
        model = LSEmbeddingModel(
            client=mock_ls_client,
            model_id="all-MiniLM-L6-v2",
            params={"embedding_dimension": 384, "context_length": 512},
        )

        assert model.params.embedding_dimension == 384
        assert model.params.context_length == 512
        mock_ls_client.embeddings.create.assert_not_called()

    def test_init_without_params_auto_detects_dimension(self, mock_ls_client):
        """Test initialization without params triggers auto-detection of embedding_dimension and context_length."""
        model = LSEmbeddingModel(client=mock_ls_client, model_id="all-MiniLM-L6-v2")

        assert model.params.embedding_dimension == 4
        assert model.params.context_length == 4096
        # 1 call for dimension detection + 1 call for context_length detection (first probe succeeds)
        assert mock_ls_client.embeddings.create.call_count == 2

    def test_init_with_none_params_auto_detects_dimension(self, mock_ls_client):
        """Test initialization with params=None triggers auto-detection."""
        model = LSEmbeddingModel(client=mock_ls_client, model_id="all-MiniLM-L6-v2", params=None)

        assert model.params.embedding_dimension == 4
        assert model.params.context_length == 4096
        assert mock_ls_client.embeddings.create.call_count == 2

    def test_init_with_params_missing_dimension_auto_detects(self, mock_ls_client):
        """Test that auto-detection triggers when LSEmbeddingParams has no dimension."""
        params = LSEmbeddingParams()  # embedding_dimension and context_length default to None
        model = LSEmbeddingModel(client=mock_ls_client, model_id="all-MiniLM-L6-v2", params=params)

        assert model.params.embedding_dimension == 4
        assert model.params.context_length == 4096
        assert mock_ls_client.embeddings.create.call_count == 2

    def test_init_with_invalid_params_type(self, mock_ls_client):
        """Test initialization with invalid params type raises TypeError."""
        with pytest.raises(TypeError, match="Incorrect type of 'params' parameter"):
            LSEmbeddingModel(client=mock_ls_client, model_id="all-MiniLM-L6-v2", params="invalid")

    def test_detect_embedding_dimension_api_failure(self, mocker):
        """Test that _detect_embedding_dimension raises RuntimeError on API failure."""
        mock_client = mocker.MagicMock()
        mock_client.embeddings.create.side_effect = ConnectionError("Service unavailable")

        with pytest.raises(RuntimeError, match="Failed to auto-detect embedding dimension"):
            LSEmbeddingModel(client=mock_client, model_id="all-MiniLM-L6-v2", params=None)

    def test_detect_embedding_dimension_preserves_original_exception(self, mocker):
        """Test that the original exception is chained in the RuntimeError."""
        mock_client = mocker.MagicMock()
        original_error = ConnectionError("Connection refused")
        mock_client.embeddings.create.side_effect = original_error

        with pytest.raises(RuntimeError) as exc_info:
            LSEmbeddingModel(client=mock_client, model_id="test-model", params=None)

        assert exc_info.value.__cause__ is original_error

    def test_detect_context_length_succeeds_at_first_probe(self, mock_ls_client):
        """Test that context_length detection returns 4096 when the first probe succeeds."""
        params = LSEmbeddingParams(embedding_dimension=384)
        model = LSEmbeddingModel(client=mock_ls_client, model_id="all-MiniLM-L6-v2", params=params)

        assert model.params.context_length == 4096
        mock_ls_client.embeddings.create.assert_called_once()

    def test_detect_context_length_falls_back_to_smaller_probe(self, mocker):
        """Test that context_length detection falls back to smaller probe sizes."""
        mock_client = mocker.MagicMock()
        response = _make_mock_ls_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        def side_effect(**kwargs):
            text = kwargs.get("input", "")
            if isinstance(text, str) and len(text) > 2048 * 5:
                raise ValueError("Input too long")
            return response

        mock_client.embeddings.create.side_effect = side_effect

        params = LSEmbeddingParams(embedding_dimension=384)
        model = LSEmbeddingModel(client=mock_client, model_id="test-model", params=params)

        assert model.params.context_length == 2048

    def test_detect_context_length_all_probes_fail(self, mocker):
        """Test that RuntimeError is raised when all context_length probes fail."""
        mock_client = mocker.MagicMock()
        dim_response = _make_mock_ls_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return dim_response  # dimension detection
            raise ValueError("Input too long")

        mock_client.embeddings.create.side_effect = side_effect

        with pytest.raises(RuntimeError, match="Failed to auto-detect 'context_length'"):
            LSEmbeddingModel(client=mock_client, model_id="test-model", params=None)

    def test_detect_context_length_skipped_when_explicit(self, mock_ls_client):
        """Test that context_length detection is skipped when explicitly provided."""
        params = LSEmbeddingParams(embedding_dimension=384, context_length=1024)
        model = LSEmbeddingModel(client=mock_ls_client, model_id="all-MiniLM-L6-v2", params=params)

        assert model.params.context_length == 1024
        mock_ls_client.embeddings.create.assert_not_called()

    def test_embed_documents(self, mock_ls_client, mocker):
        """Test embed_documents method."""
        model = LSEmbeddingModel(
            client=mock_ls_client,
            model_id="all-MiniLM-L6-v2",
            params=LSEmbeddingParams(embedding_dimension=3, context_length=512),
        )
        mock_ls_client.embeddings.create.return_value = _make_mock_ls_embedding_response(
            mocker, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        embeddings = model.embed_documents(["text1", "text2"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    def test_embed_query(self, mock_ls_client, mocker):
        """Test embed_query method."""
        model = LSEmbeddingModel(
            client=mock_ls_client,
            model_id="all-MiniLM-L6-v2",
            params=LSEmbeddingParams(embedding_dimension=3, context_length=512),
        )
        mock_ls_client.embeddings.create.return_value = _make_mock_ls_embedding_response(mocker, [[0.1, 0.2, 0.3]])

        embedding = model.embed_query("test query")

        assert embedding == [0.1, 0.2, 0.3]

    def test_model_repr(self, mock_ls_client):
        """Test string representation."""
        model = LSEmbeddingModel(
            client=mock_ls_client,
            model_id="all-MiniLM-L6-v2",
            params=LSEmbeddingParams(embedding_dimension=384, context_length=512),
        )

        assert repr(model) == "all-MiniLM-L6-v2"
        assert str(model) == "all-MiniLM-L6-v2"
