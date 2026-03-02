# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for prepare_search_space module."""

from unittest.mock import MagicMock, Mock

import pytest
from llama_stack_client import LlamaStackClient

from ai4rag.search_space.prepare_search_space import (
    _are_provided_models_available,
    _get_default_llama_stack_models,
    prepare_search_space_with_llama_stack,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


class TestGetDefaultLlamaStackModels:
    """Test _get_default_llama_stack_models function."""

    def test_returns_foundation_and_embedding_models(self):
        """Test that function returns both foundation and embedding models."""
        # Mock client
        mock_client = MagicMock()

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "test-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "test-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Call function
        result = _get_default_llama_stack_models(mock_client)

        # Assertions
        assert "foundation_models" in result
        assert "embedding_models" in result
        assert len(result["foundation_models"]) == 1
        assert len(result["embedding_models"]) == 1
        assert result["foundation_models"][0].model_id == "test-llm"
        assert result["embedding_models"][0].model_id == "test-embedding"

    def test_raises_error_when_no_llm_models(self):
        """Test that function raises error when no LLM models available."""
        mock_client = MagicMock()

        mock_embedding = Mock()
        mock_embedding.id = "test-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_embedding]

        with pytest.raises(SearchSpaceValueError, match="no available models of type 'llm'"):
            _get_default_llama_stack_models(mock_client)

    def test_raises_error_when_no_embedding_models(self):
        """Test that function raises error when no embedding models available."""
        mock_client = MagicMock()

        mock_llm = Mock()
        mock_llm.id = "test-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_client.models.list.return_value = [mock_llm]

        with pytest.raises(SearchSpaceValueError, match="no available models of type 'embedding'"):
            _get_default_llama_stack_models(mock_client)


class TestAreProvidedModelsAvailable:
    """Test _are_provided_models_available function."""

    def test_returns_true_when_all_models_available(self):
        """Test that function returns True when all provided models are available."""
        from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
        from ai4rag.search_space.prepare.input_payload_types import AI4RAGFoundationModel

        mock_client = MagicMock()
        available_models = [
            LSFoundationModel(model_id="model-1", client=mock_client),
            LSFoundationModel(model_id="model-2", client=mock_client),
        ]

        provided_models = [
            AI4RAGFoundationModel(model_id="model-1"),
            AI4RAGFoundationModel(model_id="model-2"),
        ]

        # Should return True without raising
        result = _are_provided_models_available(provided_models, available_models)
        assert result is True

    def test_raises_error_when_model_not_available(self):
        """Test that function raises error when provided model is not available."""
        from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
        from ai4rag.search_space.prepare.input_payload_types import AI4RAGFoundationModel

        mock_client = MagicMock()
        available_models = [
            LSFoundationModel(model_id="model-1", client=mock_client),
        ]

        provided_models = [
            AI4RAGFoundationModel(model_id="model-2"),  # Not available
        ]

        with pytest.raises(SearchSpaceValueError, match="model-2.*not available"):
            _are_provided_models_available(provided_models, available_models)


class TestPrepareSearchSpaceWithLlamaStack:
    """Test prepare_search_space_with_llama_stack function."""

    def test_basic_payload_with_defaults(self):
        """Test preparation with empty payload using defaults."""
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Empty payload should use defaults
        payload = {}

        result = prepare_search_space_with_llama_stack(payload, mock_client)

        # Should create search space with default models
        # Note: AI4RAGSearchSpace merges user params with defaults, so we get more than 2 params
        assert result is not None
        assert len(result.params) > 0
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

    def test_payload_with_custom_foundation_models(self):
        """Test preparation with custom foundation models."""
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "custom-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        payload = {"foundation_models": [{"model_id": "custom-llm"}]}

        result = prepare_search_space_with_llama_stack(payload, mock_client)

        assert result is not None
        # Verify parameter names are correct
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

        # Verify the custom foundation model is included
        foundation_param = result["foundation_model"]
        assert len(foundation_param.values) == 1
        assert foundation_param.values[0].model_id == "custom-llm"

    def test_payload_with_custom_embedding_models(self):
        """Test preparation with custom embedding models."""
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "custom-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 1024}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        payload = {"embedding_models": [{"model_id": "custom-embedding"}]}

        result = prepare_search_space_with_llama_stack(payload, mock_client)

        assert result is not None
        # Verify parameter names are correct
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

        # Verify the custom embedding model is included
        embedding_param = result["embedding_model"]
        assert len(embedding_param.values) == 1
        assert embedding_param.values[0].model_id == "custom-embedding"

    def test_invalid_payload_raises_validation_error(self):
        """Test that invalid payload raises validation error."""
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Invalid payload with unrecognized parameter
        payload = {"invalid_parameter": "value"}

        with pytest.raises(SearchSpaceValueError, match="Unknown validation error|invalid_parameter"):
            prepare_search_space_with_llama_stack(payload, mock_client)

    def test_non_llama_stack_client_raises_error(self):
        """Test that non-LlamaStackClient raises error."""
        mock_client = MagicMock(spec=object)  # Not a LlamaStackClient

        payload = {}

        with pytest.raises(SearchSpaceValueError, match="Unrecognized client type"):
            prepare_search_space_with_llama_stack(payload, mock_client)
