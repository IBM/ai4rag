# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock, Mock

import pytest

from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.search_space.prepare.llama_stack_utils import (
    SearchSpaceValueError,
    _are_provided_models_available,
    _get_default_llama_stack_models,
    _validate_embedding_model,
    _validate_foundation_model,
)


class TestValidateFoundationModel:
    """Test _validate_foundation_model function."""

    def test_returns_true_when_model_responds(self):
        """Test that validation returns True when model responds successfully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Mock(choices=[])

        model = LSFoundationModel(model_id="test-model", client=mock_client)

        result = _validate_foundation_model(model)

        assert result is True
        mock_client.chat.completions.create.assert_called_once()

    def test_returns_false_when_model_fails(self):
        """Test that validation returns False when model raises exception."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Model error")

        model = LSFoundationModel(model_id="test-model", client=mock_client)

        result = _validate_foundation_model(model)

        assert result is False


class TestValidateEmbeddingModel:
    """Test _validate_embedding_model function."""

    def test_returns_true_when_model_responds(self):
        """Test that validation returns True when model responds successfully."""
        mock_client = MagicMock()
        mock_data = Mock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response

        model = LSEmbeddingModel(
            model_id="test-model",
            client=mock_client,
            params=LSEmbeddingParams(embedding_dimension=768, context_length=512),
        )

        result = _validate_embedding_model(model)

        assert result is True

    def test_returns_false_when_model_fails(self):
        """Test that validation returns False when model raises exception."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_data = Mock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_data]

        # First call succeeds (for embed_query in validation), but we need to
        # create the model first with context_length set to avoid detection.
        model = LSEmbeddingModel(
            model_id="test-model",
            client=mock_client,
            params=LSEmbeddingParams(embedding_dimension=768, context_length=512),
        )

        # Now set embed to fail for validation
        mock_client.embeddings.create.side_effect = Exception("Model error")

        result = _validate_embedding_model(model)

        assert result is False


class TestGetDefaultLlamaStackModels:
    """Test _get_default_llama_stack_models function."""

    def test_returns_foundation_and_embedding_models(self, mocker):
        """Test that function returns both foundation and embedding models."""
        # Mock client
        mock_client = MagicMock()

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "test-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "test-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 512}
        mock_embedding.metadata = {}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        # Call function
        result = _get_default_llama_stack_models(mock_client)

        # Assertions
        assert "foundation_models" in result
        assert "embedding_models" in result
        assert "not_responding_foundation_models" in result
        assert "not_responding_embedding_models" in result
        assert len(result["foundation_models"]) == 1
        assert len(result["embedding_models"]) == 1
        assert result["foundation_models"][0].model_id == "test-llm"
        assert result["embedding_models"][0].model_id == "test-embedding"
        assert result["not_responding_foundation_models"] == []
        assert result["not_responding_embedding_models"] == []

    def test_raises_error_when_no_llm_models(self, mocker):
        """Test that function raises error when no LLM models available."""
        mock_client = MagicMock()

        mock_embedding = Mock()
        mock_embedding.id = "test-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 512}
        mock_embedding.metadata = {}

        mock_client.models.list.return_value = [mock_embedding]

        # Mock validation to return True for embedding models
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        with pytest.raises(SearchSpaceValueError, match="no available models of type 'llm'.*not responding"):
            _get_default_llama_stack_models(mock_client)

    def test_raises_error_when_no_embedding_models(self, mocker):
        """Test that function raises error when no embedding models available."""
        mock_client = MagicMock()

        mock_llm = Mock()
        mock_llm.id = "test-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_client.models.list.return_value = [mock_llm]

        # Mock validation to return True for foundation models
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )

        with pytest.raises(SearchSpaceValueError, match="no available models of type 'embedding'.*not responding"):
            _get_default_llama_stack_models(mock_client)

    def test_excludes_models_that_fail_validation(self, mocker):
        """Test that models failing validation are excluded from results."""
        mock_client = MagicMock()

        # Create multiple models of each type
        mock_llm1 = Mock()
        mock_llm1.id = "test-llm-1"
        mock_llm1.custom_metadata = {"model_type": "llm"}

        mock_llm2 = Mock()
        mock_llm2.id = "test-llm-2"
        mock_llm2.custom_metadata = {"model_type": "llm"}

        mock_embedding1 = Mock()
        mock_embedding1.id = "test-embedding-1"
        mock_embedding1.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 512}
        mock_embedding1.metadata = {}

        mock_embedding2 = Mock()
        mock_embedding2.id = "test-embedding-2"
        mock_embedding2.custom_metadata = {
            "model_type": "embedding",
            "embedding_dimension": 1024,
            "context_length": 512,
        }
        mock_embedding2.metadata = {}

        mock_client.models.list.return_value = [mock_llm1, mock_llm2, mock_embedding1, mock_embedding2]

        # Mock validation to fail for first model of each type
        def mock_validate_foundation(model):
            return model.model_id != "test-llm-1"

        def mock_validate_embedding(model):
            return model.model_id != "test-embedding-1"

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            side_effect=mock_validate_foundation,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            side_effect=mock_validate_embedding,
        )

        result = _get_default_llama_stack_models(mock_client)

        # Only validated models should be included
        assert len(result["foundation_models"]) == 1
        assert result["foundation_models"][0].model_id == "test-llm-2"
        assert len(result["embedding_models"]) == 1
        assert result["embedding_models"][0].model_id == "test-embedding-2"

        # Failed models should be in the not_responding lists
        assert len(result["not_responding_foundation_models"]) == 1
        assert result["not_responding_foundation_models"][0].model_id == "test-llm-1"
        assert len(result["not_responding_embedding_models"]) == 1
        assert result["not_responding_embedding_models"][0].model_id == "test-embedding-1"

    def test_raises_error_when_all_foundation_models_fail_validation(self, mocker):
        """Test that error is raised when all foundation models fail validation."""
        mock_client = MagicMock()

        mock_llm = Mock()
        mock_llm.id = "test-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "test-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 512}
        mock_embedding.metadata = {}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Mock validation to fail for foundation model
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=False,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        with pytest.raises(SearchSpaceValueError, match="no available models of type 'llm'.*not responding"):
            _get_default_llama_stack_models(mock_client)

    def test_raises_error_when_all_embedding_models_fail_validation(self, mocker):
        """Test that error is raised when all embedding models fail validation."""
        mock_client = MagicMock()

        mock_llm = Mock()
        mock_llm.id = "test-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "test-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 512}
        mock_embedding.metadata = {}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Mock validation to fail for embedding model
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=False,
        )

        with pytest.raises(SearchSpaceValueError, match="no available models of type 'embedding'.*not responding"):
            _get_default_llama_stack_models(mock_client)


class TestAreProvidedModelsAvailable:
    """Test _are_provided_models_available function."""

    def test_no_error_when_all_models_available(self):
        """Test that function does not raise when all provided models are available."""
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

        _are_provided_models_available(provided_models, available_models, not_responding_models=[])

    def test_raises_error_when_model_not_registered(self):
        """Test that function raises error when provided model is not registered."""
        from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
        from ai4rag.search_space.prepare.input_payload_types import AI4RAGFoundationModel

        mock_client = MagicMock()
        available_models = [
            LSFoundationModel(model_id="model-1", client=mock_client),
        ]

        provided_models = [
            AI4RAGFoundationModel(model_id="model-2"),
        ]

        with pytest.raises(SearchSpaceValueError, match="model-2.*not registered within llama-stack"):
            _are_provided_models_available(provided_models, available_models, not_responding_models=[])

    def test_raises_error_when_model_not_responding(self):
        """Test that function raises error when provided model is registered but not responding."""
        from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
        from ai4rag.search_space.prepare.input_payload_types import AI4RAGFoundationModel

        mock_client = MagicMock()
        available_models = [
            LSFoundationModel(model_id="model-1", client=mock_client),
        ]
        not_responding_models = [
            LSFoundationModel(model_id="model-2", client=mock_client),
        ]

        provided_models = [
            AI4RAGFoundationModel(model_id="model-2"),
        ]

        with pytest.raises(SearchSpaceValueError, match="model-2.*registered but do not respond"):
            _are_provided_models_available(provided_models, available_models, not_responding_models)

    def test_raises_error_with_both_not_responding_and_unregistered(self):
        """Test that error includes both not-responding and unregistered models."""
        from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
        from ai4rag.search_space.prepare.input_payload_types import AI4RAGFoundationModel

        mock_client = MagicMock()
        available_models = [
            LSFoundationModel(model_id="model-1", client=mock_client),
        ]
        not_responding_models = [
            LSFoundationModel(model_id="model-2", client=mock_client),
        ]

        provided_models = [
            AI4RAGFoundationModel(model_id="model-2"),
            AI4RAGFoundationModel(model_id="model-3"),
        ]

        with pytest.raises(SearchSpaceValueError) as exc_info:
            _are_provided_models_available(provided_models, available_models, not_responding_models)

        error_msg = str(exc_info.value)
        assert "model-2" in error_msg
        assert "registered but do not respond" in error_msg
        assert "model-3" in error_msg
        assert "not registered within llama-stack" in error_msg
