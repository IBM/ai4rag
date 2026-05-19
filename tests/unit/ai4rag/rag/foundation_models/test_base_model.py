# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

import pytest

from ai4rag.rag.foundation_models.base_model import BaseFoundationModel


class ConcreteFoundationModel(BaseFoundationModel[Any, dict]):
    """Concrete implementation of BaseFoundationModel for testing purposes."""

    def create_response(self, user_message: str, vector_store_id: str | None = None) -> str:
        """Simple create_response implementation for testing."""
        # Simulate processing the message
        if vector_store_id:
            return f"Response to '{user_message}' using vector store '{vector_store_id}'"
        return f"Response to '{user_message}'"


class TestFoundationModel:
    """Test suite for BaseFoundationModel base class."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create a mock client for testing."""
        return mocker.MagicMock()

    @pytest.fixture
    def model_params(self):
        """Create sample model parameters."""
        return {"temperature": 0.5, "max_tokens": 100}

    @pytest.fixture
    def foundation_model(self, mock_client, model_params):
        """Create a concrete foundation model instance for testing."""
        return ConcreteFoundationModel(client=mock_client, model_id="test-model-id", params=model_params)

    def test_init(self, mock_client, model_params):
        """Test BaseFoundationModel initialization."""
        model = ConcreteFoundationModel(client=mock_client, model_id="test-model-123", params=model_params)
        assert model.client == mock_client
        assert model.model_id == "test-model-123"
        assert model.params == model_params

    def test_repr(self, foundation_model):
        """Test __repr__ returns model_id."""
        assert repr(foundation_model) == "test-model-id"

    def test_str(self, foundation_model):
        """Test __str__ returns same as __repr__."""
        assert str(foundation_model) == repr(foundation_model)
        assert str(foundation_model) == "test-model-id"

    def test_eq_same_model_id(self, mock_client, model_params):
        """Test equality when model_ids are the same."""
        model1 = ConcreteFoundationModel(client=mock_client, model_id="same-id", params=model_params)
        model2 = ConcreteFoundationModel(client=mock_client, model_id="same-id", params={"different": "params"})
        assert model1 == model2

    def test_eq_different_model_id(self, mock_client, model_params):
        """Test inequality when model_ids are different."""
        model1 = ConcreteFoundationModel(client=mock_client, model_id="model-1", params=model_params)
        model2 = ConcreteFoundationModel(client=mock_client, model_id="model-2", params=model_params)
        assert model1 != model2

    def test_eq_with_non_foundation_model(self, foundation_model):
        """Test equality with non-BaseFoundationModel object returns NotImplemented."""
        assert foundation_model.__eq__("not-a-foundation-model") is NotImplemented

    def test_eq_with_none(self, foundation_model):
        """Test equality with None returns NotImplemented."""
        assert foundation_model.__eq__(None) is NotImplemented

    def test_eq_with_dict(self, foundation_model):
        """Test equality with dict returns NotImplemented."""
        assert foundation_model.__eq__({"model_id": "test-model-id"}) is NotImplemented

    def test_hash(self, foundation_model):
        """Test __hash__ returns hash of model_id."""
        assert hash(foundation_model) == hash("test-model-id")

    def test_hash_consistency(self, mock_client, model_params):
        """Test that models with same model_id have same hash."""
        model1 = ConcreteFoundationModel(client=mock_client, model_id="same-id", params=model_params)
        model2 = ConcreteFoundationModel(client=mock_client, model_id="same-id", params={"different": "params"})
        assert hash(model1) == hash(model2)

    def test_hash_different_for_different_ids(self, mock_client, model_params):
        """Test that models with different model_ids have different hashes."""
        model1 = ConcreteFoundationModel(client=mock_client, model_id="model-1", params=model_params)
        model2 = ConcreteFoundationModel(client=mock_client, model_id="model-2", params=model_params)
        assert hash(model1) != hash(model2)

    def test_models_can_be_used_in_set(self, mock_client, model_params):
        """Test that BaseFoundationModel instances can be added to a set."""
        model1 = ConcreteFoundationModel(client=mock_client, model_id="model-1", params=model_params)
        model2 = ConcreteFoundationModel(client=mock_client, model_id="model-2", params=model_params)
        model3 = ConcreteFoundationModel(
            client=mock_client, model_id="model-1", params={"different": "params"}  # Same as model1
        )
        model_set = {model1, model2, model3}
        assert len(model_set) == 2  # model1 and model3 are considered equal

    def test_models_can_be_used_as_dict_keys(self, mock_client, model_params):
        """Test that BaseFoundationModel instances can be used as dictionary keys."""
        model1 = ConcreteFoundationModel(client=mock_client, model_id="model-1", params=model_params)
        model2 = ConcreteFoundationModel(
            client=mock_client, model_id="model-1", params={"different": "params"}  # Same as model1
        )
        model_dict = {model1: "value1"}
        model_dict[model2] = "value2"
        assert len(model_dict) == 1  # model2 overwrites model1
        assert model_dict[model1] == "value2"

    def test_create_response_implementation(self, foundation_model):
        """Test that concrete implementation's create_response method works."""
        user_message = "What is AI?"
        result = foundation_model.create_response(user_message)
        assert result == "Response to 'What is AI?'"

    def test_create_response_with_vector_store(self, foundation_model):
        """Test create_response with vector_store_id parameter."""
        user_message = "What is AI?"
        vector_store_id = "test-vector-store"
        result = foundation_model.create_response(user_message, vector_store_id)
        assert result == "Response to 'What is AI?' using vector store 'test-vector-store'"

    def test_create_response_is_abstract(self):
        """Test that BaseFoundationModel.create_response is abstract and cannot be instantiated without implementation."""
        with pytest.raises(TypeError) as exc_info:
            BaseFoundationModel(client=None, model_id="test", params={})
        assert "Can't instantiate abstract class" in str(exc_info.value)

    @pytest.mark.parametrize(
        "model_id,expected_repr",
        [
            ("simple-id", "simple-id"),
            ("llama-3-70b", "llama-3-70b"),
            ("model/with/slashes", "model/with/slashes"),
            ("model:with:colons", "model:with:colons"),
            ("", ""),
        ],
    )
    def test_repr_various_model_ids(self, mock_client, model_params, model_id, expected_repr):
        """Test __repr__ with various model_id formats."""
        model = ConcreteFoundationModel(client=mock_client, model_id=model_id, params=model_params)
        assert repr(model) == expected_repr

    def test_different_client_types(self, model_params):
        """Test that BaseFoundationModel works with different client types."""

        class CustomClient:
            pass

        client = CustomClient()
        model = ConcreteFoundationModel(client=client, model_id="test-model", params=model_params)
        assert isinstance(model.client, CustomClient)

    def test_different_param_types(self, mock_client):
        """Test that BaseFoundationModel works with different parameter types."""
        # Test with dict
        model1 = ConcreteFoundationModel(client=mock_client, model_id="model-1", params={"key": "value"})
        assert model1.params == {"key": "value"}

        # Test with None
        model2 = ConcreteFoundationModel(client=mock_client, model_id="model-2", params=None)
        assert model2.params is None

        # Test with custom object
        class CustomParams:
            pass

        params = CustomParams()
        model3 = ConcreteFoundationModel(client=mock_client, model_id="model-3", params=params)
        assert model3.params is params
