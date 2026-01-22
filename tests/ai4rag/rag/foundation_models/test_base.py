# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest
from typing import Any
from pydantic import ValidationError

from ai4rag.rag.foundation_models.base_model import FoundationModel
from ai4rag.rag.foundation_models.foundation_model import ModelParameters
from ai4rag.utils.constants import ChatGenerationConstants


class ConcreteFoundationModelLegacy(FoundationModel[Any, dict[str, Any] | ModelParameters | None]):
    """Concrete implementation of legacy FoundationModel for testing purposes."""

    def chat(self, system_message: str, user_message: str) -> str:
        """Simple chat implementation for testing."""
        return f"System: {system_message}, User: {user_message}"


class TestModelParametersLegacy:
    """Test suite for ModelParameters class."""

    def test_default_values(self):
        """Test ModelParameters with default values."""
        params = ModelParameters()
        assert params.max_completion_tokens == ChatGenerationConstants.MAX_COMPLETION_TOKENS
        assert params.temperature == ChatGenerationConstants.TEMPERATURE

    def test_custom_values(self):
        """Test ModelParameters with custom valid values."""
        params = ModelParameters(max_completion_tokens=1024, temperature=0.5)
        assert params.max_completion_tokens == 1024
        assert params.temperature == 0.5

    def test_to_dict_method(self):
        """Test model_dump method returns parameters as dictionary."""
        params = ModelParameters(max_completion_tokens=512, temperature=0.7)
        params_dict = params.model_dump()
        assert isinstance(params_dict, dict)
        assert params_dict["max_completion_tokens"] == 512
        assert params_dict["temperature"] == 0.7

    def test_to_dict_with_defaults(self):
        """Test model_dump with default values."""
        params = ModelParameters()
        params_dict = params.model_dump()
        assert params_dict["max_completion_tokens"] == ChatGenerationConstants.MAX_COMPLETION_TOKENS
        assert params_dict["temperature"] == ChatGenerationConstants.TEMPERATURE

    def test_max_completion_tokens_validation(self):
        """Test max_completion_tokens validation."""
        with pytest.raises(ValidationError):
            ModelParameters(max_completion_tokens=0)

        with pytest.raises(ValidationError):
            ModelParameters(max_completion_tokens=-1)

    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            ModelParameters(temperature=-0.1)

        with pytest.raises(ValidationError):
            ModelParameters(temperature=1.1)

    def test_temperature_boundaries(self):
        """Test temperature at valid boundaries."""
        params_min = ModelParameters(temperature=0.0)
        assert params_min.temperature == 0.0

        params_max = ModelParameters(temperature=1.0)
        assert params_max.temperature == 1.0


class TestFoundationModelLegacy:
    """Test suite for legacy FoundationModel base class."""

    @pytest.fixture
    def valid_user_message_template(self):
        """Return a valid user message template."""
        return "Question: {question}\nReferences: {reference_documents}"

    @pytest.fixture
    def valid_context_template(self):
        """Return a valid context template."""
        return "Document: {document}"

    @pytest.fixture
    def valid_system_message(self):
        """Return a valid system message."""
        return "You are a helpful assistant."

    @pytest.fixture
    def mock_client(self, mocker):
        """Create a mock client for testing."""
        return mocker.MagicMock()

    @pytest.fixture
    def model_with_dict_params(
        self, mock_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create a model with dict parameters."""
        return ConcreteFoundationModelLegacy(
            client=mock_client,
            model_id="test-model-id",
            model_params={"max_completion_tokens": 1024, "temperature": 0.3},
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    @pytest.fixture
    def model_with_model_params(
        self, mock_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create a model with ModelParameters object."""
        params = ModelParameters(max_completion_tokens=512, temperature=0.7)
        return ConcreteFoundationModelLegacy(
            client=mock_client,
            model_id="test-model-id",
            model_params=params,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    @pytest.fixture
    def model_with_none_params(
        self, mock_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create a model with None parameters (should use defaults)."""
        return ConcreteFoundationModelLegacy(
            client=mock_client,
            model_id="test-model-id",
            model_params=None,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    def test_init_basic(self, mock_client):
        """Test basic initialization."""
        model = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="test-model", model_params={"max_completion_tokens": 100}
        )
        assert model.model_id == "test-model"
        assert model.model_params == {"max_completion_tokens": 100}

    def test_init_with_kwargs(
        self, mock_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Test initialization with kwargs."""
        model = ConcreteFoundationModelLegacy(
            client=mock_client,
            model_id="test-model",
            model_params=None,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        assert model.user_message_text == valid_user_message_template
        assert model.context_template_text == valid_context_template
        assert model.system_message_text == valid_system_message

    def test_init_without_optional_kwargs(self, mock_client, mocker):
        """Test initialization without optional kwargs."""
        mocker.patch(
            "ai4rag.rag.foundation_models.base_model.get_user_message_text",
            return_value="Question: {question}\nReferences: {reference_documents}",
        )
        mocker.patch(
            "ai4rag.rag.foundation_models.base_model.get_context_template_text", return_value="Document: {document}"
        )
        mocker.patch("ai4rag.rag.foundation_models.base_model.get_system_message_text", return_value="default system")
        model = ConcreteFoundationModelLegacy(client=mock_client, model_id="test-model", model_params=None)
        assert model.user_message_text == "Question: {question}\nReferences: {reference_documents}"
        assert model.context_template_text == "Document: {document}"
        assert model.system_message_text == "default system"

    def test_repr(self, model_with_dict_params):
        """Test __repr__ returns model_id."""
        assert repr(model_with_dict_params) == "test-model-id"

    def test_str(self, model_with_dict_params):
        """Test __str__ returns same as __repr__."""
        assert str(model_with_dict_params) == repr(model_with_dict_params)
        assert str(model_with_dict_params) == "test-model-id"

    def test_eq_same_model_id(self, mock_client):
        """Test equality when model_ids are the same."""
        model1 = ConcreteFoundationModelLegacy(client=mock_client, model_id="same-id", model_params=None)
        model2 = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="same-id", model_params={"different": "params"}
        )
        assert model1 == model2

    def test_eq_different_model_id(self, mock_client):
        """Test inequality when model_ids are different."""
        model1 = ConcreteFoundationModelLegacy(client=mock_client, model_id="model-1", model_params=None)
        model2 = ConcreteFoundationModelLegacy(client=mock_client, model_id="model-2", model_params=None)
        assert model1 != model2

    def test_eq_with_non_foundation_model(self, model_with_dict_params):
        """Test equality with non-FoundationModel object raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            model_with_dict_params == "not-a-foundation-model"

    def test_hash(self, model_with_dict_params):
        """Test __hash__ returns hash of model_id."""
        assert hash(model_with_dict_params) == hash("test-model-id")

    def test_hash_consistency(self, mock_client):
        """Test that models with same model_id have same hash."""
        model1 = ConcreteFoundationModelLegacy(client=mock_client, model_id="same-id", model_params=None)
        model2 = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="same-id", model_params={"different": "params"}
        )
        assert hash(model1) == hash(model2)

    def test_model_params_property_getter(self, model_with_dict_params):
        """Test model_params property getter."""
        params = model_with_dict_params.model_params
        assert params == {"max_completion_tokens": 1024, "temperature": 0.3}

    def test_system_message_text_property_getter(self, model_with_dict_params):
        """Test system_message_text property getter."""
        assert model_with_dict_params.system_message_text == "You are a helpful assistant."

    def test_system_message_text_setter_with_string(self, mock_client, mocker):
        """Test system_message_text can be set via constructor."""
        mocker.patch("ai4rag.rag.foundation_models.base_model.get_system_message_text", return_value="default")
        model = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="test", model_params=None, system_message_text="Custom system message"
        )
        assert model.system_message_text == "Custom system message"

    def test_system_message_text_setter_with_none(self, mock_client, mocker):
        """Test system_message_text with None loads default."""
        mock_get_system = mocker.patch(
            "ai4rag.rag.foundation_models.base_model.get_system_message_text", return_value="Default system message"
        )
        model = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="llama-3-70b", model_params=None, system_message_text=None
        )
        mock_get_system.assert_called_with(model_name="llama-3-70b")
        assert model.system_message_text == "Default system message"

    def test_user_message_text_property_getter(self, model_with_dict_params):
        """Test user_message_text property getter."""
        assert "question" in model_with_dict_params.user_message_text
        assert "reference_documents" in model_with_dict_params.user_message_text

    def test_user_message_text_setter_with_valid_string(self, mock_client, mocker):
        """Test user_message_text can be set via constructor with valid template string."""
        mocker.patch("ai4rag.rag.foundation_models.base_model.get_user_message_text", return_value="default")
        valid_template = "Q: {question} Refs: {reference_documents}"
        model = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="test", model_params=None, user_message_text=valid_template
        )
        assert model.user_message_text == valid_template

    def test_user_message_text_setter_with_invalid_string(self, mock_client, mocker):
        """Test user_message_text setter with invalid template raises ValueError."""
        mocker.patch("ai4rag.rag.foundation_models.base_model.get_user_message_text", return_value="default")
        invalid_template = "Only question: {question}"  # Missing reference_documents
        with pytest.raises(Exception) as exc_info:  # Could be ConstraintsValidationError or ValueError
            ConcreteFoundationModelLegacy(
                client=mock_client, model_id="test", model_params=None, user_message_text=invalid_template
            )
        assert "Incorrect number of placeholders" in str(exc_info.value) or "unexpected placeholder" in str(
            exc_info.value
        )

    def test_user_message_text_setter_with_none(self, mock_client, mocker):
        """Test user_message_text with None loads default."""
        mock_get_user = mocker.patch(
            "ai4rag.rag.foundation_models.base_model.get_user_message_text",
            return_value="Default user message: {question} {reference_documents}",
        )
        model = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="granite-13b", model_params=None, user_message_text=None
        )
        mock_get_user.assert_called_with(model_name="granite-13b")
        assert "Default user message" in model.user_message_text

    def test_chat_implementation(self, model_with_dict_params):
        """Test that concrete implementation's chat method works."""
        result = model_with_dict_params.chat("system prompt", "user query")
        assert result == "System: system prompt, User: user query"

    def test_chat_is_abstract(self, mock_client):
        """Test that FoundationModel.chat is abstract."""
        with pytest.raises(TypeError) as exc_info:
            FoundationModel(client=mock_client, model_id="test", model_params=None)
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_models_can_be_used_in_set(self, mock_client):
        """Test that FoundationModel instances can be added to a set."""
        model1 = ConcreteFoundationModelLegacy(client=mock_client, model_id="model-1", model_params=None)
        model2 = ConcreteFoundationModelLegacy(client=mock_client, model_id="model-2", model_params=None)
        model3 = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="model-1", model_params={"different": "params"}
        )
        model_set = {model1, model2, model3}
        assert len(model_set) == 2  # model1 and model3 are considered equal

    def test_models_can_be_used_as_dict_keys(self, mock_client):
        """Test that FoundationModel instances can be used as dictionary keys."""
        model1 = ConcreteFoundationModelLegacy(client=mock_client, model_id="model-1", model_params=None)
        model2 = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="model-1", model_params={"different": "params"}
        )
        model_dict = {model1: "value1"}
        model_dict[model2] = "value2"
        assert len(model_dict) == 1
        assert model_dict[model1] == "value2"

    @pytest.mark.parametrize(
        "model_id,expected_repr",
        [
            ("simple-id", "simple-id"),
            ("llama-3-70b", "llama-3-70b"),
            ("model/with/slashes", "model/with/slashes"),
            ("", ""),
        ],
    )
    def test_repr_various_model_ids(self, mock_client, model_id, expected_repr):
        """Test __repr__ with various model_id formats."""
        model = ConcreteFoundationModelLegacy(client=mock_client, model_id=model_id, model_params=None)
        assert repr(model) == expected_repr

    def test_init_with_dict_params(self, mock_client):
        """Test that dict params are stored as-is."""
        model = ConcreteFoundationModelLegacy(
            client=mock_client, model_id="test", model_params={"max_completion_tokens": 100, "temperature": 0.1}
        )
        assert model.model_params == {"max_completion_tokens": 100, "temperature": 0.1}

    def test_init_with_model_parameters_object(self, mock_client):
        """Test initialization with ModelParameters object."""
        params = ModelParameters(max_completion_tokens=256, temperature=0.8)
        model = ConcreteFoundationModelLegacy(client=mock_client, model_id="test", model_params=params)
        assert model.model_params is params

    def test_init_with_none(self, mock_client):
        """Test initialization with None."""
        model = ConcreteFoundationModelLegacy(client=mock_client, model_id="test", model_params=None)
        assert model.model_params is None

    def test_invalid_placeholder_in_user_message(self, mock_client, mocker):
        """Test that invalid placeholder in user_message_text raises error."""
        mocker.patch("ai4rag.rag.foundation_models.base_model.get_user_message_text", return_value="default")
        with pytest.raises(Exception) as exc_info:  # Could be ConstraintsValidationError
            ConcreteFoundationModelLegacy(
                client=mock_client,
                model_id="test",
                model_params=None,
                user_message_text="Question: {question} Context: {document}",
            )
        assert "unexpected placeholder" in str(exc_info.value)

    def test_multiple_kwargs_are_consumed(self, mock_client):
        """Test that all kwargs are properly consumed during initialization."""
        model = ConcreteFoundationModelLegacy(
            client=mock_client,
            model_id="test",
            model_params=None,
            user_message_text="Q: {question} R: {reference_documents}",
            context_template_text="Doc: {document}",
            system_message_text="System",
        )
        assert model.user_message_text == "Q: {question} R: {reference_documents}"
        assert model.context_template_text == "Doc: {document}"
        assert model.system_message_text == "System"
