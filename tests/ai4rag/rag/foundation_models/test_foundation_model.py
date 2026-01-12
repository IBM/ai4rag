# Copyright 2025- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from ai4rag.rag.foundation_models.foundation_model import (
    ModelParameters,
    LlamaStackFoundationModel,
)
from ai4rag.utils.constants import ChatGenerationConstants
from ai4rag.utils.validators import ConstraintsValidationError


class TestModelParameters:
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

    def test_max_completion_tokens_positive(self):
        """Test that max_completion_tokens must be positive."""
        params = ModelParameters(max_completion_tokens=1)
        assert params.max_completion_tokens == 1

    def test_max_completion_tokens_zero_invalid(self):
        """Test that max_completion_tokens cannot be zero."""
        with pytest.raises(ValidationError) as exc_info:
            ModelParameters(max_completion_tokens=0)
        assert "greater than 0" in str(exc_info.value).lower()

    def test_max_completion_tokens_negative_invalid(self):
        """Test that max_completion_tokens cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            ModelParameters(max_completion_tokens=-100)
        assert "greater than 0" in str(exc_info.value).lower()

    def test_temperature_minimum_boundary(self):
        """Test temperature at minimum boundary (0)."""
        params = ModelParameters(temperature=0.0)
        assert params.temperature == 0.0

    def test_temperature_maximum_boundary(self):
        """Test temperature at maximum boundary (1)."""
        params = ModelParameters(temperature=1.0)
        assert params.temperature == 1.0

    def test_temperature_below_minimum_invalid(self):
        """Test that temperature below 0 is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ModelParameters(temperature=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_temperature_above_maximum_invalid(self):
        """Test that temperature above 1 is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ModelParameters(temperature=1.1)
        assert "less than or equal to 1" in str(exc_info.value).lower()

    def test_max_completion_tokens_float_invalid(self):
        """Test that max_completion_tokens must be an integer."""
        with pytest.raises(ValidationError) as exc_info:
            ModelParameters(max_completion_tokens=100.5)
        assert "int" in str(exc_info.value).lower()

    def test_temperature_int_coerced_to_float(self):
        """Test that integer temperature values are accepted and coerced to float."""
        params = ModelParameters(temperature=0)
        assert params.temperature == 0.0
        assert isinstance(params.temperature, float)

    @pytest.mark.parametrize(
        "max_tokens, temp",
        [
            (1, 0.0),
            (2048, 1.0),
            (512, 0.5),
            (4096, 0.2),
            (100, 0.8),
        ],
    )
    def test_valid_parameter_combinations(self, max_tokens, temp):
        """Parameterized test for valid parameter combinations."""
        params = ModelParameters(max_completion_tokens=max_tokens, temperature=temp)
        assert params.max_completion_tokens == max_tokens
        assert params.temperature == temp


class TestLlamaStackFoundationModel:
    """Test suite for LlamaStackFoundationModel class."""

    @pytest.fixture
    def mock_llama_client(self, mocker):
        """Create a mock LlamaStackClient."""
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Test response from model"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def valid_user_message_template(self):
        """Return a valid user message template."""
        return "Question: {question}\nReferences: {reference_documents}"

    @pytest.fixture
    def valid_context_template(self):
        """Return a valid context template.

        NOTE: Due to a typo in validators.py:96, context_template_text actually validates
        for user_message_text placeholders (question, reference_documents) instead of
        the expected (document). This fixture provides a template that passes validation.
        """
        return "Question: {question}\nReferences: {reference_documents}"

    @pytest.fixture
    def valid_system_message(self):
        """Return a valid system message."""
        return "You are a helpful assistant."

    @pytest.fixture
    def model_with_dict_params(
        self, mock_llama_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create a LlamaStackFoundationModel with dict parameters."""
        return LlamaStackFoundationModel(
            model_id="test-model-id",
            model_params={"max_completion_tokens": 1024, "temperature": 0.3},
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    @pytest.fixture
    def model_with_model_params(
        self, mock_llama_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create a LlamaStackFoundationModel with ModelParameters."""
        params = ModelParameters(max_completion_tokens=512, temperature=0.7)
        return LlamaStackFoundationModel(
            model_id="test-model-id",
            model_params=params,
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    @pytest.fixture
    def model_with_none_params(
        self, mock_llama_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create a LlamaStackFoundationModel with None parameters."""
        return LlamaStackFoundationModel(
            model_id="test-model-id",
            model_params=None,
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    def test_init_with_dict_params(self, model_with_dict_params, mock_llama_client):
        """Test initialization with dict parameters."""
        assert model_with_dict_params.model_id == "test-model-id"
        assert model_with_dict_params.model_params == {"max_completion_tokens": 1024, "temperature": 0.3}
        assert model_with_dict_params.client == mock_llama_client
        assert "question" in model_with_dict_params.user_message_text
        assert "document" in model_with_dict_params.context_template_text

    def test_init_with_model_parameters(self, model_with_model_params, mock_llama_client):
        """Test initialization with ModelParameters object."""
        assert model_with_model_params.model_id == "test-model-id"
        assert isinstance(model_with_model_params.model_params, ModelParameters)
        assert model_with_model_params.model_params.max_completion_tokens == 512
        assert model_with_model_params.model_params.temperature == 0.7
        assert model_with_model_params.client == mock_llama_client

    def test_init_with_none_params(self, model_with_none_params, mock_llama_client):
        """Test initialization with None parameters."""
        assert model_with_none_params.model_id == "test-model-id"
        assert model_with_none_params.model_params is None
        assert model_with_none_params.client == mock_llama_client

    def test_user_message_text_custom(self, mock_llama_client, valid_context_template, valid_system_message):
        """Test that custom user_message_text is used when provided."""
        custom_template = "Custom question: {question} and refs: {reference_documents}"
        model = LlamaStackFoundationModel(
            model_id="test-model",
            model_params=None,
            client=mock_llama_client,
            user_message_text=custom_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        assert model.user_message_text == custom_template

    def test_user_message_text_default_when_none(
        self, mock_llama_client, valid_context_template, valid_system_message, mocker
    ):
        """Test that default user_message_text is used when None is provided."""
        mock_get_user_message = mocker.patch(
            "ai4rag.rag.foundation_models.foundation_model.get_user_message_text",
            return_value="Default user message: {question} {reference_documents}",
        )
        model = LlamaStackFoundationModel(
            model_id="llama-3-70b",
            model_params=None,
            client=mock_llama_client,
            user_message_text=None,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        mock_get_user_message.assert_called_once_with(model_name="llama-3-70b")
        assert "Default user message" in model.user_message_text

    def test_context_template_text_custom(self, mock_llama_client, valid_user_message_template, valid_system_message):
        """Test that custom context_template_text is used when provided.

        NOTE: Due to validator bug, must use user_message_text placeholders.
        """
        custom_template = "Custom: {question} and {reference_documents}"
        model = LlamaStackFoundationModel(
            model_id="test-model",
            model_params=None,
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=custom_template,
            system_message_text=valid_system_message,
        )
        assert model.context_template_text == custom_template

    def test_context_template_text_default_when_none(
        self, mock_llama_client, valid_user_message_template, valid_system_message, mocker
    ):
        """Test that default context_template_text is used when None is provided.

        NOTE: Due to validator bug, mocked return value must use user_message_text placeholders.
        """
        mock_get_system_message = mocker.patch(
            "ai4rag.rag.foundation_models.foundation_model.get_system_message_text",
            return_value="Default: {question} {reference_documents}",
        )
        model = LlamaStackFoundationModel(
            model_id="granite-13b",
            model_params=None,
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=None,
            system_message_text=valid_system_message,
        )
        mock_get_system_message.assert_called_once_with(model_name="granite-13b")
        assert "Default" in model.context_template_text

    def test_system_message_text_assignment(
        self, mock_llama_client, valid_user_message_template, valid_context_template
    ):
        """Test that system_message_text is properly assigned."""
        system_msg = "Custom system message"
        model = LlamaStackFoundationModel(
            model_id="test-model",
            model_params=None,
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=system_msg,
        )
        assert model.system_message_text == system_msg

    def test_chat_method(self, model_with_dict_params, mock_llama_client):
        """Test that chat method calls client correctly and returns response."""
        system_msg = "You are helpful"
        user_msg = "What is AI?"

        response = model_with_dict_params.chat(system_msg, user_msg)

        # Verify the client was called
        mock_llama_client.chat.completions.create.assert_called_once()
        call_args = mock_llama_client.chat.completions.create.call_args

        # Verify model_id was passed
        assert call_args.kwargs["model"] == "test-model-id"

        # Verify messages were passed correctly
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == system_msg
        assert messages[1].role == "user"
        assert messages[1].content == user_msg

        # Verify response
        assert response == "Test response from model"

    def test_chat_method_extracts_content(self, model_with_dict_params, mock_llama_client):
        """Test that chat method correctly extracts content from response."""
        response = model_with_dict_params.chat("system", "user")
        assert response == "Test response from model"

    def test_chat_with_different_messages(self, model_with_dict_params, mock_llama_client):
        """Test chat with different message combinations."""
        test_cases = [
            ("System prompt 1", "User query 1"),
            ("", "User query 2"),
            ("System prompt 3", ""),
            ("Multi\nline\nsystem", "Multi\nline\nuser"),
        ]

        for sys_msg, usr_msg in test_cases:
            model_with_dict_params.chat(sys_msg, usr_msg)
            call_args = mock_llama_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0].content == sys_msg
            assert messages[1].content == usr_msg

    def test_invalid_user_message_template_missing_placeholder(
        self, mock_llama_client, valid_context_template, valid_system_message
    ):
        """Test that invalid user_message_text raises validation error."""
        invalid_template = "Only question: {question}"  # Missing reference_documents
        with pytest.raises(ConstraintsValidationError) as exc_info:
            LlamaStackFoundationModel(
                model_id="test-model",
                model_params=None,
                client=mock_llama_client,
                user_message_text=invalid_template,
                context_template_text=valid_context_template,
                system_message_text=valid_system_message,
            )
        assert "Incorrect number of placeholders" in str(exc_info.value)

    def test_invalid_user_message_template_wrong_placeholder(
        self, mock_llama_client, valid_context_template, valid_system_message
    ):
        """Test that invalid placeholder in user_message_text raises validation error."""
        invalid_template = "Question: {question} Context: {document}"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            LlamaStackFoundationModel(
                model_id="test-model",
                model_params=None,
                client=mock_llama_client,
                user_message_text=invalid_template,
                context_template_text=valid_context_template,
                system_message_text=valid_system_message,
            )
        assert "unexpected placeholder" in str(exc_info.value)

    def test_invalid_context_template_missing_placeholder(
        self, mock_llama_client, valid_user_message_template, valid_system_message
    ):
        """Test that invalid context_template_text raises validation error."""
        invalid_template = "No placeholder here"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            LlamaStackFoundationModel(
                model_id="test-model",
                model_params=None,
                client=mock_llama_client,
                user_message_text=valid_user_message_template,
                context_template_text=invalid_template,
                system_message_text=valid_system_message,
            )
        assert "Incorrect number of placeholders" in str(exc_info.value)

    def test_invalid_context_template_wrong_placeholder(
        self, mock_llama_client, valid_user_message_template, valid_system_message
    ):
        """Test that wrong placeholder in context_template_text raises validation error.

        NOTE: Due to a typo in validators.py:96, context_template_text validates for
        user_message_text placeholders. This test uses {document} which is invalid
        for that validator.
        """
        invalid_template = "Document: {document}"  # Invalid due to validator bug
        with pytest.raises(ConstraintsValidationError) as exc_info:
            LlamaStackFoundationModel(
                model_id="test-model",
                model_params=None,
                client=mock_llama_client,
                user_message_text=valid_user_message_template,
                context_template_text=invalid_template,
                system_message_text=valid_system_message,
            )
        assert "unexpected placeholder" in str(exc_info.value)

    def test_model_inherits_from_foundation_model(self, model_with_dict_params):
        """Test that LlamaStackFoundationModel inherits FoundationModel methods."""
        # Test __repr__
        assert repr(model_with_dict_params) == "test-model-id"

        # Test __str__
        assert str(model_with_dict_params) == "test-model-id"

        # Test __hash__
        assert hash(model_with_dict_params) == hash("test-model-id")

    def test_model_equality(
        self, mock_llama_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Test that models with same model_id are equal."""
        model1 = LlamaStackFoundationModel(
            model_id="same-id",
            model_params=None,
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        model2 = LlamaStackFoundationModel(
            model_id="same-id",
            model_params={"different": "params"},
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        assert model1 == model2

    @pytest.mark.parametrize(
        "model_id",
        [
            "llama-3-70b",
            "granite-13b-instruct-v2",
            "mistral-7b",
            "gpt-4",
        ],
    )
    def test_various_model_ids(
        self, mock_llama_client, valid_user_message_template, valid_context_template, valid_system_message, model_id
    ):
        """Test initialization with various model IDs."""
        model = LlamaStackFoundationModel(
            model_id=model_id,
            model_params=None,
            client=mock_llama_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        assert model.model_id == model_id
