# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel


class TestOpenAIFoundationModel:
    """Test suite for OpenAIFoundationModel class."""

    @pytest.fixture
    def mock_openai_client(self, mocker):
        """Create a mock OpenAI client."""
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Test response from OpenAI model"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

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
    def model_with_dict_params(
        self, mock_openai_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create an OpenAIFoundationModel with dict parameters."""
        return OpenAIFoundationModel(
            model_id="gpt-4",
            params={"temperature": 0.3, "max_tokens": 1024},
            client=mock_openai_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    @pytest.fixture
    def model_with_none_params(
        self, mock_openai_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Create an OpenAIFoundationModel with None parameters."""
        return OpenAIFoundationModel(
            model_id="gpt-3.5-turbo",
            params=None,
            client=mock_openai_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )

    def test_init_with_dict_params(self, model_with_dict_params, mock_openai_client):
        """Test initialization with dict parameters."""
        assert model_with_dict_params.model_id == "gpt-4"
        assert model_with_dict_params.params == {"temperature": 0.3, "max_tokens": 1024}
        assert model_with_dict_params.client == mock_openai_client
        assert "question" in model_with_dict_params.user_message_text
        assert "document" in model_with_dict_params.context_template_text

    def test_init_with_none_params(self, model_with_none_params, mock_openai_client):
        """Test initialization with None parameters."""
        assert model_with_none_params.model_id == "gpt-3.5-turbo"
        assert model_with_none_params.params is None
        assert model_with_none_params.client == mock_openai_client

    def test_system_message_text_assignment(
        self, mock_openai_client, valid_user_message_template, valid_context_template
    ):
        """Test that system_message_text is properly assigned."""
        system_msg = "Custom system message for OpenAI"
        model = OpenAIFoundationModel(
            model_id="gpt-4",
            params=None,
            client=mock_openai_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=system_msg,
        )
        assert model.system_message_text == system_msg

    def test_chat_method(self, model_with_dict_params, mock_openai_client):
        """Test that chat method calls client correctly and returns response."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is AI?"},
        ]

        response = model_with_dict_params.chat(messages)

        # Verify the client was called
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args

        # Verify model_id was passed
        assert call_args.kwargs["model"] == "gpt-4"

        # Verify messages were passed correctly
        passed_messages = call_args.kwargs["messages"]
        assert len(passed_messages) == 2
        assert passed_messages[0]["role"] == "system"
        assert passed_messages[0]["content"] == "You are helpful"
        assert passed_messages[1]["role"] == "user"
        assert passed_messages[1]["content"] == "What is AI?"

        # Verify response - should return choices list
        assert len(response) == 1
        assert response[0].message.content == "Test response from OpenAI model"

    def test_chat_method_extracts_content(self, model_with_dict_params, mock_openai_client):
        """Test that chat method correctly returns choices from response."""
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        response = model_with_dict_params.chat(messages)
        assert len(response) == 1
        assert response[0].message.content == "Test response from OpenAI model"

    def test_chat_with_different_messages(self, model_with_dict_params, mock_openai_client):
        """Test chat with different message combinations."""
        test_cases = [
            [{"role": "system", "content": "System prompt 1"}, {"role": "user", "content": "User query 1"}],
            [{"role": "system", "content": ""}, {"role": "user", "content": "User query 2"}],
            [{"role": "system", "content": "System prompt 3"}, {"role": "user", "content": ""}],
            [{"role": "system", "content": "Multi\nline\nsystem"}, {"role": "user", "content": "Multi\nline\nuser"}],
        ]

        for test_messages in test_cases:
            model_with_dict_params.chat(test_messages)
            call_args = mock_openai_client.chat.completions.create.call_args
            passed_messages = call_args.kwargs["messages"]
            assert passed_messages[0]["content"] == test_messages[0]["content"]
            assert passed_messages[1]["content"] == test_messages[1]["content"]

    def test_model_inherits_from_foundation_model(self, model_with_dict_params):
        """Test that OpenAIFoundationModel inherits BaseFoundationModel methods."""
        # Test __repr__
        assert repr(model_with_dict_params) == "gpt-4"

        # Test __str__
        assert str(model_with_dict_params) == "gpt-4"

    def test_model_equality(
        self, mock_openai_client, valid_user_message_template, valid_context_template, valid_system_message
    ):
        """Test that models with same model_id are considered equal based on model_id."""
        model1 = OpenAIFoundationModel(
            model_id="gpt-4",
            params=None,
            client=mock_openai_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        model2 = OpenAIFoundationModel(
            model_id="gpt-4",
            params={"different": "params"},
            client=mock_openai_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        assert str(model1) == str(model2)

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-4o",
        ],
    )
    def test_various_model_ids(
        self, mock_openai_client, valid_user_message_template, valid_context_template, valid_system_message, model_id
    ):
        """Test initialization with various OpenAI model IDs."""
        model = OpenAIFoundationModel(
            model_id=model_id,
            params=None,
            client=mock_openai_client,
            user_message_text=valid_user_message_template,
            context_template_text=valid_context_template,
            system_message_text=valid_system_message,
        )
        assert model.model_id == model_id

    def test_chat_with_empty_response(self, model_with_dict_params, mock_openai_client):
        """Test chat method with empty response content."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = ""
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        response = model_with_dict_params.chat(messages)
        assert len(response) == 1
        assert response[0].message.content == ""
