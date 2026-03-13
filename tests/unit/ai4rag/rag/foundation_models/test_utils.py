# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.rag.foundation_models.utils import (
    RAGPromptTemplateString,
    _validate_prompt_templates_placeholders,
)
from ai4rag.search_space.src.model_props import (
    CONTEXT_TEXT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
)
from ai4rag.utils.validators import ConstraintsValidationError


class TestValidatePromptTemplatesPlaceholders:
    """Test suite for _validate_prompt_templates_placeholders function."""

    def test_valid_context_template_text(self):
        """Test validation of valid context template text with correct placeholder."""
        template = f"Here is the context: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        result = _validate_prompt_templates_placeholders(template, "context_template_text")
        assert result == template

    def test_valid_user_message_text(self):
        """Test validation of valid user message text with both required placeholders."""
        template = f"Question: {{{QUESTION_PLACEHOLDER}}}\n" f"References: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}"
        result = _validate_prompt_templates_placeholders(template, "user_message_text")
        assert result == template

    def test_valid_user_message_text_different_order(self):
        """Test validation when placeholders are in different order."""
        template = f"References: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n" f"Question: {{{QUESTION_PLACEHOLDER}}}"
        result = _validate_prompt_templates_placeholders(template, "user_message_text")
        assert result == template

    def test_invalid_template_name(self):
        """Test that invalid template name raises ValueError."""
        template = "Some template text"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "invalid_template_name")
        assert "Cannot validate presence of expected template placeholders" in str(exc_info.value)
        assert "invalid_template_name" in str(exc_info.value)

    def test_context_template_missing_placeholder(self):
        """Test that context template without placeholder raises ValueError."""
        template = "This is just text without any placeholder"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "context_template_text")
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 1 but got 0" in str(exc_info.value)

    def test_user_message_missing_one_placeholder(self):
        """Test that user message with only one placeholder raises ValueError."""
        template = f"Question: {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "user_message_text")
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 2 but got 1" in str(exc_info.value)

    def test_user_message_missing_both_placeholders(self):
        """Test that user message without placeholders raises ValueError."""
        template = "This is a template without any placeholders"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "user_message_text")
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 2 but got 0" in str(exc_info.value)

    def test_context_template_wrong_placeholder(self):
        """Test that context template with wrong placeholder raises ValueError."""
        template = f"Context: {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "context_template_text")
        assert "got unexpected placeholder" in str(exc_info.value)
        assert QUESTION_PLACEHOLDER in str(exc_info.value)

    def test_user_message_wrong_placeholder(self):
        """Test that user message with wrong placeholder raises ValueError."""
        template = f"Question: {{{QUESTION_PLACEHOLDER}}}\nContext: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "user_message_text")
        assert "got unexpected placeholder" in str(exc_info.value)
        assert CONTEXT_TEXT_PLACEHOLDER in str(exc_info.value)

    def test_context_template_extra_placeholder(self):
        """Test that context template with extra placeholder raises ValueError."""
        template = f"Context: {{{CONTEXT_TEXT_PLACEHOLDER}}} and {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "context_template_text")
        assert "got unexpected placeholder" in str(exc_info.value)

    def test_user_message_extra_placeholder(self):
        """Test that user message with extra placeholder raises ValueError."""
        template = (
            f"Question: {{{QUESTION_PLACEHOLDER}}}\n"
            f"References: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n"
            f"Context: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        )
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "user_message_text")
        assert "got unexpected placeholder" in str(exc_info.value)

    def test_template_with_only_text(self):
        """Test template with text but an invalid placeholder."""
        template = "This is text with {not_a_real_placeholder}"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "context_template_text")
        assert "unexpected placeholder" in str(exc_info.value)

    def test_context_template_duplicate_placeholder(self):
        """Test that context template with duplicate placeholder raises ValueError."""
        template = f"First: {{{CONTEXT_TEXT_PLACEHOLDER}}} Second: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "context_template_text")
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 1 but got 2" in str(exc_info.value)

    def test_user_message_duplicate_placeholder(self):
        """Test that user message with duplicate placeholder raises ValueError."""
        template = (
            f"Q1: {{{QUESTION_PLACEHOLDER}}}\n"
            f"Q2: {{{QUESTION_PLACEHOLDER}}}\n"
            f"Refs: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}"
        )
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "user_message_text")
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 2 but got 3" in str(exc_info.value)

    def test_empty_template_string(self):
        """Test validation with empty template string."""
        template = ""
        with pytest.raises(ValueError) as exc_info:
            _validate_prompt_templates_placeholders(template, "context_template_text")
        assert "Incorrect number of placeholders" in str(exc_info.value)

    @pytest.mark.parametrize(
        "template_name,template,expected_result",
        [
            (
                "context_template_text",
                f"Doc: {{{CONTEXT_TEXT_PLACEHOLDER}}}",
                f"Doc: {{{CONTEXT_TEXT_PLACEHOLDER}}}",
            ),
            (
                "user_message_text",
                f"Q: {{{QUESTION_PLACEHOLDER}}} R: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}",
                f"Q: {{{QUESTION_PLACEHOLDER}}} R: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}",
            ),
        ],
    )
    def test_valid_templates_parametrized(self, template_name, template, expected_result):
        """Parameterized test for valid templates."""
        result = _validate_prompt_templates_placeholders(template, template_name)
        assert result == expected_result


class TestRAGPromptTemplateString:
    """Test suite for RAGPromptTemplateString validator class."""

    def test_initialization_context_template(self):
        """Test validator initialization for context template."""
        validator = RAGPromptTemplateString("context_template_text")
        assert validator.template_name == "context_template_text"
        assert validator._required_placeholders == (CONTEXT_TEXT_PLACEHOLDER,)

    def test_initialization_user_message(self):
        """Test validator initialization for user message template."""
        validator = RAGPromptTemplateString("user_message_text")
        assert validator.template_name == "user_message_text"
        assert validator._required_placeholders == (QUESTION_PLACEHOLDER, REFERENCE_DOCUMENTS_PLACEHOLDER)

    def test_valid_context_template_text(self):
        """Test validation of valid context template text with correct placeholder."""
        validator = RAGPromptTemplateString("context_template_text")
        template = f"Here is the context: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        result = validator.validate(None, template)
        assert result == template

    def test_valid_user_message_text(self):
        """Test validation of valid user message text with both required placeholders."""
        validator = RAGPromptTemplateString("user_message_text")
        template = f"Question: {{{QUESTION_PLACEHOLDER}}}\nReferences: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}"
        result = validator.validate(None, template)
        assert result == template

    def test_valid_user_message_different_order(self):
        """Test validation when placeholders are in different order."""
        validator = RAGPromptTemplateString("user_message_text")
        template = f"References: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\nQuestion: {{{QUESTION_PLACEHOLDER}}}"
        result = validator.validate(None, template)
        assert result == template

    def test_type_error_on_non_string_value(self):
        """Test that non-string value raises TypeError."""
        validator = RAGPromptTemplateString("context_template_text")
        with pytest.raises(TypeError) as exc_info:
            validator.validate(None, 123)
        assert "Expected 123 to be a str or None" in str(exc_info.value)

    def test_type_error_on_none_value(self):
        """Test that None value raises TypeError."""
        validator = RAGPromptTemplateString("context_template_text")
        with pytest.raises(TypeError) as exc_info:
            validator.validate(None, None)
        assert "Expected None to be a str or None" in str(exc_info.value)

    def test_type_error_on_list_value(self):
        """Test that list value raises TypeError."""
        validator = RAGPromptTemplateString("user_message_text")
        with pytest.raises(TypeError) as exc_info:
            validator.validate(None, ["template"])
        assert "to be a str or None" in str(exc_info.value)

    def test_context_template_missing_placeholder(self):
        """Test that context template without placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("context_template_text")
        template = "This is just text without any placeholder"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 1 but got 0" in str(exc_info.value)

    def test_user_message_missing_one_placeholder(self):
        """Test that user message with only one placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("user_message_text")
        template = f"Question: {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 2 but got 1" in str(exc_info.value)

    def test_user_message_missing_both_placeholders(self):
        """Test that user message without placeholders raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("user_message_text")
        template = "This is a template without any placeholders"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 2 but got 0" in str(exc_info.value)

    def test_context_template_wrong_placeholder(self):
        """Test that context template with wrong placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("context_template_text")
        template = f"Context: {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "got unexpected placeholder" in str(exc_info.value)
        assert QUESTION_PLACEHOLDER in str(exc_info.value)

    def test_user_message_wrong_placeholder(self):
        """Test that user message with wrong placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("user_message_text")
        template = f"Question: {{{QUESTION_PLACEHOLDER}}}\nContext: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "got unexpected placeholder" in str(exc_info.value)
        assert CONTEXT_TEXT_PLACEHOLDER in str(exc_info.value)

    def test_context_template_extra_placeholder(self):
        """Test that context template with extra placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("context_template_text")
        template = f"Context: {{{CONTEXT_TEXT_PLACEHOLDER}}} and {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "got unexpected placeholder" in str(exc_info.value)

    def test_user_message_extra_placeholder(self):
        """Test that user message with extra placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("user_message_text")
        template = (
            f"Question: {{{QUESTION_PLACEHOLDER}}}\n"
            f"References: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n"
            f"Context: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        )
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "got unexpected placeholder" in str(exc_info.value)

    def test_template_with_invalid_placeholder(self):
        """Test template with text but an invalid placeholder."""
        validator = RAGPromptTemplateString("context_template_text")
        template = "This is text with {not_a_real_placeholder}"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "unexpected placeholder" in str(exc_info.value)

    def test_context_template_duplicate_placeholder(self):
        """Test that context template with duplicate placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("context_template_text")
        template = f"First: {{{CONTEXT_TEXT_PLACEHOLDER}}} Second: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 1 but got 2" in str(exc_info.value)

    def test_user_message_duplicate_placeholder(self):
        """Test that user message with duplicate placeholder raises ConstraintsValidationError."""
        validator = RAGPromptTemplateString("user_message_text")
        template = (
            f"Q1: {{{QUESTION_PLACEHOLDER}}}\n"
            f"Q2: {{{QUESTION_PLACEHOLDER}}}\n"
            f"Refs: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}"
        )
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "Incorrect number of placeholders" in str(exc_info.value)
        assert "expected 2 but got 3" in str(exc_info.value)

    def test_empty_template_string(self):
        """Test validation with empty template string."""
        validator = RAGPromptTemplateString("context_template_text")
        template = ""
        with pytest.raises(ConstraintsValidationError) as exc_info:
            validator.validate(None, template)
        assert "Incorrect number of placeholders" in str(exc_info.value)

    def test_descriptor_usage_context_template(self):
        """Test validator as descriptor on a class for context template."""

        class TestClass:
            context_template = RAGPromptTemplateString("context_template_text")

            def __init__(self, template):
                self.context_template = template

        # Valid template
        valid_template = f"Context: {{{CONTEXT_TEXT_PLACEHOLDER}}}"
        obj = TestClass(valid_template)
        assert obj.context_template == valid_template

        # Invalid template should raise error
        invalid_template = f"Question: {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ConstraintsValidationError):
            TestClass(invalid_template)

    def test_descriptor_usage_user_message(self):
        """Test validator as descriptor on a class for user message template."""

        class TestClass:
            user_message = RAGPromptTemplateString("user_message_text")

            def __init__(self, template):
                self.user_message = template

        # Valid template
        valid_template = f"Q: {{{QUESTION_PLACEHOLDER}}} R: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}"
        obj = TestClass(valid_template)
        assert obj.user_message == valid_template

        # Invalid template should raise error
        invalid_template = f"Only question: {{{QUESTION_PLACEHOLDER}}}"
        with pytest.raises(ConstraintsValidationError):
            TestClass(invalid_template)

    @pytest.mark.parametrize(
        "template_name,template,expected_result",
        [
            (
                "context_template_text",
                f"Doc: {{{CONTEXT_TEXT_PLACEHOLDER}}}",
                f"Doc: {{{CONTEXT_TEXT_PLACEHOLDER}}}",
            ),
            (
                "context_template_text",
                f"[Document]\n{{{CONTEXT_TEXT_PLACEHOLDER}}}\n[End]",
                f"[Document]\n{{{CONTEXT_TEXT_PLACEHOLDER}}}\n[End]",
            ),
            (
                "user_message_text",
                f"Q: {{{QUESTION_PLACEHOLDER}}} R: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}",
                f"Q: {{{QUESTION_PLACEHOLDER}}} R: {{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}",
            ),
            (
                "user_message_text",
                f"\n\nContext:\n{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}:\n\nQuestion: {{{QUESTION_PLACEHOLDER}}}. \n",
                f"\n\nContext:\n{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}:\n\nQuestion: {{{QUESTION_PLACEHOLDER}}}. \n",
            ),
        ],
    )
    def test_valid_templates_parametrized(self, template_name, template, expected_result):
        """Parameterized test for valid templates."""
        validator = RAGPromptTemplateString(template_name)
        result = validator.validate(None, template)
        assert result == expected_result

    @pytest.mark.parametrize(
        "template_name,invalid_value",
        [
            ("context_template_text", 123),
            ("context_template_text", None),
            ("context_template_text", []),
            ("context_template_text", {}),
            ("user_message_text", 123),
            ("user_message_text", None),
            ("user_message_text", []),
            ("user_message_text", {}),
        ],
    )
    def test_type_error_parametrized(self, template_name, invalid_value):
        """Parameterized test for type errors."""
        validator = RAGPromptTemplateString(template_name)
        with pytest.raises(TypeError):
            validator.validate(None, invalid_value)
