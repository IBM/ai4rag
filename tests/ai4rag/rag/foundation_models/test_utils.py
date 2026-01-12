# Copyright 2025- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import pytest

from ai4rag.rag.foundation_models.utils import _validate_prompt_templates_placeholders
from ai4rag.search_space.src.model_props import (
    CONTEXT_TEXT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
)


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
