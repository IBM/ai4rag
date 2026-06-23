# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from collections import Counter
from string import Formatter
from typing import Literal, TypeVar

from ai4rag.search_space.src.model_props import (
    CONTEXT_TEXT_PLACEHOLDER,
    DOCUMENT_NUMBER_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
)
from ai4rag.utils.validators import ConstraintsValidationError, OneOf, Validator

T = TypeVar("T")

_CONTEXT_TEMPLATE_ALLOWED_PLACEHOLDERS = frozenset(
    {CONTEXT_TEXT_PLACEHOLDER, DOCUMENT_NUMBER_PLACEHOLDER},
)


def _count_template_placeholders(template_str: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, field_name, _, _ in Formatter().parse(template_str):
        if field_name is not None:
            counts[field_name] += 1
    return counts


def _validate_context_template_placeholders(template_str: str) -> str:
    counts = _count_template_placeholders(template_str)

    for field_name in counts:
        if field_name not in _CONTEXT_TEMPLATE_ALLOWED_PLACEHOLDERS:
            raise ValueError(
                f"Custom context template text got unexpected placeholder `{field_name}`, "
                f"valid placeholders are `{tuple(_CONTEXT_TEMPLATE_ALLOWED_PLACEHOLDERS)}`."
            )

    if counts.get(CONTEXT_TEXT_PLACEHOLDER, 0) != 1:
        raise ValueError(
            "Incorrect number of placeholders required for context template text, "
            f"expected exactly one `{CONTEXT_TEXT_PLACEHOLDER}` placeholder but got "
            f"{counts.get(CONTEXT_TEXT_PLACEHOLDER, 0)}."
        )

    if counts.get(DOCUMENT_NUMBER_PLACEHOLDER, 0) > 1:
        raise ValueError(
            "Incorrect number of placeholders required for context template text, "
            f"expected at most one `{DOCUMENT_NUMBER_PLACEHOLDER}` placeholder but got "
            f"{counts[DOCUMENT_NUMBER_PLACEHOLDER]}."
        )

    return template_str


def _validate_user_message_template_placeholders(template_str: str) -> str:
    required_placeholders = (QUESTION_PLACEHOLDER, REFERENCE_DOCUMENTS_PLACEHOLDER)
    counts = _count_template_placeholders(template_str)

    for field_name in counts:
        if field_name not in required_placeholders:
            raise ValueError(
                f"Custom user template text got unexpected placeholder `{field_name}`, "
                f"valid placeholders are `{required_placeholders}`."
            )

    for field_name in required_placeholders:
        if counts.get(field_name, 0) != 1:
            raise ValueError(
                "Incorrect number of placeholders required for user template text, "
                f"expected 2 but got {sum(counts.values())}."
            )

    return template_str


class RAGPromptTemplateString(Validator[str]):
    """Validates RAG template string."""

    template_name: OneOf[Literal["context_template_text", "user_message_text"]] = OneOf(
        "context_template_text", "user_message_text"
    )

    def __init__(
        self,
        template_name: Literal["context_template_text", "user_message_text"],
    ) -> None:
        super().__init__()
        self.template_name = template_name

        self._required_placeholders: tuple[str, ...] = (
            (CONTEXT_TEXT_PLACEHOLDER,)
            if template_name == "context_template_text"
            else (QUESTION_PLACEHOLDER, REFERENCE_DOCUMENTS_PLACEHOLDER)
        )

    def validate(self, _: object, value: T) -> T:
        """
        Validates if user provided correct placeholders in given template text in respect to default placeholders.

        Parameters
        ----------
        template_str : str
            Prompt template with proper placeholders to be validated.

        template_name : Literal["context_template_text", "user_message_text"]
            Name of the template that will be validated. Used for required placeholders selection.

        Returns
        -------
        str
            Prompt template with filled placeholders.

        Raises
        ------
        ValueError
            When user provided less placeholders than expected.

            When user provided wrong placeholder name.
        """
        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be a str or None.")

        try:
            if self.template_name == "context_template_text":
                _validate_context_template_placeholders(value)
            else:
                _validate_user_message_template_placeholders(value)
        except ValueError as exc:
            raise ConstraintsValidationError(str(exc)) from exc

        return value


def _validate_prompt_templates_placeholders(
    template_str: str,
    template_name: Literal["context_template_text", "user_message_text"],
) -> str:
    """
    Validates if user provided correct placeholders in given template text in respect to default placeholders.

    Parameters
    ----------
    template_str : str
        Prompt template with proper placeholders to be validated.

    template_name : Literal["context_template_text", "user_message_text"]
        Name of the template that will be validated. Used for required placeholders selection.

    Returns
    -------
    str
        Prompt template with filled placeholders.

    Raises
    ------
    ValueError
        When user provided less placeholders than expected.

        When user provided wrong placeholder name.
    """
    if template_name == "context_template_text":
        return _validate_context_template_placeholders(template_str)
    if template_name == "user_message_text":
        return _validate_user_message_template_placeholders(template_str)

    raise ValueError(f"Cannot validate presence of expected template placeholders on field: {template_name}")
