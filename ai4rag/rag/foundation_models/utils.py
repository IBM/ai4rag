from typing import Literal
from string import Formatter

from ai4rag.search_space.src.model_props import (
    CONTEXT_TEXT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
)


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
        required_placeholders = (CONTEXT_TEXT_PLACEHOLDER,)
    elif template_name == "user_message_text":
        required_placeholders = (QUESTION_PLACEHOLDER, REFERENCE_DOCUMENTS_PLACEHOLDER)
    else:
        raise ValueError(f"Cannot validate presence of expected template placeholders on field: {template_name}")

    placeholders_count = 0

    for _, field_name, _, _ in Formatter().parse(template_str):
        if field_name is None:
            # when there is text NOT followed by a placeholder template
            continue
        if field_name not in required_placeholders:
            raise ValueError(
                f"Custom {field_name.split('_')[0]} template text got unexpected placeholder `{field_name}`, "
                f"valid placeholders are `{required_placeholders}`."
            )

        placeholders_count += 1

    if placeholders_count != len(required_placeholders):
        raise ValueError(
            f"Incorrect number of placeholders required for {template_name.split('_')[0]} template text, "
            f"expected {len(required_placeholders)} but got {placeholders_count}."
        )
    return template_str
