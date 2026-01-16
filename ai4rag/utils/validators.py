# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------


from abc import ABC, abstractmethod
from string import Formatter
from typing import Any, Protocol, Literal, TypeVar, Generic, Self, overload

from ai4rag.search_space.src.model_props import (
    CONTEXT_TEXT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
)


T = TypeVar("T")
CT = TypeVar("CT", bound="Comparable")


class ConstraintsValidationError(Exception):
    """Error raised when validation has failed."""


class Comparable(Protocol):
    """Protocol which ensures that the type is comparable."""

    def __eq__(self, other: Self, /) -> bool: ...
    def __lt__(self, other: Self, /) -> bool: ...
    def __le__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...


class Validator(Generic[T], ABC):
    """Base validator class."""

    def __init__(self):
        self.private_name: str

    def __set_name__(self, _, name):
        self.private_name = f"_{name}"

    def __set__(self, obj, value: T):
        validated_value = self.validate(obj, value)
        setattr(obj, self.private_name, validated_value)

    @overload
    def __get__(self, instance: None, owner: Any) -> Self: ...

    @overload
    def __get__(self, instance: Any, owner: Any) -> T: ...

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name)

    @abstractmethod
    def validate(self, obj, value):
        """Base validate method."""


class OneOf(Validator[T]):
    """Validates if given value is within provided set of values."""

    def __init__(self, *options: T):
        super().__init__()
        self.options = set(options)

    def validate(self, obj, value: T):
        if value not in self.options:
            raise ConstraintsValidationError(
                f"Expected {value!r} to be one of {self.options!r} for {type(obj).__name__} on attribute {self.private_name[1:]}"
            )
        return value


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
            if template_name == "context_tepmlate_text"
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

        placeholders_count = 0

        for _, field_name, _, _ in Formatter().parse(value):
            if field_name is None:
                # when there is text NOT followed by a placeholder template
                continue
            if field_name not in self._required_placeholders:
                raise ConstraintsValidationError(
                    f"Custom {field_name.split('_')[0]} template text got unexpected placeholder `{field_name}`, "
                    f"valid placeholders are `{self._required_placeholders}`."
                )

            placeholders_count += 1

        if placeholders_count != len(self._required_placeholders):
            raise ConstraintsValidationError(
                f"Incorrect number of placeholders required for {value.split('_')[0]} template text, "
                f"expected {len(self._required_placeholders)} but got {placeholders_count}."
            )

        return value
