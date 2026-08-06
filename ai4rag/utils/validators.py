# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, Self, TypeVar, overload

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
                f"Expected {value!r} to be one of {self.options!r} for {type(obj).__name__} "
                f"on attribute {self.private_name[1:]}."
            )
        return value


def validate_model_list(models: list[str] | None, name: str) -> None:
    """Validate that a model list, if provided, contains only non-empty strings."""
    if models is None:
        return
    if not isinstance(models, list):
        raise TypeError(f"{name} must be a list.")
    for i, m in enumerate(models):
        if not m:
            raise TypeError(f"{name}[{i}] must be a non-empty string.")
