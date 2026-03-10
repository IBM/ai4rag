# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, fields
from math import floor
from typing import Any, Generic, Literal, Optional, Protocol, Sequence, TypeVar


def _get_hashable_repr(dct: dict):
    """
    Returns
    -------
    A hashable representation of the provided dictionary.
    """
    queue = deque((k, v, 0, None) for k, v in dct.items())
    dict_unpacked = []
    while queue:
        key, val, lvl, p_ref = queue.pop()
        if hasattr(val, "items"):  # we have a nested dict
            dict_unpacked.append((key, "+", lvl, p_ref))  # key is an aggregator at this level (that's why '+')
            if hash(key) != p_ref:  # but it could be an aggregator for a Sequence (and not other dict)
                lvl += 1
            queue.extendleft((k, v, lvl, hash(key)) for k, v in val.items())
        elif isinstance(val, Hashable):
            dict_unpacked.append((key, val, lvl, p_ref))
        elif isinstance(val, Sequence):
            # only sequences supported now
            dict_unpacked.append((key, "+", lvl, p_ref))

            queue.extendleft((key, vv, floor(lvl) + ind * 0.01, hash(key)) for ind, vv in enumerate(val, 1))

        else:
            raise ValueError(f"Some value in the provided dict is not supported. {type(val)} is not supported")

    return tuple(sorted(dict_unpacked, key=lambda it: (it[2], it[0])))


class HashableProtocol(Protocol):
    """Protocol definition for hashable types."""

    def __hash__(self) -> int: ...


HashableT = TypeVar("HashableT", bound=HashableProtocol)


class ParameterValueError(ValueError):
    """Error representing incorrect value given for parameter type."""


@dataclass(frozen=True)
class Parameter(Generic[HashableT]):
    """
    Representation of the general parameter used in optimization process.
    """

    name: str
    param_type: Literal["B", "I", "R", "C"] = "C"
    v_min: Optional[int | float] = None
    v_max: Optional[int | float] = None
    values: Optional[Sequence[HashableT]] = None

    def __post_init__(self):
        """Perform basic validations for Parameter initializations."""
        param_types = ["B", "I", "R", "C"]
        if self.param_type not in param_types:
            raise ParameterValueError(f"param_type must be one of: {param_types}")

        if self.param_type in ["B", "C"]:
            if not self.values:
                raise ParameterValueError("For Boolean or Categorical types, 'values' must be provided.")
            if self.v_min is not None or self.v_max is not None:
                raise ParameterValueError(
                    "For Boolean or Categorical types, minimum and maximum values are not allowed."
                )

        if self.param_type in ["I", "R"]:
            if self.v_min is None or self.v_max is None:
                raise ParameterValueError("For Integer or Real types, minimum and maximum values must be provided.")
            if self.values is not None:
                raise ParameterValueError("For Integer or Real types, values is not supported.")
            if self.param_type == "I":
                if not (isinstance(self.v_min, int) and isinstance(self.v_max, int)):
                    raise ParameterValueError(
                        "Parameter of type categorical ('C') has to specify both 'v_max' and 'v_min' "
                        "fields as int types."
                    )

    def _hashable_field_repr_categorical_values(self) -> tuple[str | int, ...] | None:
        """
        Parameter with `param_type="C"`allow specifying a list of (possible nested) objects as available values.
        In order to reliably compare such Parameter instances between each other and ensure hashability, this
        utility function exists.
        It may seem to be computationally heavy, but for typical input its execution time still lies in
        the order of magnitude of 100ths of a milisecond.

        Returns
        -------
        tuple
            For values being a list of strings:
                sorted (string-wise, alphabetically) tuple of members in `self.values` field.
            For values being a list of dictionaries:
                a list of sorted hashes calculated for the list members
        """
        if not self.values:
            return None
        if isinstance(self.values[0], (str, bool, int, float)):
            return tuple(sorted(self.values))
        if isinstance(self.values[0], dict):
            return tuple(sorted((hash(dct_hash_repr) for dct_hash_repr in map(_get_hashable_repr, self.values))))
        # tmp workaround for param.name='inference_model_id'
        try:
            return tuple(sorted(map(hash, self.values)))
        except TypeError:
            raise ParameterValueError(
                f"Trying to get a hashable representation of values field containing "
                f"unsupported types: {type(self.values[0])}."
            ) from None

    def all_values(self) -> Sequence:
        """Get all possible values that parameter can take. It is not applicable for R."""
        match self.param_type:
            case "C":
                return self.values
            case "B":
                return [True, False]
            case "I":
                return list(range(self.v_min, self.v_max + 1))
            case _:
                raise ParameterValueError("Cannot specify possible values for R type.")

    def __eq__(self, other: Any) -> bool:
        """Parameter objects are considered equal when their fields are the same (name- and value-wise)."""
        if not isinstance(other, Parameter):
            return False

        if self._hashable_field_repr_categorical_values() != other._hashable_field_repr_categorical_values():
            return False

        for f in fields(self):
            if f.name == "values":  # already compared above
                continue
            if getattr(self, f.name) != getattr(other, f.name):
                return False

        return True

    def __hash__(self) -> int:
        if self.values:
            fields_no_values = [getattr(self, f.name) for f in fields(self) if f.name != "values"]
            return hash((*fields_no_values, self._hashable_field_repr_categorical_values()))
        return hash(tuple(getattr(self, f.name) for f in fields(self)))
