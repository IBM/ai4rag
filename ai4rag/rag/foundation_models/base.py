# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Annotated
from annotated_types import Ge, Le, Gt

from pydantic import BaseModel

from ai4rag.rag.foundation_models.utils import _validate_prompt_templates_placeholders
from ai4rag.search_space.src.model_props import get_system_message_text, get_user_message_text
from ai4rag.utils.constants import ChatGenerationConstants


class ModelParameters(BaseModel):
    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE

    def to_dict(self) -> dict[str, Any]:
        """Return parameters as dict."""
        return vars(self)


class FoundationModel(ABC):
    def __init__(self, model_id: str, model_params: dict[str, Any], **kwargs):
        self.model_id = model_id
        self.model_params = model_params
        self._system_message_text = kwargs.pop("system_message_text", None)
        self._user_message_text = kwargs.pop("user_message_text", None)
        self._context_template_text = kwargs.pop("context_template_text", None)

    def __repr__(self) -> str:
        return self.model_id

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FoundationModel):
            return NotImplemented

        return self.model_id == other.model_id

    def __hash__(self):
        return hash(self.model_id)

    @property
    def model_params(self) -> ModelParameters:
        return self._model_params

    @model_params.setter
    def model_params(self, val: dict[str, Any] | ModelParameters | None) -> None:
        if val is None:
            self._model_params = ModelParameters()
        elif isinstance(val, ModelParameters):
            self._model_params = val
        elif isinstance(val, dict):
            self._model_params = ModelParameters(**val)
        else:
            raise TypeError(f"Expected ModelParameters | dict | None, got {type(val)} instead.")

    @property
    def system_message_text(self) -> str:
        return self._system_message_text

    @system_message_text.setter
    def system_message_text(self, val: str | None) -> None:
        if val is None:
            self._system_message_text = get_system_message_text(self.model_id)
        else:
            self._system_message_text = val

    @property
    def user_message_text(self) -> str:
        return self._user_message_text

    @user_message_text.setter
    def user_message_text(self, val: str | None) -> None:
        if val is None:
            self._user_message_text = get_user_message_text(self.model_id)
        else:
            self._user_message_text = _validate_prompt_templates_placeholders(val, "user_message_text")

    @abstractmethod
    def chat(self, system_message: str, user_message: str) -> str:
        """Docstring here"""
