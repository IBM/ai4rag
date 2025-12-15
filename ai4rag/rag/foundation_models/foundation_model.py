# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Literal, Annotated
from string import Formatter

from annotated_types import Gt, Le, Ge
from pydantic import BaseModel

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage

from ai4rag.search_space.src.model_props import (
    CONTEXT_TEXT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
    get_system_message_text,
    get_user_message_text,
    get_context_template_text,
)
from ai4rag.utils.constants import ChatGenerationConstants


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


class ModelParameters(BaseModel):
    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class FoundationModel(ABC):
    def __init__(self, model_id: str, model_params: dict[str, Any]):
        self.model_id = model_id
        self.model_params = model_params

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

    @abstractmethod
    def chat(self, system_message: str, user_message: str) -> str:
        """Docstring here"""


class LlamaStackFoundationModel(FoundationModel):
    """Integration point to use any model via Llama-stack API / client"""
    def __init__(
            self,
            model_id: str,
            model_params: dict[str, Any] | ModelParameters | None,
            ls_client: LlamaStackClient,
            **kwargs
    ):
        super().__init__(model_id, model_params)
        self._ls_client = ls_client
        self._model_params = model_params
        self._system_message_text = kwargs.pop("system_message_text", None)
        self._user_message_text = kwargs.pop("user_message_text", None)
        self._context_template_text = kwargs.pop("context_template_text", None)

    @property
    def ls_client(self) -> LlamaStackClient:
        return self._ls_client

    @ls_client.setter
    def ls_client(self, val: LlamaStackClient):
        if not isinstance(val, LlamaStackClient):
            raise TypeError(f"Expected instance of LlamaStackClient, got instance of {val.__class__.__name__} instead.")
        self._ls_client = val

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


    def chat(self, system_message: str, user_message: str) -> str:
        """
        Chat completion for communication with selected foundation model.

        Parameters
        ----------
        system_message : str
            System messages in the str format.

        user_message : str
            User message in the str format.

        Returns
        -------
        str
            Chat response from the model.
        """
        response_chat = self.ls_client.chat.completions.create(
            model=self.model_id,
            messages=[
                SystemMessage(role="system", content=system_message),
                UserMessage(role="user", content=user_message),
            ]
        )
        answer = response_chat.choices[0].message.content

        return answer
