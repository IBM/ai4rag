# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, Annotated

from annotated_types import Gt, Le, Ge
from pydantic import BaseModel

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage

from ai4rag.search_space.src.model_props import (
    get_system_message_text,
    get_user_message_text,
)
from ai4rag.utils.constants import ChatGenerationConstants

from .base import FoundationModel
from .utils import _validate_prompt_templates_placeholders


class ModelParameters(BaseModel):
    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class LlamaStackFoundationModel(FoundationModel):
    """Integration point to use any model via Llama-stack API / client"""

    def __init__(
        self,
        model_id: str,
        model_params: dict[str, Any] | ModelParameters | None,
        ls_client: LlamaStackClient,
        **kwargs,
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
            ],
        )
        answer = response_chat.choices[0].message.content

        return answer
