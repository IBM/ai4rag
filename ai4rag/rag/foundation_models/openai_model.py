# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated, Any

from annotated_types import Ge, Gt, Le
from openai import OpenAI
from pydantic import BaseModel

from ai4rag.rag.foundation_models.base_model import BaseFoundationModel, MessageTyped
from ai4rag.utils.constants import ChatGenerationConstants

# pylint: disable=duplicate-code


class OpenAIModelParameters(BaseModel):
    """Parameters to use for LSFoundationModel."""

    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class OpenAIFoundationModel(BaseFoundationModel[OpenAI, dict[str, Any] | OpenAIModelParameters | None]):
    """Wrapper for OpenAI client handled foundation models."""

    def __init__(
        self,
        client: OpenAI,
        model_id: str,
        params: dict[str, Any] | OpenAIModelParameters | None = None,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
    ):
        super().__init__(
            client=client,
            model_id=model_id,
            params=params,
            system_message_text=system_message_text,
            user_message_text=user_message_text,
            context_template_text=context_template_text,
        )

    @property
    def params(self) -> OpenAIModelParameters:
        return self._params

    @params.setter
    def params(self, params: dict | OpenAIModelParameters | None) -> None:
        if isinstance(params, dict):
            self._params = OpenAIModelParameters(**params)
        elif isinstance(params, OpenAIModelParameters):
            self._params = params
        else:
            self._params = OpenAIModelParameters()

    def chat(self, messages: list[MessageTyped]) -> list[MessageTyped]:
        """
        Chat completion for communication with selected foundation model.

        Parameters
        ----------
        messages : list[MessageTyped]
            Messages to be included in the chat completion.

        Returns
        -------
        list[MessageTyped]
            Chat response choices from the model.
        """
        response_chat = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        response_choices = response_chat.choices

        return response_choices
