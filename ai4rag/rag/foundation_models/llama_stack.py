# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated, Any

from annotated_types import Ge, Gt, Le
from llama_stack_client import LlamaStackClient
from pydantic import BaseModel

from ai4rag.rag.foundation_models.base_model import BaseFoundationModel, MessageTyped
from ai4rag.utils.constants import ChatGenerationConstants

# pylint: disable=duplicate-code


class LSModelParameters(BaseModel):
    """Parameters to use for LSFoundationModel."""

    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class LSFoundationModel(BaseFoundationModel[LlamaStackClient, dict[str, Any] | LSModelParameters | None]):
    """Integration point to use any model via Llama-stack API / client"""

    def __init__(
        self,
        client: LlamaStackClient,
        model_id: str,
        params: dict[str, Any] | LSModelParameters | None = None,
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
    def params(self) -> LSModelParameters:
        """Get models params."""
        return self._params

    @params.setter
    def params(self, params: dict | LSModelParameters | None) -> None:
        """Set models params."""
        if isinstance(params, dict):
            self._params = LSModelParameters(**params)
        elif isinstance(params, LSModelParameters):
            self._params = params
        else:
            self._params = LSModelParameters()

    def chat(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
        """
        Chat completion for communication with selected foundation model.

        Parameters
        ----------
        messages : list[MessageTyped]
            Messages to be included in the chat completion.

        **kwargs
            Additional parameters forwarded to the chat completions API
            (e.g. ``response_format``).

        Returns
        -------
        str
            Chat response from the model.
        """

        chat_params = {
            "max_completion_tokens": self.params.max_completion_tokens,
            "temperature": self.params.temperature,
        }
        updated_chat_params = chat_params | kwargs

        response_chat = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **updated_chat_params,
        )
        response_choices = response_chat.choices

        return response_choices
