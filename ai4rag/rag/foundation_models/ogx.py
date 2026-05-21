# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated, Any

from annotated_types import Ge, Gt, Le
from ogx_client import OgxClient
from pydantic import BaseModel

from ai4rag.rag.foundation_models.base_model import BaseFoundationModel, MessageTyped
from ai4rag.utils.constants import ChatGenerationConstants

# pylint: disable=duplicate-code


class OGXModelParameters(BaseModel):
    """Parameters to use for OGXFoundationModel."""

    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class OGXFoundationModel(BaseFoundationModel[OgxClient, dict[str, Any] | OGXModelParameters | None]):
    """Integration point to use any model via OGX API / client"""

    def __init__(
        self,
        client: OgxClient,
        model_id: str,
        params: dict[str, Any] | OGXModelParameters | None = None,
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
    def params(self) -> OGXModelParameters:
        """Get models params."""
        return self._params

    @params.setter
    def params(self, params: dict | OGXModelParameters | None) -> None:
        """Set models params."""
        if isinstance(params, dict):
            self._params = OGXModelParameters(**params)
        elif isinstance(params, OGXModelParameters):
            self._params = params
        else:
            self._params = OGXModelParameters()

    def chat(self, messages: list[MessageTyped]) -> list[MessageTyped]:
        """
        Chat completion for communication with selected foundation model.

        Parameters
        ----------
        messages : list[MessageTyped]
            Messages to be included in the chat completion.

        Returns
        -------
        str
            Chat response from the model.
        """
        response_chat = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=self.params.max_completion_tokens,
            temperature=self.params.temperature,
        )
        response_choices = response_chat.choices

        return response_choices

    def create_response(self, user_message: str, vector_store_id: str | None = None) -> str:
        """
        Utilise Responses API (agent loop) to interact with the model.

        Parameters
        ----------
        user_message : str
            User message for the model to answer.

        vector_store_id : str | None
            If provided then references the vector store to search against using
            the built-int `file_search` tool.

        Returns
        -------
        str
            Response text from the model.

        Notes
        -----
        For the time being the only supported input type is a string representing the user's message.
        For more input types please refer to:
        https://developers.openai.com/api/reference/resources/responses/methods/create
        """

        tools = None
        if vector_store_id:
            tools = [{"type": "file_search", "vector_store_ids": vector_store_id}]

        response = self.client.responses.create(
            model=self.model_id,
            instructions=self.system_message_text,
            input=user_message,
            max_completion_tokens=self.params.max_completion_tokens,
            temperature=self.params.temperature,
            tools=tools,
        )

        return response.output_text
