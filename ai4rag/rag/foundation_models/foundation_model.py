# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, Annotated
from annotated_types import Gt, Le, Ge
from pydantic import BaseModel

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage

from ai4rag.utils.constants import ChatGenerationConstants
from ai4rag.rag.foundation_models.base_model import FoundationModel
from ai4rag.utils.validators import RAGPromptTemplateString
from ai4rag.search_space.src.model_props import (
    get_system_message_text,
    get_user_message_text,
)


class ModelParameters(BaseModel):
    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class LlamaStackFoundationModel(FoundationModel[LlamaStackClient, dict[str, Any] | ModelParameters | None]):
    """Integration point to use any model via Llama-stack API / client"""

    user_message_text: RAGPromptTemplateString = RAGPromptTemplateString(template_name="user_message_text")
    context_template_text: RAGPromptTemplateString = RAGPromptTemplateString(template_name="context_template_text")

    def __init__(
        self,
        model_id: str,
        model_params: dict[str, Any] | ModelParameters | None,
        client: LlamaStackClient,
        user_message_text: str,
        context_template_text: str,
        system_message_text: str,
    ):
        super().__init__(client=client, model_id=model_id, model_params=model_params)
        self.model_params = model_params
        self.system_message_text = system_message_text
        self.user_message_text = (
            user_message_text if user_message_text is not None else get_user_message_text(model_name=model_id)
        )
        self.context_template_text = (
            context_template_text if context_template_text is not None else get_system_message_text(model_name=model_id)
        )

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
        response_chat = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                SystemMessage(role="system", content=system_message),
                UserMessage(role="user", content=user_message),
            ],
        )
        answer = response_chat.choices[0].message.content

        return answer
