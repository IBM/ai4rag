# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from openai import OpenAI

from ai4rag.rag.foundation_models.base_model import BaseFoundationModel


class OpenAIFoundationModel(BaseFoundationModel):
    """Wrapper for OpenAI client handled foundation models."""

    def __init__(
        self,
        client: OpenAI,
        model_id: str,
        model_params: dict[str, Any] | None = None,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
    ):
        super().__init__(
            client=client,
            model_id=model_id,
            model_params=model_params,
            system_message_text=system_message_text,
            user_message_text=user_message_text,
            context_template_text=context_template_text,
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
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response_chat.choices[0].message.content

        return answer
