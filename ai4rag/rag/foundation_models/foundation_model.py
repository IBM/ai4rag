# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage

from .base import FoundationModel, ModelParameters


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
