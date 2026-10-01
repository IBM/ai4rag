# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated, Any

from annotated_types import Ge, Gt, Le
from openai import APITimeoutError, OpenAI
from pydantic import BaseModel

from ai4rag import logger
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel, Language, MessageTyped
from ai4rag.utils.constants import ChatGenerationConstants

_FALLBACK_TIMEOUT = 1200.0


class OpenAIModelParameters(BaseModel):
    """Parameters to use for OpenAIFoundationModel."""

    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class OpenAIFoundationModel(BaseFoundationModel[OpenAI, dict[str, Any] | OpenAIModelParameters | None]):
    """Integration point to use any model via an OpenAI-compatible API / client."""

    def __init__(
        self,
        client: OpenAI,
        model_id: str,
        params: dict[str, Any] | OpenAIModelParameters | None = None,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
        language: Language | None = None,
    ):
        super().__init__(
            client=client,
            model_id=model_id,
            params=params,
            system_message_text=system_message_text,
            user_message_text=user_message_text,
            context_template_text=context_template_text,
            language=language,
        )

    @property
    def params(self) -> OpenAIModelParameters:
        """Get models params."""
        return self._params

    @params.setter
    def params(self, params: dict | OpenAIModelParameters | None) -> None:
        """Set models params."""
        if isinstance(params, dict):
            self._params = OpenAIModelParameters(**params)
        elif isinstance(params, OpenAIModelParameters):
            self._params = params
        else:
            self._params = OpenAIModelParameters()

    def chat(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
        """Chat completion for communication with selected foundation model.

        On ``APITimeoutError``, retries once with a 20-minute timeout
        and no client-level retries to accommodate slow (CPU-deployed)
        models.

        Parameters
        ----------
        messages : list[MessageTyped]
            Messages to be included in the chat completion.

        Returns
        -------
        list[MessageTyped]
            Chat response choices from the model.
        """
        chat_params = {
            "max_completion_tokens": self.params.max_completion_tokens,
            "temperature": self.params.temperature,
        } | kwargs

        try:
            return self.client.chat.completions.create(model=self.model_id, messages=messages, **chat_params).choices
        except APITimeoutError:
            logger.warning(
                "Chat request timed out. Retrying with %.0fs timeout (no retries).",
                _FALLBACK_TIMEOUT,
            )

            no_retries_client = self.client.with_options(timeout=_FALLBACK_TIMEOUT, max_retries=0)
            return no_retries_client.chat.completions.create(
                model=self.model_id, messages=messages, **chat_params
            ).choices
