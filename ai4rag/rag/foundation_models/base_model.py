# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Generic, TypedDict, TypeVar

from ai4rag.rag.foundation_models.utils import RAGPromptTemplateString
from ai4rag.search_space.src.model_props import (
    get_context_template_text,
    get_system_message_text,
    get_user_message_text,
)

FoundationModelClientT = TypeVar("FoundationModelClientT")
FoundationModelParamsT = TypeVar("FoundationModelParamsT")


class MessageTyped(TypedDict):
    """Type of the messages used by the client."""

    role: str
    content: str


class BaseFoundationModel(Generic[FoundationModelClientT, FoundationModelParamsT], ABC):
    """Interface definition for the foundation model used for `ai4rag`."""

    user_message_text: RAGPromptTemplateString = RAGPromptTemplateString(template_name="user_message_text")
    context_template_text: RAGPromptTemplateString = RAGPromptTemplateString(template_name="context_template_text")

    def __init__(
        self,
        client: FoundationModelClientT,
        model_id: str,
        params: FoundationModelParamsT,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
    ):
        self.client = client
        self.model_id = model_id
        self.params = params
        self.system_message_text = system_message_text or get_system_message_text(model_name=model_id)
        self.user_message_text = (
            user_message_text if user_message_text is not None else get_user_message_text(model_name=model_id)
        )
        self.context_template_text = (
            context_template_text
            if context_template_text is not None
            else get_context_template_text(model_name=model_id)
        )

    def __repr__(self) -> str:
        return self.model_id

    def __str__(self) -> str:
        return self.model_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseFoundationModel):
            return NotImplemented

        return self.model_id == other.model_id

    def __hash__(self):
        return hash(self.model_id)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, BaseFoundationModel):
            return NotImplemented
        return self.model_id < other.model_id

    @abstractmethod
    def chat(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
        """Chat with the model base on the client capabilities."""
