# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypedDict, TypeVar

from ai4rag import logger
from ai4rag.rag.foundation_models.utils import validate_prompt_templates_placeholders
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


@dataclass
class Language:
    """Settings for multilingual handling."""

    code: str
    name: str

    def to_dict(self) -> dict:
        """Save language settings as a dict."""
        return {"code": self.code, "name": self.name}


class BaseFoundationModel(Generic[FoundationModelClientT, FoundationModelParamsT], ABC):
    """Interface definition for the foundation model used for `ai4rag`."""

    def __init__(
        self,
        client: FoundationModelClientT,
        model_id: str,
        params: FoundationModelParamsT,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
        language: Language | None = None,
    ):
        language = language or Language(code="", name="auto")
        self.client = client
        self.model_id = model_id
        self.params = params
        self.system_message_text = system_message_text or get_system_message_text(model_name=model_id)
        self.user_message_text = user_message_text or get_user_message_text(model_name=model_id, language=language.name)
        self.context_template_text = context_template_text or get_context_template_text()
        self._language = language

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

    @property
    def language(self):
        """Get language settings."""
        return self._language

    @language.setter
    def language(self, value: Language):
        """Set language settings and regenerate user_message_text for the new language."""
        self._language = value
        self.user_message_text = get_user_message_text(model_name=self.model_id, language=value.name)
        logger.info("Model %s: user_message_text regenerated for language '%s'.", self.model_id, value.name)

    @property
    def user_message_text(self):
        """Get user_message_text."""
        return self._user_message_text

    @user_message_text.setter
    def user_message_text(self, value: str):
        """Set user_message_text and validate template."""
        self._user_message_text = validate_prompt_templates_placeholders(
            template_str=value, template_name="user_message_text"
        )

    @property
    def context_template_text(self):
        """Get context_template_text."""
        return self._context_template_text

    @context_template_text.setter
    def context_template_text(self, value: str):
        """Set context_template_text and validate template."""
        self._context_template_text = validate_prompt_templates_placeholders(
            template_str=value, template_name="context_template_text"
        )

    @abstractmethod
    def chat(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
        """Chat with the model base on the client capabilities."""
