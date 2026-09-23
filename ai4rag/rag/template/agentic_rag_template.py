# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

from ..foundation_models.base_model import MessageTyped
from .base_template import BaseRAGTemplate


class AgenticRAG(BaseRAGTemplate):
    """Single-step agentic RAG template.

    The first version intentionally performs one retrieval for the current
    user turn and one foundation-model call.  It is kept as a separate
    template so the retrieval/generation policy can later grow to include
    query rewriting or additional retrieval steps without changing the
    experiment and starter-kit interfaces.
    """

    def generate(self, question: str, **kwargs) -> dict[str, Any]:
        """Generate an answer after one retrieval step."""
        reference_documents, user_message = self._build_enriched_user_message(question, **kwargs)
        response = self.foundation_model.chat(
            messages=[
                {"role": "system", "content": self.foundation_model.system_message_text},
                {"role": "user", "content": user_message},
            ]
        )
        return {
            "answer": response[0].message.content,
            "reference_documents": reference_documents,
            "question": question,
        }

    def generate_stream(self, question: str, **kwargs):
        """Yield the complete answer as one chunk until streaming is supported."""
        yield self.generate(question, **kwargs)["answer"]

    def respond(self, messages: list[MessageTyped]):
        """Retrieve context and return a native Responses API response."""
        rag_messages = self._build_rag_messages(messages)
        return self.foundation_model.responses(messages=rag_messages)

    def chat(self, messages: list[MessageTyped], **kwargs) -> list[Any]:
        """Run one-step retrieval for the last user message in the conversation."""
        rag_messages = self._build_rag_messages(messages, **kwargs)
        return self.foundation_model.chat(messages=rag_messages, **kwargs)
