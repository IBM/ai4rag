# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

from ai4rag.rag.chunking.chunk import AI4RAGChunk

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

    def _build_enriched_user_message(self, question: str, **kwargs) -> tuple[list[AI4RAGChunk], str]:
        """Retrieve context and render the optimized user prompt."""
        reference_documents = self.retriever.retrieve(question, **kwargs)
        context = "\n\n".join(
            self.foundation_model.context_template_text.format(document=chunk.text, doc_number=doc_number)
            for doc_number, chunk in enumerate(reference_documents, start=1)
        )
        user_message = self.foundation_model.user_message_text.format(
            reference_documents=context,
            question=question,
        )
        return reference_documents, user_message

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

    def chat(self, messages: list[MessageTyped], **kwargs) -> list[Any]:
        """Run one-step retrieval for the last user message in the conversation."""
        if not messages:
            raise ValueError("`messages` must contain at least one message.")

        *history, last_message = messages
        _, enriched_content = self._build_enriched_user_message(last_message["content"], **kwargs)
        rag_messages = [
            {"role": "system", "content": self.foundation_model.system_message_text},
            *history,
            {**last_message, "content": enriched_content},
        ]
        return self.foundation_model.chat(messages=rag_messages, **kwargs)
