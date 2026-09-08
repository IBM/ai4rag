# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

from ai4rag.rag.chunking.chunk import AI4RAGChunk

from ..foundation_models.base_model import MessageTyped
from .base_template import BaseRAGTemplate


class SimpleRAG(BaseRAGTemplate):
    """
    RAG template composing a retriever and a foundation model for retrieval
    and generation.

    Parameters
    ----------
    foundation_model : BaseFoundationModel
        Initialized foundation model for text generation.

    retriever : Retriever
        Initialized retriever for document retrieval.
    """

    def _build_enriched_user_message(self, question: str, **kwargs) -> tuple[list[AI4RAGChunk], str]:
        """
        Retrieve context for `question` and render the RAG-enriched user message.

        Parameters
        ----------
        question : str
            The question used as the retrieval query.

        **kwargs
            Additional parameters forwarded to the retriever (e.g. number_of_chunks).

        Returns
        -------
        tuple[list[AI4RAGChunk], str]
            The retrieved chunks and the rendered user message containing them.
        """
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
        """
        Generate an answer for a question using RAG pipeline.

        Parameters
        ----------
        question : str
            The user's question.

        **kwargs
            Additional parameters (e.g., number_of_chunks).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "answer": The generated answer
            - "reference_documents": The retrieved document chunks
            - "question": The original question
        """
        reference_documents, user_message = self._build_enriched_user_message(question, **kwargs)

        messages = [
            {"role": "system", "content": self.foundation_model.system_message_text},
            {"role": "user", "content": user_message},
        ]

        chat_response = self.foundation_model.chat(messages=messages)

        return {
            "answer": chat_response[0].message.content,
            "reference_documents": reference_documents,
            "question": question,
        }

    def generate_stream(self, question: str, **kwargs):
        """
        Generate a streaming answer for a question using RAG pipeline.

        Note: This is a placeholder implementation. Full streaming support
        would require streaming capabilities in the OpenAIFoundationModel.

        Parameters
        ----------
        question : str
            The user's question.

        **kwargs
            Additional parameters (e.g., number_of_chunks).

        Yields
        ------
        str
            Chunks of the generated answer.
        """
        result = self.generate(question, **kwargs)
        yield result["answer"]

    def chat(self, messages: list[MessageTyped], **kwargs) -> list[Any]:
        """
        Run a RAG-enriched chat completion over a conversation history.

        Mimics a chat-completions call: the message history is forwarded to the
        foundation model as-is, except the last message (the current user turn),
        whose content is used as the retrieval query and replaced with its
        RAG-enriched version before being sent to the model. The template's own
        system message is always prepended, so `messages` should not include one.

        Parameters
        ----------
        messages : list[MessageTyped]
            Conversation history, e.g. [{"role": "user", "content": "..."}].
            Must contain at least one message.

        **kwargs
            Additional parameters forwarded to the retriever (e.g. number_of_chunks).

        Returns
        -------
        list[Any]
            Chat response choices from the foundation model.
        """
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
