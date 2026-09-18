# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod

from ai4rag.rag.chunking.chunk import AI4RAGChunk

from ..foundation_models.base_model import BaseFoundationModel, MessageTyped
from ..retrieval.retriever import Retriever


class BaseRAGTemplate(ABC):
    """
    Base abstract class for Retrieval-Augmented Generation (RAG) templates.

    This class defines the interface for RAG templates that combine a retriever
    and a foundation model to enable question-answering over previously indexed
    document collections.

    A RAG template orchestrates the following workflow:
    1. Retrieval: Find relevant documents for a given query
    2. Generation: Use a foundation model to generate answers based on retrieved context

    Index building is a separate, upstream concern (see `ai4rag.rag.vector_store`)
    and is intentionally out of scope for RAG templates.

    Parameters
    ----------
    foundation_model : BaseFoundationModel
        The foundation model (LLM) used to generate answers based on retrieved context.

    retriever : Retriever
        The retriever component responsible for finding relevant documents from the vector store.

    Notes
    -----
    Subclasses must implement all abstract methods: generate, generate_stream, chat.
    """

    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        retriever: Retriever,
    ):
        self.foundation_model: BaseFoundationModel = foundation_model
        self.retriever: Retriever = retriever

    def _build_enriched_user_message(self, question: str, **kwargs) -> tuple[list[AI4RAGChunk], str]:
        """Retrieve context for a question and render the user message."""
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

    def _build_rag_messages(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
        """Add retrieved context to the last user message in a conversation."""
        if not messages:
            raise ValueError("`messages` must contain at least one message.")

        *history, last_message = messages
        _, enriched_content = self._build_enriched_user_message(last_message["content"], **kwargs)
        return [
            {"role": "system", "content": self.foundation_model.system_message_text},
            *history,
            {**last_message, "content": enriched_content},
        ]

    @abstractmethod
    def generate(
        self,
        *args,
        **kwargs,
    ):
        """Template generation method."""

    @abstractmethod
    def generate_stream(
        self,
        *args,
        **kwargs,
    ):
        """Template generation stream method."""

    @abstractmethod
    def chat(
        self,
        *args,
        **kwargs,
    ):
        """Chat-completion style generation method."""
