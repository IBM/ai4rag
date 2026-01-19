# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document

from ai4rag.rag.embedding.base_model import EmbeddingModel
from ai4rag.rag.embedding.llama_stack import LLamaStackEmbeddingModel
from ai4rag.rag.foundation_models.base import FoundationModel
from ai4rag.rag.foundation_models.foundation_model import LlamaStackFoundationModel
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.llama_stack import LlamaStackVectorStore
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker


class BaseAutoRAGTemplate(ABC):
    """
    Base abstract class for Retrieval-Augmented Generation (RAG) templates.

    This class defines the interface for RAG templates that combine embedding models,
    vector stores, retrievers, and foundation models to enable question-answering
    over custom document collections.

    A RAG template orchestrates the following workflow:
    1. Index building: Process and store documents in a vector store
    2. Retrieval: Find relevant documents for a given query
    3. Generation: Use a foundation model to generate answers based on retrieved context
    4. Evaluation: Calculate quality scores for generated responses

    Parameters
    ----------
    embedding : EmbeddingModel
        The embedding model used to convert text into vector representations.

    foundation_model : FoundationModel
        The foundation model (LLM) used to generate answers based on retrieved context.

    retriever : Retriever
        The retriever component responsible for finding relevant documents from the vector store.

    vector_store : BaseVectorStore
        The vector store that maintains indexed documents and supports similarity search.

    Attributes
    ----------
    embedding : EmbeddingModel
        Reference to the initialized embedding model.

    foundation_model : FoundationModel
        Reference to the initialized foundation model.

    retriever : Retriever
        Reference to the initialized retriever.

    vector_store : BaseVectorStore
        Reference to the initialized vector store.

    Notes
    -----
    Subclasses must implement all abstract methods: build_index, generate,
    generate_stream, and calculate_score.
    """

    def __init__(
        self,
        embedding: EmbeddingModel,
        foundation_model: FoundationModel,
        retriever: Retriever,
        vector_store: BaseVectorStore,
    ):
        self.embedding: EmbeddingModel = embedding
        self.foundation_model: FoundationModel = foundation_model
        self.retriever: Retriever = retriever
        self.vector_store: BaseVectorStore = vector_store

    @abstractmethod
    def build_index(
        self,
        *args,
        **kwargs,
    ):
        """Index building method."""

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
    def calculate_score(
        self,
        *args,
        **kwargs,
    ):
        """Template score calculation method."""


class LlamaStackRAGTemplate(BaseAutoRAGTemplate):
    """
    RAG template using Llama Stack components for embedding, vector store,
    retrieval, and foundation model, with LangChain for document chunking.

    This template implements the BaseAutoRAGTemplate using Llama Stack services
    for all major RAG components while utilizing LangChain's chunking capabilities.

    Parameters
    ----------
    embedding_model : LLamaStackEmbeddingModel
        Initialized Llama Stack embedding model.

    vector_store : LlamaStackVectorStore
        Initialized Llama Stack vector store.

    retriever : Retriever
        Initialized retriever for document retrieval.

    foundation_model : LlamaStackFoundationModel
        Initialized Llama Stack foundation model for text generation.

    chunker : LangChainChunker
        Initialized LangChain chunker for document splitting.

    system_message_text : str | None, default=None
        System message template for chat completion. If None, uses default.

    user_message_text : str | None, default=None
        User message template for chat completion. If None, uses default.

    context_template_text : str | None, default=None
        Template for formatting retrieved context. If None, uses default.
    """

    def __init__(
        self,
        embedding_model: LLamaStackEmbeddingModel,
        vector_store: LlamaStackVectorStore,
        retriever: Retriever,
        foundation_model: LlamaStackFoundationModel,
        chunker: LangChainChunker,
        system_message_text: Optional[str] = None,
        user_message_text: Optional[str] = None,
        context_template_text: Optional[str] = None,
    ):
        self.chunker = chunker
        self.system_message_text = system_message_text or self._get_default_system_message()
        self.user_message_text = user_message_text or self._get_default_user_message()
        self.context_template_text = context_template_text or self._get_default_context_template()

        super().__init__(
            embedding=embedding_model,
            foundation_model=foundation_model,
            retriever=retriever,
            vector_store=vector_store,
        )

    @staticmethod
    def _get_default_system_message() -> str:
        """Default system message template."""
        return (
            "You are a helpful AI assistant. Use the provided context to answer "
            "the user's question accurately and concisely. If the context doesn't "
            "contain enough information to answer the question, say so."
        )

    @staticmethod
    def _get_default_user_message() -> str:
        """Default user message template with placeholders."""
        return "Context:\n{reference_documents}\n\n" "Question: {question}\n\n" "Answer:"

    @staticmethod
    def _get_default_context_template() -> str:
        """Default context template for formatting retrieved documents."""
        return "Document: {document}\n"

    def build_index(self, documents: list[Document]) -> None:
        """
        Index documents into the vector store.

        This method chunks the documents using the LangChain chunker and
        adds them to the vector store.

        Parameters
        ----------
        documents : list[Document]
            List of LangChain Document objects to index.
        """
        chunks = self.chunker.split_documents(documents)

        self.vector_store.add_documents(chunks)

    def generate(
        self,
        question: str,
        **retrieval_kwargs,
    ) -> dict[str, Any]:
        """
        Generate an answer for a question using RAG pipeline.

        Parameters
        ----------
        question : str
            The user's question.

        **retrieval_kwargs
            Additional parameters for retrieval (e.g., number_of_chunks).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "answer": The generated answer
            - "retrieved_docs": The retrieved document chunks
            - "question": The original question
        """
        retrieved_docs = self.retriever.retrieve(question, **retrieval_kwargs)

        context = "\n".join(
            [self.context_template_text.format(document=doc.get("content", "")) for doc in retrieved_docs]
        )

        user_message = self.user_message_text.format(
            reference_documents=context,
            question=question,
        )

        answer = self.foundation_model.chat(
            system_message=self.system_message_text,
            user_message=user_message,
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "question": question,
        }

    def generate_stream(self, question: str, **retrieval_kwargs):
        """
        Generate a streaming answer for a question using RAG pipeline.

        Note: This is a placeholder implementation. Full streaming support
        would require streaming capabilities in the LlamaStackFoundationModel.

        Parameters
        ----------
        question : str
            The user's question.

        **retrieval_kwargs
            Additional parameters for retrieval (e.g., number_of_chunks).

        Yields
        ------
        str
            Chunks of the generated answer.
        """
        result = self.generate(question, **retrieval_kwargs)
        yield result["answer"]

    def calculate_score(self, question: str, answer: str, retrieved_docs: list[dict]) -> dict[str, Any]:
        """
        Calculate quality scores for the RAG response.

        This is a placeholder implementation. Real scoring would involve
        metrics like relevance, faithfulness, and answer quality.

        Parameters
        ----------
        question : str
            The original question.

        answer : str
            The generated answer.

        retrieved_docs : list[dict]
            The retrieved document chunks.

        Returns
        -------
        dict[str, Any]
            Dictionary containing various quality scores.
        """
