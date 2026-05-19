# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

from langchain_core.documents import Document

from ai4rag.rag.chunking.langchain_chunker import LangChainChunker
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.vector_store.ogx import OGXVectorStore

from ..embedding.base_model import BaseEmbeddingModel
from ..foundation_models.base_model import BaseFoundationModel
from .base_template import BaseRAGTemplate, RAGTemplateError


class SimpleRAG(BaseRAGTemplate):
    """
    RAG template using OGX components for embedding, vector store,
    retrieval, and foundation model, with LangChain for document chunking.

    This template implements the BaseRAGTemplate using OGX services
    for all major RAG components while utilizing LangChain's chunking capabilities.

    Parameters
    ----------
    foundation_model : BaseFoundationModel
        Initialized OGX foundation model for text generation.

    retriever : Retriever
        Initialized retriever for document retrieval.

    chunker : LangChainChunker | None, default=None
        Initialized LangChain chunker for document splitting.

    embedding_model : OGXEmbeddingModel | None, default=None
        Initialized OGX embedding model.

    vector_store : OGXVectorStore | None, default=None
        Initialized OGX vector store.
    """

    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        retriever: Retriever,
        chunker: LangChainChunker | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: OGXVectorStore | None = None,
    ):
        super().__init__(
            foundation_model=foundation_model,
            retriever=retriever,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        self.chunker = chunker

    def build_index(self, documents: list[Document], **kwargs) -> None:
        """
        Index documents into the vector store.

        This method chunks the documents using the LangChain chunker and
        adds them to the vector store.

        Parameters
        ----------
        documents : list[Document]
            List of LangChain Document objects to index.
        """
        if self.chunker is None and self.embedding_model is None and self.vector_store is None:
            raise RAGTemplateError()
        chunks = self.chunker.split_documents(documents)

        self.vector_store.add_documents(chunks)

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
        reference_documents = self.retriever.retrieve(question, **kwargs)

        context = "\n".join(
            [
                self.foundation_model.context_template_text.format(document=getattr(doc, "page_content", ""))
                for doc in reference_documents
            ]
        )

        user_message = self.foundation_model.user_message_text.format(
            reference_documents=context,
            question=question,
        )

        answer = self.foundation_model.create_response(
            user_message=user_message,
            vector_store_id=self.retriever.collection_name,
        )

        return {
            "answer": answer,
            "reference_documents": reference_documents,
            "question": question,
        }

    def generate_stream(self, question: str, **kwargs):
        """
        Generate a streaming answer for a question using RAG pipeline.

        Note: This is a placeholder implementation. Full streaming support
        would require streaming capabilities in the OGXFoundationModel.

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
