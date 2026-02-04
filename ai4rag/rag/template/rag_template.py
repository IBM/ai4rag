# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

from langchain_core.documents import Document


from .base_template import BaseRAGTemplate, RAGTemplateError
from ..embedding.llama_stack import LSEmbeddingModel
from ..foundation_models.base_model import FoundationModel
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.vector_store.llama_stack import LSVectorStore
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker


class LlamaStackRAG(BaseRAGTemplate):
    """
    RAG template using Llama Stack components for embedding, vector store,
    retrieval, and foundation model, with LangChain for document chunking.

    This template implements the BaseRAGTemplate using Llama Stack services
    for all major RAG components while utilizing LangChain's chunking capabilities.

    Parameters
    ----------
    foundation_model : FoundationModel
        Initialized Llama Stack foundation model for text generation.

    retriever : Retriever
        Initialized retriever for document retrieval.

    chunker : LangChainChunker | None, default=None
        Initialized LangChain chunker for document splitting.

    embedding_model : LSEmbeddingModel | None, default=None
        Initialized Llama Stack embedding model.

    vector_store : LSVectorStore | None, default=None
        Initialized Llama Stack vector store.
    """

    def __init__(
        self,
        foundation_model: FoundationModel,
        retriever: Retriever,
        chunker: LangChainChunker | None = None,
        embedding_model: LSEmbeddingModel | None = None,
        vector_store: LSVectorStore | None = None,
    ):
        super().__init__(
            foundation_model=foundation_model,
            retriever=retriever,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        self.chunker = chunker

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
        if self.chunker is None and self.embedding_model is None and self.vector_store is None:
            raise RAGTemplateError()
        chunks = self.chunker.split_documents(documents)

        self.vector_store.add_documents(chunks)

    def generate(self, question: str, **retrieval_kwargs) -> dict[str, Any]:
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
            - "reference_documents": The retrieved document chunks
            - "question": The original question
        """
        reference_documents = self.retriever.retrieve(question, **retrieval_kwargs)

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

        answer = self.foundation_model.chat(
            system_message=self.foundation_model.system_message_text,
            user_message=user_message,
        )

        return {
            "answer": answer,
            "reference_documents": reference_documents,
            "question": question,
        }

    def generate_stream(self, question: str, **retrieval_kwargs):
        """
        Generate a streaming answer for a question using RAG pipeline.

        Note: This is a placeholder implementation. Full streaming support
        would require streaming capabilities in the LSFoundationModel.

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
