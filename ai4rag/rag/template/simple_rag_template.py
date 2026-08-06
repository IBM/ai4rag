# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

from docling_core.types.doc import DoclingDocument

from ai4rag.rag.chunking.base_chunker import BaseChunker
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore

from ..embedding.base_model import BaseEmbeddingModel
from ..foundation_models.base_model import BaseFoundationModel
from .base_template import BaseRAGTemplate, RAGTemplateError


class SimpleRAG(BaseRAGTemplate):
    """
    RAG template composing embedding, vector store, retrieval, and
    foundation model components.

    Parameters
    ----------
    foundation_model : BaseFoundationModel
        Initialized foundation model for text generation.

    retriever : Retriever
        Initialized retriever for document retrieval.

    chunker : BaseChunker | None, default=None
        Initialized chunker for document splitting.

    embedding_model : BaseEmbeddingModel | None, default=None
        Initialized embedding model.

    vector_store : BaseVectorStore | None, default=None
        Initialized vector store.
    """

    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        retriever: Retriever,
        chunker: BaseChunker | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: BaseVectorStore | None = None,
    ):
        super().__init__(
            foundation_model=foundation_model,
            retriever=retriever,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        self.chunker = chunker

    def build_index(self, documents: list[DoclingDocument], **kwargs) -> None:
        """
        Index documents into the vector store.

        This method chunks the documents and adds them to the vector store.

        Parameters
        ----------
        documents : list[DoclingDocument]
            Parsed docling documents to index.
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

        context = "\n\n".join(
            self.foundation_model.context_template_text.format(document=chunk.text, doc_number=doc_number)
            for doc_number, chunk in enumerate(reference_documents, start=1)
        )

        user_message = self.foundation_model.user_message_text.format(
            reference_documents=context,
            question=question,
        )

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
