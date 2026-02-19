# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod

from ..embedding.base_model import BaseEmbeddingModel
from ..foundation_models.base_model import BaseFoundationModel
from ..retrieval.retriever import Retriever
from ..vector_store.base_vector_store import BaseVectorStore


class RAGTemplateError(Exception):
    """Instance raised when error occurs in the RAG template."""


class BaseRAGTemplate(ABC):
    """
    Base abstract class for Retrieval-Augmented Generation (RAG) templates.

    This class defines the interface for RAG templates that combine embedding models,
    vector stores, retrievers, and foundation models to enable question-answering
    over custom document collections.

    A RAG template orchestrates the following workflow:
    1. Index building: Process and store documents in a vector store
    2. Retrieval: Find relevant documents for a given query
    3. Generation: Use a foundation model to generate answers based on retrieved context

    Parameters
    ----------
    foundation_model : BaseFoundationModel
        The foundation model (LLM) used to generate answers based on retrieved context.

    retriever : Retriever
        The retriever component responsible for finding relevant documents from the vector store.

    embedding_model : BaseEmbeddingModel | None, default=None
        The embedding model used to convert text into vector representations.

    vector_store : BaseVectorStore | None, default=None
        The vector store that maintains indexed documents and supports similarity search.

    Notes
    -----
    Subclasses must implement all abstract methods: build_index, generate, generate_stream.
    """

    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        retriever: Retriever,
        vector_store: BaseVectorStore | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
    ):
        self.embedding_model: BaseEmbeddingModel = embedding_model
        self.foundation_model: BaseFoundationModel = foundation_model
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
