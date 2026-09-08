# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod

from ..foundation_models.base_model import BaseFoundationModel
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
