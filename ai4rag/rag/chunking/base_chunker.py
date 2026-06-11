# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Sequence

from docling_core.types.doc import DoclingDocument

from .chunk import AI4RAGChunk

__all__ = [
    "BaseChunker",
]


class BaseChunker(ABC):
    """
    Responsible for handling splitting document operations
    in the RAG application.

    All chunkers accept ``DoclingDocument`` as input and
    produce ``AI4RAGChunk`` as output.
    """

    @abstractmethod
    def split_documents(self, documents: Sequence[DoclingDocument]) -> list[AI4RAGChunk]:
        """
        Split series of documents into smaller parts based on
        the provided chunker settings.

        Parameters
        ----------
        documents : Sequence[DoclingDocument]
            Parsed docling documents to chunk.

        Returns
        -------
        list[AI4RAGChunk]
            List of chunks produced from the input documents.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return dictionary that can be used to recreate an instance of the BaseChunker."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseChunker":
        """Create an instance from the dictionary."""
