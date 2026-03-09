# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

__all__ = [
    "BaseChunker",
]

ChunkT = TypeVar("ChunkT")


class BaseChunker(ABC, Generic[ChunkT]):
    """
    Responsible for handling splitting document operations
    in the RAG application.
    """

    @abstractmethod
    def split_documents(self, documents: Sequence[ChunkT]) -> list[ChunkT]:
        """
        Split series of documents into smaller parts based on
        the provided chunker settings.

        Parameters
        ----------
        documents : Sequence[ChunkType]
            Sequence of elements that contain context in a text format.

        Returns
        -------
        list[ChunkType]
            List of documents split into smaller ones, having less content.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return dictionary that can be used to recreate an instance of the BaseChunker."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseChunker":
        """Create an instance from the dictionary."""
