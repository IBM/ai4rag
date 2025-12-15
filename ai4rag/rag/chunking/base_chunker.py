from typing import Sequence, Any, Generic, TypeVar
from abc import ABC, abstractmethod


__all__ = [
    "BaseChunker",
]

ChunkType = TypeVar("ChunkType")


class BaseChunker(ABC, Generic[ChunkType]):
    """
    Responsible for handling splitting document operations
    in the RAG application.
    """

    @abstractmethod
    def split_documents(self, documents: Sequence[ChunkType]) -> list[ChunkType]:
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
