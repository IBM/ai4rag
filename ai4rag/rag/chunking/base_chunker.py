# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Sequence, TypeVar

from langchain_core.documents import Document

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

    @staticmethod
    def _set_document_id_in_metadata_if_missing(documents: Iterable[Document]) -> None:
        """
        Sets "document_id" in the metadata if it is missing.
        The document_id is the hash of the document's content.

        Parameters
        ----------
        documents : Iterable[Document]
            Sequence of documents for which document ids will be provided.
        """
        for doc in documents:
            if "document_id" not in doc.metadata:
                doc.metadata["document_id"] = str(hash(doc.page_content))

    @staticmethod
    def _set_sequence_number_in_metadata(chunks: list[Document]) -> list[Document]:
        """
        Sets "sequence_number" in the metadata, sorted by chunks' "start_index".

        Parameters
        ----------
        chunks : list[Document]
            Sequence of chunks of documents that contain context in a text format.

        Returns
        -------
        list[Document]
            List of updated chunks, sorted by document_id and sequence_number.
        """
        sorted_chunks = sorted(chunks, key=lambda x: (x.metadata["document_id"], x.metadata["start_index"]))

        document_sequence: dict[str, int] = {}
        for chunk in sorted_chunks:
            doc_id = chunk.metadata["document_id"]
            prev_seq_num = document_sequence.get(doc_id, 0)
            seq_num = prev_seq_num + 1
            document_sequence[doc_id] = seq_num
            chunk.metadata["sequence_number"] = seq_num

        return sorted_chunks
