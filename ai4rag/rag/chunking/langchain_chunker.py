# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, Iterable, Literal, Sequence

import tiktoken
from docling_core.types.doc import DoclingDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from .base_chunker import BaseChunker
from .chunk import AI4RAGChunk

__all__ = [
    "LangChainChunker",
]

_DEFAULT_TIKTOKEN_MODEL = "text-embedding-3-small"


class LangChainChunker(BaseChunker):
    """
    Wrapper for LangChain TextSplitter operating on ``DoclingDocument`` input.

    Converts each ``DoclingDocument`` to markdown internally, applies
    token-based splitting via tiktoken, and returns ``AI4RAGChunk`` objects.

    Parameters
    ----------
    method : Literal["recursive", "character", "token"], default="recursive"
        Describes the type of TextSplitter as the main instance performing the chunking.

    chunk_size : int, default=2048
        Maximum number of tokens per chunk.

    chunk_overlap : int, default=256
        Overlap in tokens between chunks.

    Other Parameters
    ----------------
    separators : list[str]
        Separators between chunks.
    """

    supported_methods = ("recursive",)

    def __init__(
        self,
        method: Literal["recursive", "character", "token"] = "recursive",
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = kwargs.pop("separators", ["\n\n", r"(?<=\. )", "\n", " ", ""])
        self._encoding = tiktoken.encoding_for_model(_DEFAULT_TIKTOKEN_MODEL)
        self._text_splitter = self._get_text_splitter()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LangChainChunker):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def _get_text_splitter(self) -> TextSplitter:
        """Create an instance of TextSplitter based on the settings."""

        match self.method:
            case "recursive":

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators,
                    length_function=lambda text: len(self._encoding.encode(text)),
                    add_start_index=True,
                )

            case _:
                raise ValueError(
                    f"Chunker method '{self.method}' is not supported. Use one of {self.supported_methods}."
                )

        return text_splitter

    def to_dict(self) -> dict[str, Any]:
        """
        Return dictionary that can be used to recreate an instance of the LangChainChunker.
        """
        params = (
            "method",
            "chunk_size",
            "chunk_overlap",
        )

        ret = {k: v for k, v in self.__dict__.items() if k in params}

        return ret

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LangChainChunker":
        """Create an instance from the dictionary."""

        return cls(**d)

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
        # sort chunks by start_index for each document_id
        sorted_chunks = sorted(chunks, key=lambda x: (x.metadata["document_id"], x.metadata["start_index"]))

        document_sequence: dict[str, int] = {}
        for chunk in sorted_chunks:
            doc_id = chunk.metadata["document_id"]
            prev_seq_num = document_sequence.get(doc_id, 0)
            seq_num = prev_seq_num + 1
            document_sequence[doc_id] = seq_num
            chunk.metadata["sequence_number"] = seq_num

        return sorted_chunks

    @staticmethod
    def _docling_to_langchain(documents: Sequence[DoclingDocument]) -> list[Document]:
        """
        Convert ``DoclingDocument`` objects to langchain ``Document`` objects
        by exporting each to markdown.

        Parameters
        ----------
        documents : Sequence[DoclingDocument]
            Parsed docling documents.

        Returns
        -------
        list[Document]
            Langchain documents with markdown content and ``document_id`` metadata.
        """
        return [
            Document(
                page_content=doc.export_to_markdown(),
                metadata={"document_id": doc.name or str(hash(str(doc)))},
            )
            for doc in documents
        ]

    def split_documents(self, documents: Sequence[DoclingDocument]) -> list[AI4RAGChunk]:
        """
        Split docling documents into smaller chunks using token-based splitting.

        Each ``DoclingDocument`` is first exported to markdown, then split using
        the configured ``TextSplitter``. Results are returned as ``AI4RAGChunk``.

        Parameters
        ----------
        documents : Sequence[DoclingDocument]
            Parsed docling documents to chunk.

        Returns
        -------
        list[AI4RAGChunk]
            Chunks with ``document_id``, ``sequence_number``, and ``start_index`` metadata.
        """
        lc_docs = self._docling_to_langchain(documents)
        self._set_document_id_in_metadata_if_missing(lc_docs)
        chunks = self._text_splitter.split_documents(lc_docs)
        sorted_chunks = self._set_sequence_number_in_metadata(chunks)
        return [AI4RAGChunk(text=chunk.page_content, metadata=chunk.metadata) for chunk in sorted_chunks]
