# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, Literal, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker

__all__ = [
    "LangChainChunker",
]

_DEFAULT_HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


class LangChainChunker(BaseChunker[Document]):
    """
    Wrapper for LangChain TextSplitter.

    Parameters
    ----------
    method : Literal["recursive", "markdown", "markdown_header"], default="recursive"
        Describes the type of TextSplitter as the main instance performing the chunking.

    chunk_size : int, default=2048
        Maximum size of a single chunk that is returned.

    chunk_overlap : int, default=256
        Overlap in characters between chunks.

    headers_to_split_on : list[tuple[str, str]] | None, default=None
        Headers to split on when using the "markdown_header" method.
        Defaults to [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")].

    Other Parameters
    ----------------
    separators : list[str]
        Separators between chunks.
    """

    supported_methods = ("recursive", "markdown", "markdown_header")

    def __init__(
        self,
        method: Literal["recursive", "character", "token", "markdown", "markdown_header"] = "recursive",
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on or _DEFAULT_HEADERS_TO_SPLIT_ON
        self.separators = kwargs.pop("separators", ["\n\n", r"(?<=\. )", "\n", " ", ""])
        self._text_splitter = self._get_text_splitter()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LangChainChunker):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def _get_text_splitter(self):
        """Create an instance of TextSplitter based on the settings."""

        match self.method:
            case "recursive":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators,
                    length_function=len,
                    add_start_index=True,
                )

            case "markdown":
                text_splitter = MarkdownTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    add_start_index=True,
                )

            case "markdown_header":
                text_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=self.headers_to_split_on,
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
        ret = {
            "method": self.method,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        if self.method == "markdown_header" and self.headers_to_split_on:
            ret["headers_to_split_on"] = self.headers_to_split_on

        return ret

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LangChainChunker":
        """Create an instance from the dictionary."""

        return cls(**d)

    def _split_markdown_header(self, documents: Sequence[Document]) -> list[Document]:
        """Split documents using MarkdownHeaderTextSplitter with optional refinement.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents to split by markdown headers.

        Returns
        -------
        list[Document]
            Chunks split at header boundaries with parent metadata preserved.
        """
        all_chunks = []

        for doc in documents:
            parent_metadata = dict(doc.metadata)
            header_chunks = self._text_splitter.split_text(doc.page_content)

            search_start = 0
            for header_chunk in header_chunks:
                chunk_content = header_chunk.page_content
                start_index = doc.page_content.find(chunk_content, search_start)
                if start_index == -1:
                    start_index = search_start
                search_start = start_index + len(chunk_content)

                merged_metadata = {**parent_metadata, **header_chunk.metadata, "start_index": start_index}

                if self.chunk_size > 0 and len(chunk_content) > self.chunk_size:
                    refine_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=len,
                        add_start_index=True,
                    )
                    sub_chunks = refine_splitter.split_text(chunk_content)
                    for sub_text in sub_chunks:
                        sub_start = chunk_content.find(sub_text)
                        sub_metadata = {**merged_metadata, "start_index": start_index + max(sub_start, 0)}
                        all_chunks.append(Document(page_content=sub_text, metadata=sub_metadata))
                else:
                    all_chunks.append(Document(page_content=chunk_content, metadata=merged_metadata))

        return all_chunks

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """
        Split series of documents into smaller chunks based on the provided
        chunker settings. Each chunk has metadata that includes the document_id,
        sequence_number, and start_index.

        Parameters
        ----------
        documents : Sequence[Document]
            Sequence of elements that contain context in a text format.

        Returns
        -------
        list[Document]
            List of documents split into smaller chunks.
        """
        self._set_document_id_in_metadata_if_missing(documents)
        if self.method == "markdown_header":
            chunks = self._split_markdown_header(documents)
        else:
            chunks = self._text_splitter.split_documents(documents)
        sorted_chunks = self._set_sequence_number_in_metadata(chunks)
        return sorted_chunks
