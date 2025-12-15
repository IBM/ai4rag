from typing import Literal, Sequence, Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document

from .base_chunker import BaseChunker


__all__ = [
    "LangChainChunker",
]


class LangChainChunker(BaseChunker[Document]):
    """
    Wrapper for LangChain TextSplitter.

    Parameters
    ----------
    method : Literal["recursive", "character", "token"], default="recursive"
        Describes the type of TextSplitter as the main instance performing the chunking.

    chunk_suze : int, default=2048
        Maximum size of a single chunk that is returned.

    chunk_overlap : int, default=256
        Overlap in characters between chunks.

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
        self.separators = kwargs.pop("separators", ["\n\n", "(?<=\. )", "\n", " ", ""])
        self._text_splitter = self._get_text_splitter()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LangChainChunker):
            return self.to_dict() == other.to_dict()
        else:
            return NotImplemented

    def _get_text_splitter(self) -> TextSplitter:
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

            case _:
                raise ValueError(
                    "Chunker method '{}' is not supported. Use one of {}".format(
                        self.method, self.supported_methods
                    )
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
        sorted_chunks = sorted(
            chunks, key=lambda x: (x.metadata["document_id"], x.metadata["start_index"])
        )

        document_sequence: dict[str, int] = {}
        for chunk in sorted_chunks:
            doc_id = chunk.metadata["document_id"]
            prev_seq_num = document_sequence.get(doc_id, 0)
            seq_num = prev_seq_num + 1
            document_sequence[doc_id] = seq_num
            chunk.metadata["sequence_number"] = seq_num

        return sorted_chunks

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
        chunks = self._text_splitter.split_documents(documents)
        sorted_chunks = self._set_sequence_number_in_metadata(chunks)
        return sorted_chunks
