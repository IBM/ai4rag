# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, Sequence

import tiktoken
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument

from .base_chunker import BaseChunker
from .chunk import AI4RAGChunk

__all__ = ["DoclingChunker"]

_DEFAULT_TIKTOKEN_MODEL = "text-embedding-3-small"


class DoclingChunker(BaseChunker):
    """
    Structure-aware, token-aware chunker wrapping docling's ``HybridChunker``.

    Operates directly on ``DoclingDocument`` objects, preserving document
    hierarchy (headings, tables, figures) during chunking. Chunks are
    bounded by a token limit aligned to the embedding model.

    Parameters
    ----------
    max_tokens : int, default=8192
        Maximum number of tokens per chunk.

    contextualize : bool, default=True
        When ``True``, each chunk's text is enriched with its heading
        hierarchy via ``HybridChunker.contextualize``. This improves
        embedding quality at the cost of increased token usage.

    tokenizer : BaseTokenizer | None, default=None
        Tokenizer for token counting and split-point decisions.
        When ``None``, defaults to OpenAI tiktoken (``cl100k_base``,
        zero model downloads).

    merge_peers : bool, default=True
        Merge adjacent undersized chunks that share the same heading
        and caption context.
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        contextualize: bool = True,
        tokenizer: BaseTokenizer | None = None,
        merge_peers: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.contextualize = contextualize
        self.merge_peers = merge_peers

        if tokenizer is None:
            encoding = tiktoken.encoding_for_model(_DEFAULT_TIKTOKEN_MODEL)
            tokenizer = OpenAITokenizer(tokenizer=encoding, max_tokens=max_tokens)

        self._tokenizer = tokenizer
        self._chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=merge_peers,
        )

    def split_documents(self, documents: Sequence[DoclingDocument]) -> list[AI4RAGChunk]:
        """
        Split docling documents into token-bounded chunks.

        Parameters
        ----------
        documents : Sequence[DoclingDocument]
            Parsed documents to chunk.

        Returns
        -------
        list[AI4RAGChunk]
            Chunks with ``document_id``, ``sequence_number``, and
            optional heading/provenance metadata.
        """
        all_chunks: list[AI4RAGChunk] = []

        for doc in documents:
            doc_id = doc.name or str(hash(str(doc)))
            seq_num = 0

            for chunk in self._chunker.chunk(doc):
                seq_num += 1

                text = self._chunker.contextualize(chunk) if self.contextualize else chunk.text

                metadata: dict[str, Any] = {
                    "document_id": doc_id,
                    "sequence_number": seq_num,
                }

                if chunk.meta.headings:
                    metadata["headings"] = chunk.meta.headings

                all_chunks.append(AI4RAGChunk(text=text, metadata=metadata))

        return all_chunks

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary that can be used to recreate an instance."""
        return {
            "max_tokens": self.max_tokens,
            "contextualize": self.contextualize,
            "merge_peers": self.merge_peers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DoclingChunker":
        """Create an instance from the dictionary."""
        return cls(**d)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DoclingChunker):
            return self.to_dict() == other.to_dict()
        return NotImplemented
