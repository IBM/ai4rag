# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AI4RAGChunk"]


@dataclass(frozen=True)
class AI4RAGChunk:
    """
    Framework-agnostic chunk representation used across the ai4rag pipeline.

    Parameters
    ----------
    text : str
        The textual content of the chunk.

    metadata : dict[str, Any]
        Chunk metadata. Expected keys include ``document_id`` and
        ``sequence_number``; additional keys (headings, provenance)
        are chunker-dependent.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Deterministic SHA-256 identifier derived from document_id, sequence_number, and text."""
        hasher = hashlib.sha256()
        hasher.update(self.metadata.get("document_id", "").encode())
        hasher.update(self.metadata.get("sequence_number", 0).to_bytes(4, "big"))
        hasher.update(self.text.encode())
        return hasher.hexdigest()
