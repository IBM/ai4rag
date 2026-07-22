# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AI4RAGChunk"]


@dataclass
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

    Attributes
    ----------
    chunk_id : str
        Deterministic SHA-256 hex digest derived from ``document_id``,
        ``sequence_number``, and ``text``. Computed automatically on
        construction; not passed as an ``__init__`` argument.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        hasher = hashlib.sha256()
        hasher.update(self.metadata.get("document_id", "").encode())
        seq = self.metadata.get("sequence_number", 0)
        if isinstance(seq, list):
            for s in sorted(seq):
                hasher.update(s.to_bytes(4, "big"))
        else:
            hasher.update(seq.to_bytes(4, "big"))
        hasher.update(self.text.encode())
        self.chunk_id = hasher.hexdigest()
