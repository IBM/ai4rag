# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from .base_chunker import BaseChunker
from .langchain_chunker import LangChainChunker

__all__ = [
    "get_chunker",
]

_SUPPORTED_METHODS = ("recursive", "markdown", "markdown_header")


def get_chunker(chunking_method: str, chunk_size: int, chunk_overlap: int, **kwargs: Any) -> BaseChunker:
    """Create a chunker instance based on the chunking method.

    Parameters
    ----------
    chunking_method : str
        The chunking method to use.
    chunk_size : int
        Maximum chunk size.
    chunk_overlap : int
        Overlap between chunks.
    **kwargs
        Additional keyword arguments passed to the chunker constructor.

    Returns
    -------
    BaseChunker
        An initialized chunker instance.

    Raises
    ------
    ValueError
        When the chunking method is not supported.
    """
    if chunking_method in _SUPPORTED_METHODS:
        return LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    raise ValueError(
        f"Chunking method '{chunking_method}' is not supported. "
        f"Use one of {_SUPPORTED_METHODS}."
    )
