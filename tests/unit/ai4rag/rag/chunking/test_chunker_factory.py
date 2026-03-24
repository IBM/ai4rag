# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.rag.chunking.chunker_factory import get_chunker
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker


class TestGetChunker:
    """Test suite for the get_chunker factory function."""

    def test_get_chunker_recursive(self):
        chunker = get_chunker(chunking_method="recursive", chunk_size=1024, chunk_overlap=128)
        assert isinstance(chunker, LangChainChunker)
        assert chunker.method == "recursive"

    def test_get_chunker_markdown(self):
        chunker = get_chunker(chunking_method="markdown", chunk_size=1024, chunk_overlap=128)
        assert isinstance(chunker, LangChainChunker)
        assert chunker.method == "markdown"

    def test_get_chunker_markdown_header(self):
        chunker = get_chunker(chunking_method="markdown_header", chunk_size=512, chunk_overlap=64)
        assert isinstance(chunker, LangChainChunker)
        assert chunker.method == "markdown_header"

    def test_get_chunker_unsupported(self):
        with pytest.raises(ValueError, match="not supported"):
            get_chunker(chunking_method="unsupported_method", chunk_size=512, chunk_overlap=64)
