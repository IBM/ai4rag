# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import math

import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document

from ai4rag.rag.chunking.char_approx_tokenizer import _CHARS_PER_TOKEN
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker


def _make_docling_doc(name: str, text: str) -> DoclingDocument:
    """Helper to create a DoclingDocument with a single paragraph."""
    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
    return doc


class TestLangChainChunkerInitialization:
    """Test suite for LangChainChunker initialization."""

    def test_init_with_defaults(self):
        chunker = LangChainChunker()
        assert chunker.method == "recursive"
        assert chunker.chunk_size == 2048
        assert chunker.chunk_overlap == 256
        assert chunker.separators == ["\n\n", "(?<=\. )", "\n", " ", ""]

    def test_init_with_custom_parameters(self):
        chunker = LangChainChunker(method="recursive", chunk_size=1024, chunk_overlap=128, separators=["\n", " "])
        assert chunker.method == "recursive"
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 128
        assert chunker.separators == ["\n", " "]

    def test_init_with_unsupported_method(self):
        with pytest.raises(ValueError, match="not supported"):
            LangChainChunker(method="character")

    def test_init_with_token_method(self):
        with pytest.raises(ValueError, match="not supported"):
            LangChainChunker(method="token")

    def test_text_splitter_is_created(self):
        chunker = LangChainChunker()
        assert chunker._text_splitter is not None
        assert hasattr(chunker._text_splitter, "split_documents")


class TestLangChainChunkerSplitDocuments:
    """Test suite for LangChainChunker.split_documents with DoclingDocument input."""

    @pytest.fixture
    def sample_documents(self):
        return [
            _make_docling_doc("doc1.pdf", "This is a test document. It has multiple sentences."),
            _make_docling_doc("doc2.pdf", "Another document with some content."),
        ]

    @pytest.fixture
    def chunker_small(self):
        return LangChainChunker(chunk_size=20, chunk_overlap=5)

    def test_split_returns_ai4rag_chunks(self, chunker_small, sample_documents):
        chunks = chunker_small.split_documents(sample_documents)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, AI4RAGChunk) for chunk in chunks)

    def test_chunks_have_text_attribute(self, chunker_small, sample_documents):
        chunks = chunker_small.split_documents(sample_documents)
        for chunk in chunks:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0

    def test_split_adds_document_id_from_doc_name(self, chunker_small, sample_documents):
        chunks = chunker_small.split_documents(sample_documents)
        doc_ids = {chunk.metadata["document_id"] for chunk in chunks}
        assert "doc1.pdf" in doc_ids
        assert "doc2.pdf" in doc_ids

    def test_split_adds_sequence_number(self, chunker_small, sample_documents):
        chunks = chunker_small.split_documents(sample_documents)
        for chunk in chunks:
            assert "sequence_number" in chunk.metadata
            assert isinstance(chunk.metadata["sequence_number"], int)
            assert chunk.metadata["sequence_number"] > 0

    def test_sequence_numbers_are_sequential(self, chunker_small):
        doc = _make_docling_doc("doc1.pdf", "A" * 100)
        chunks = chunker_small.split_documents([doc])
        sequence_numbers = [chunk.metadata["sequence_number"] for chunk in chunks]
        assert sequence_numbers == list(range(1, len(chunks) + 1))

    def test_split_with_empty_list(self, chunker_small):
        chunks = chunker_small.split_documents([])
        assert chunks == []

    def test_split_respects_chunk_size(self, sample_documents):
        chunker = LangChainChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.split_documents(sample_documents)
        for chunk in chunks:
            assert math.ceil(len(chunk.text) / _CHARS_PER_TOKEN) <= 10

    def test_split_multiple_documents(self, chunker_small):
        docs = [
            _make_docling_doc("doc1.pdf", "First document with some content."),
            _make_docling_doc("doc2.pdf", "Second document with different content."),
            _make_docling_doc("doc3.pdf", "Third document."),
        ]
        chunks = chunker_small.split_documents(docs)
        assert len(chunks) > 0
        doc_ids = {chunk.metadata["document_id"] for chunk in chunks}
        assert "doc1.pdf" in doc_ids
        assert "doc2.pdf" in doc_ids
        assert "doc3.pdf" in doc_ids


class TestLangChainChunkerToDict:
    """Test suite for LangChainChunker.to_dict method."""

    def test_to_dict_returns_dict(self):
        result = LangChainChunker().to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_all_params(self):
        chunker = LangChainChunker(method="recursive", chunk_size=512, chunk_overlap=64)
        result = chunker.to_dict()
        assert set(result.keys()) == {"method", "chunk_size", "chunk_overlap"}
        assert result["method"] == "recursive"
        assert result["chunk_size"] == 512
        assert result["chunk_overlap"] == 64

    def test_to_dict_excludes_internal_attributes(self):
        result = LangChainChunker().to_dict()
        assert "_text_splitter" not in result
        assert "separators" not in result


class TestLangChainChunkerFromDict:
    """Test suite for LangChainChunker.from_dict method."""

    def test_from_dict_creates_instance(self):
        d = {"method": "recursive", "chunk_size": 1024, "chunk_overlap": 128}
        chunker = LangChainChunker.from_dict(d)
        assert isinstance(chunker, LangChainChunker)

    def test_from_dict_round_trip(self):
        original = LangChainChunker(method="recursive", chunk_size=1024, chunk_overlap=128)
        recreated = LangChainChunker.from_dict(original.to_dict())
        assert recreated.method == original.method
        assert recreated.chunk_size == original.chunk_size
        assert recreated.chunk_overlap == original.chunk_overlap


class TestLangChainChunkerEquality:
    """Test suite for LangChainChunker.__eq__ method."""

    def test_eq_same_parameters(self):
        c1 = LangChainChunker(method="recursive", chunk_size=1024, chunk_overlap=128)
        c2 = LangChainChunker(method="recursive", chunk_size=1024, chunk_overlap=128)
        assert c1 == c2

    def test_neq_different_chunk_size(self):
        c1 = LangChainChunker(chunk_size=1024)
        c2 = LangChainChunker(chunk_size=2048)
        assert c1 != c2

    def test_eq_with_non_chunker(self):
        assert LangChainChunker().__eq__("not a chunker") is NotImplemented

    def test_eq_with_none(self):
        assert LangChainChunker().__eq__(None) is NotImplemented


class TestLangChainChunkerStaticMethods:
    """Test suite for LangChainChunker static methods (internal langchain operations)."""

    def test_set_document_id_in_metadata_if_missing(self):
        documents = [
            Document(page_content="Test content 1", metadata={}),
            Document(page_content="Test content 2", metadata={}),
        ]
        LangChainChunker._set_document_id_in_metadata_if_missing(documents)
        for doc in documents:
            assert "document_id" in doc.metadata
            assert isinstance(doc.metadata["document_id"], str)

    def test_set_document_id_preserves_existing(self):
        documents = [Document(page_content="Test content", metadata={"document_id": "existing-id"})]
        LangChainChunker._set_document_id_in_metadata_if_missing(documents)
        assert documents[0].metadata["document_id"] == "existing-id"

    def test_set_sequence_number_in_metadata(self):
        chunks = [
            Document(page_content="Chunk 1", metadata={"document_id": "doc1", "start_index": 0}),
            Document(page_content="Chunk 2", metadata={"document_id": "doc1", "start_index": 10}),
            Document(page_content="Chunk 3", metadata={"document_id": "doc1", "start_index": 20}),
        ]
        result = LangChainChunker._set_sequence_number_in_metadata(chunks)
        assert result[0].metadata["sequence_number"] == 1
        assert result[1].metadata["sequence_number"] == 2
        assert result[2].metadata["sequence_number"] == 3

    def test_set_sequence_number_multiple_documents(self):
        chunks = [
            Document(page_content="C1", metadata={"document_id": "doc1", "start_index": 0}),
            Document(page_content="C2", metadata={"document_id": "doc1", "start_index": 10}),
            Document(page_content="CA", metadata={"document_id": "doc2", "start_index": 0}),
            Document(page_content="CB", metadata={"document_id": "doc2", "start_index": 10}),
        ]
        result = LangChainChunker._set_sequence_number_in_metadata(chunks)
        doc1_chunks = [c for c in result if c.metadata["document_id"] == "doc1"]
        doc2_chunks = [c for c in result if c.metadata["document_id"] == "doc2"]
        assert doc1_chunks[0].metadata["sequence_number"] == 1
        assert doc1_chunks[1].metadata["sequence_number"] == 2
        assert doc2_chunks[0].metadata["sequence_number"] == 1
        assert doc2_chunks[1].metadata["sequence_number"] == 2

    def test_docling_to_langchain(self):
        docs = [
            _make_docling_doc("file1.pdf", "Content of file 1"),
            _make_docling_doc("file2.pdf", "Content of file 2"),
        ]
        lc_docs = LangChainChunker._docling_to_langchain(docs)
        assert len(lc_docs) == 2
        assert all(isinstance(d, Document) for d in lc_docs)
        assert lc_docs[0].metadata["document_id"] == "file1.pdf"
        assert lc_docs[1].metadata["document_id"] == "file2.pdf"


class TestLangChainChunkerEdgeCases:
    """Test suite for edge cases."""

    def test_chunker_with_very_small_chunk_size(self):
        chunker = LangChainChunker(chunk_size=5, chunk_overlap=1)
        docs = [_make_docling_doc("test.pdf", "This is a longer document.")]
        chunks = chunker.split_documents(docs)
        assert len(chunks) > 1

    def test_chunker_with_large_chunk_size(self):
        chunker = LangChainChunker(chunk_size=10000, chunk_overlap=100)
        docs = [_make_docling_doc("test.pdf", "Short document.")]
        chunks = chunker.split_documents(docs)
        assert len(chunks) >= 1

    def test_chunker_with_zero_overlap(self):
        chunker = LangChainChunker(chunk_size=20, chunk_overlap=0)
        docs = [_make_docling_doc("test.pdf", "A" * 100)]
        chunks = chunker.split_documents(docs)
        assert len(chunks) > 0

    def test_chunker_with_very_long_document(self):
        chunker = LangChainChunker(chunk_size=100, chunk_overlap=10)
        long_content = "This is a sentence. " * 1000
        docs = [_make_docling_doc("big.pdf", long_content)]
        chunks = chunker.split_documents(docs)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "document_id" in chunk.metadata
            assert "sequence_number" in chunk.metadata
