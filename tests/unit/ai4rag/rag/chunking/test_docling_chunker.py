# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.chunking.docling_chunker import DoclingChunker


def _make_doc(name: str, sections: list[tuple[str, str]] | None = None) -> DoclingDocument:
    """Helper: create a DoclingDocument with optional heading-paragraph pairs."""
    doc = DoclingDocument(name=name)
    if sections:
        for heading, paragraph in sections:
            doc.add_heading(text=heading, level=1)
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=paragraph)
    return doc


class TestDoclingChunkerInitialization:

    def test_init_with_defaults(self):
        chunker = DoclingChunker()
        assert chunker.max_tokens == 8192
        assert chunker.contextualize is True
        assert chunker.merge_peers is True

    def test_init_with_custom_parameters(self):
        chunker = DoclingChunker(max_tokens=512, contextualize=False, merge_peers=False)
        assert chunker.max_tokens == 512
        assert chunker.contextualize is False
        assert chunker.merge_peers is False

    def test_init_creates_hybrid_chunker(self):
        chunker = DoclingChunker()
        assert chunker._chunker is not None


class TestDoclingChunkerSplitDocuments:

    @pytest.fixture
    def doc_with_sections(self):
        return _make_doc(
            "report.pdf",
            [
                ("Introduction", "Machine learning is a branch of AI."),
                ("Methods", "We used gradient descent for optimization."),
                ("Results", "The model achieved 95% accuracy."),
            ],
        )

    @pytest.fixture
    def chunker(self):
        return DoclingChunker(max_tokens=8192)

    def test_returns_ai4rag_chunks(self, chunker, doc_with_sections):
        chunks = chunker.split_documents([doc_with_sections])
        assert isinstance(chunks, list)
        assert all(isinstance(c, AI4RAGChunk) for c in chunks)

    def test_document_id_from_doc_name(self, chunker, doc_with_sections):
        chunks = chunker.split_documents([doc_with_sections])
        for chunk in chunks:
            assert chunk.metadata["document_id"] == "report.pdf"

    def test_sequence_numbers_start_at_one(self, chunker, doc_with_sections):
        chunks = chunker.split_documents([doc_with_sections])
        seq_nums = [c.metadata["sequence_number"] for c in chunks]
        assert seq_nums[0] == 1
        assert seq_nums == list(range(1, len(chunks) + 1))

    def test_contextualize_true_includes_headings(self, doc_with_sections):
        chunker = DoclingChunker(contextualize=True)
        chunks = chunker.split_documents([doc_with_sections])
        heading_chunks = [c for c in chunks if c.metadata.get("headings")]
        assert len(heading_chunks) > 0
        for chunk in heading_chunks:
            assert chunk.metadata["headings"][0] in chunk.text

    def test_contextualize_false_excludes_headings(self, doc_with_sections):
        chunker = DoclingChunker(contextualize=False)
        chunks = chunker.split_documents([doc_with_sections])
        heading_chunks = [c for c in chunks if c.metadata.get("headings")]
        for chunk in heading_chunks:
            heading = chunk.metadata["headings"][0]
            assert not chunk.text.startswith(heading)

    def test_headings_in_metadata(self, chunker, doc_with_sections):
        chunks = chunker.split_documents([doc_with_sections])
        heading_chunks = [c for c in chunks if c.metadata.get("headings")]
        heading_values = [c.metadata["headings"][0] for c in heading_chunks]
        assert "Introduction" in heading_values
        assert "Methods" in heading_values
        assert "Results" in heading_values

    def test_multiple_documents(self, chunker):
        doc1 = _make_doc("doc1.pdf", [("Intro", "Content one.")])
        doc2 = _make_doc("doc2.pdf", [("Intro", "Content two.")])
        chunks = chunker.split_documents([doc1, doc2])
        doc_ids = {c.metadata["document_id"] for c in chunks}
        assert doc_ids == {"doc1.pdf", "doc2.pdf"}

    def test_sequence_numbers_reset_per_document(self, chunker):
        doc1 = _make_doc("doc1.pdf", [("A", "Text A."), ("B", "Text B.")])
        doc2 = _make_doc("doc2.pdf", [("C", "Text C.")])
        chunks = chunker.split_documents([doc1, doc2])
        doc2_chunks = [c for c in chunks if c.metadata["document_id"] == "doc2.pdf"]
        assert doc2_chunks[0].metadata["sequence_number"] == 1

    def test_empty_input(self, chunker):
        chunks = chunker.split_documents([])
        assert chunks == []


class TestDoclingChunkerSerialization:

    def test_to_dict(self):
        chunker = DoclingChunker(max_tokens=512, contextualize=False, merge_peers=True)
        d = chunker.to_dict()
        assert d == {"max_tokens": 512, "contextualize": False, "merge_peers": True}

    def test_from_dict(self):
        d = {"max_tokens": 1024, "contextualize": True, "merge_peers": False}
        chunker = DoclingChunker.from_dict(d)
        assert chunker.max_tokens == 1024
        assert chunker.contextualize is True
        assert chunker.merge_peers is False

    def test_round_trip(self):
        original = DoclingChunker(max_tokens=2048, contextualize=False)
        recreated = DoclingChunker.from_dict(original.to_dict())
        assert original == recreated


class TestDoclingChunkerEquality:

    def test_equal_instances(self):
        c1 = DoclingChunker(max_tokens=512)
        c2 = DoclingChunker(max_tokens=512)
        assert c1 == c2

    def test_different_max_tokens(self):
        c1 = DoclingChunker(max_tokens=512)
        c2 = DoclingChunker(max_tokens=1024)
        assert c1 != c2

    def test_different_contextualize(self):
        c1 = DoclingChunker(contextualize=True)
        c2 = DoclingChunker(contextualize=False)
        assert c1 != c2

    def test_non_chunker_comparison(self):
        assert DoclingChunker().__eq__("not a chunker") is NotImplemented
