# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from ai4rag.rag.chunking.contextual_chunker import (
    ContextualChunker,
    _build_chunks_section,
    _parse_batch_response,
)
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker


def _make_mock_model(response_text="A test context."):
    """Create a mock foundation model that returns the given text."""
    model = MagicMock()
    model.model_id = "test-model"
    choice = MagicMock()
    choice.message.content = response_text
    model.chat.return_value = [choice]
    return model


def _make_json_batch_response(contexts):
    """Create a JSON object batch response (matches json_object mode output)."""
    items = [{"id": i + 1, "context": ctx} for i, ctx in enumerate(contexts)]
    return json.dumps({"contexts": items})


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="The company reported strong growth in Q3. Revenue increased by 15%.",
            metadata={"document_id": "doc1"},
        ),
        Document(
            page_content="Environmental regulations require annual compliance audits.",
            metadata={"document_id": "doc2"},
        ),
    ]


@pytest.fixture
def base_chunker():
    return LangChainChunker(method="recursive", chunk_size=50, chunk_overlap=10)


class TestContextualChunkerInit:

    def test_default_parameters(self, base_chunker):
        model = _make_mock_model()
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model)
        assert chunker.base_chunker is base_chunker
        assert chunker.context_model is model
        assert chunker.max_context_tokens == 100
        assert chunker.batch_size == 10
        assert chunker.max_workers == 4

    def test_custom_parameters(self, base_chunker):
        model = _make_mock_model()
        chunker = ContextualChunker(
            base_chunker=base_chunker,
            context_model=model,
            max_context_tokens=50,
            batch_size=5,
            max_workers=2,
            prompt_template="custom {chunk}",
        )
        assert chunker.max_context_tokens == 50
        assert chunker.batch_size == 5
        assert chunker.max_workers == 2
        assert chunker.prompt_template == "custom {chunk}"


class TestContextualChunkerSplitDocuments:

    def test_chunks_get_context_prepended(self, sample_documents, base_chunker):
        model = _make_mock_model("This chunk is about revenue.")
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1)

        result = chunker.split_documents(sample_documents)

        assert len(result) > 0
        for chunk in result:
            assert chunk.page_content.startswith("[Context: This chunk is about revenue.]")
            assert chunk.metadata["contextualized"] is True
            assert "original_page_content" in chunk.metadata

    def test_original_content_preserved_in_metadata(self, sample_documents, base_chunker):
        model = _make_mock_model("Some context.")
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1)

        raw_chunks = base_chunker.split_documents(
            [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in sample_documents]
        )
        result = chunker.split_documents(sample_documents)

        for res_chunk, raw_chunk in zip(result, raw_chunks):
            assert res_chunk.metadata["original_page_content"] == raw_chunk.page_content

    def test_metadata_preserved(self, sample_documents, base_chunker):
        model = _make_mock_model("context")
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1)
        result = chunker.split_documents(sample_documents)

        for chunk in result:
            assert "document_id" in chunk.metadata
            assert "sequence_number" in chunk.metadata

    def test_batch_mode(self, sample_documents, base_chunker):
        batch_response = _make_json_batch_response(
            ["context for chunk 1", "context for chunk 2", "context for chunk 3"]
        )
        model = _make_mock_model(batch_response)
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=20, max_workers=1)

        result = chunker.split_documents(sample_documents)

        assert len(result) > 0
        for chunk in result:
            assert "contextualized" in chunk.metadata

    def test_parallel_processing(self, sample_documents, base_chunker):
        model = _make_mock_model("parallel context")
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=4)

        result = chunker.split_documents(sample_documents)

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["contextualized"] is True

    def test_system_message_contains_document(self, base_chunker):
        doc = Document(page_content="Important document content here.", metadata={"document_id": "doc1"})
        model = _make_mock_model("context")
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1)

        chunker.split_documents([doc])

        call_args = model.chat.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert "Important document content here." in call_args[0]["content"]
        assert call_args[1]["role"] == "user"


class TestContextualChunkerDocumentSizeGuard:

    def test_large_document_skipped(self, base_chunker):
        large_doc = Document(page_content="x" * 200_000, metadata={"document_id": "big_doc"})
        model = _make_mock_model("should not be called")
        chunker = ContextualChunker(
            base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1, max_document_size=100_000
        )

        result = chunker.split_documents([large_doc])

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["contextualized"] is False
            assert not chunk.page_content.startswith("[Context:")
            assert "original_page_content" in chunk.metadata
        model.chat.assert_not_called()

    def test_small_document_enriched(self, base_chunker):
        small_doc = Document(page_content="A short document with content.", metadata={"document_id": "small_doc"})
        model = _make_mock_model("some context")
        chunker = ContextualChunker(
            base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1, max_document_size=100_000
        )

        result = chunker.split_documents([small_doc])

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["contextualized"] is True

    def test_mixed_documents(self, base_chunker):
        large_doc = Document(page_content="x" * 200_000, metadata={"document_id": "big"})
        small_doc = Document(page_content="A small document with text.", metadata={"document_id": "small"})
        model = _make_mock_model("context")
        chunker = ContextualChunker(
            base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1, max_document_size=100_000
        )

        result = chunker.split_documents([large_doc, small_doc])

        big_chunks = [c for c in result if c.metadata["document_id"] == "big"]
        small_chunks = [c for c in result if c.metadata["document_id"] == "small"]

        assert all(c.metadata["contextualized"] is False for c in big_chunks)
        assert all(c.metadata["contextualized"] is True for c in small_chunks)

    def test_custom_max_document_size(self, base_chunker):
        doc = Document(page_content="x" * 500, metadata={"document_id": "doc1"})
        model = _make_mock_model("context")
        chunker = ContextualChunker(
            base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1, max_document_size=100
        )

        result = chunker.split_documents([doc])

        for chunk in result:
            assert chunk.metadata["contextualized"] is False
        model.chat.assert_not_called()


class TestContextualChunkerFallback:

    def test_single_chunk_failure_uses_original(self, sample_documents, base_chunker):
        model = _make_mock_model()
        model.chat.side_effect = RuntimeError("API error")
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=1, max_workers=1)

        result = chunker.split_documents(sample_documents)

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["contextualized"] is False
            assert not chunk.page_content.startswith("[Context:")

    def test_batch_failure_falls_back_to_single(self, sample_documents, base_chunker):
        call_count = 0

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Batch API error")
            choice = MagicMock()
            choice.message.content = "fallback context"
            return [choice]

        model = _make_mock_model()
        model.chat.side_effect = side_effect
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=20, max_workers=1)

        result = chunker.split_documents(sample_documents)

        assert len(result) > 0
        contextualized = [c for c in result if c.metadata["contextualized"]]
        assert len(contextualized) > 0

    def test_partial_batch_parse_retries_missing(self, base_chunker):
        doc = Document(
            page_content="Word " * 100,
            metadata={"document_id": "doc1"},
        )

        call_count = 0

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            choice = MagicMock()
            if call_count == 1:
                # JSON response with only first chunk
                choice.message.content = json.dumps({"contexts": [{"id": 1, "context": "first context"}]})
            else:
                choice.message.content = "retry context"
            return [choice]

        model = _make_mock_model()
        model.chat.side_effect = side_effect
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, batch_size=20, max_workers=1)

        result = chunker.split_documents([doc])

        contextualized = [c for c in result if c.metadata["contextualized"]]
        assert len(contextualized) == len(result)


class TestParseBatchResponse:

    def test_parse_json_object_with_contexts_key(self):
        response = json.dumps(
            {"contexts": [{"id": 1, "context": "First context"}, {"id": 2, "context": "Second context"}]}
        )
        result = _parse_batch_response(response, 2)
        assert result == ["First context", "Second context"]

    def test_parse_json_object_with_arbitrary_key(self):
        response = json.dumps({"results": [{"id": 1, "context": "Found it"}]})
        result = _parse_batch_response(response, 1)
        assert result == ["Found it"]

    def test_parse_top_level_array(self):
        response = json.dumps([{"id": 1, "context": "First"}, {"id": 2, "context": "Second"}])
        result = _parse_batch_response(response, 2)
        assert result == ["First", "Second"]

    def test_parse_missing_entries(self):
        response = json.dumps({"contexts": [{"id": 1, "context": "Only first"}]})
        result = _parse_batch_response(response, 3)
        assert result == ["Only first", None, None]

    def test_parse_empty_response(self):
        result = _parse_batch_response("", 2)
        assert result == [None, None]

    def test_parse_invalid_json(self):
        result = _parse_batch_response("not json at all", 2)
        assert result == [None, None]

    def test_parse_ignores_out_of_range_ids(self):
        response = json.dumps({"contexts": [{"id": 1, "context": "valid"}, {"id": 99, "context": "out of range"}]})
        result = _parse_batch_response(response, 2)
        assert result == ["valid", None]

    def test_parse_json_in_markdown_fences(self):
        response = '```json\n{"contexts": [{"id": 1, "context": "fenced"}]}\n```'
        result = _parse_batch_response(response, 1)
        assert result == ["fenced"]

    def test_parse_dict_without_list_value(self):
        response = json.dumps({"id": 1, "context": "no list inside"})
        result = _parse_batch_response(response, 1)
        assert result == [None]


class TestBuildHelpers:

    def test_build_chunks_section(self):
        chunks = [
            Document(page_content="chunk one", metadata={}),
            Document(page_content="chunk two", metadata={}),
        ]
        result = _build_chunks_section(chunks)
        assert '<chunk id="1">' in result
        assert '<chunk id="2">' in result
        assert "chunk one" in result
        assert "chunk two" in result


class TestContextualChunkerSerialization:

    def test_to_dict(self, base_chunker):
        model = _make_mock_model()
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model, max_context_tokens=80, batch_size=5)
        d = chunker.to_dict()
        assert d["context_model_id"] == "test-model"
        assert d["max_context_tokens"] == 80
        assert d["batch_size"] == 5
        assert "base_chunker" in d

    def test_from_dict_raises(self, base_chunker):
        model = _make_mock_model()
        chunker = ContextualChunker(base_chunker=base_chunker, context_model=model)
        with pytest.raises(NotImplementedError):
            ContextualChunker.from_dict(chunker.to_dict())
