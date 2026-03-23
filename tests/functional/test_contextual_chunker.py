# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional tests for ContextualChunker against a live Llama Stack server."""

import os
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv
from llama_stack_client import LlamaStackClient

from ai4rag.rag.chunking.contextual_chunker import ContextualChunker
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from dev_utils.file_store import FileStore

load_dotenv(find_dotenv())


DATA_PATH = os.environ.get("AI4RAG_TEST_DATA_PATH")


pytestmark = pytest.mark.skipif(
    DATA_PATH is None,
    reason="AI4RAG_TEST_DATA_PATH environment variable not set",
)


@pytest.fixture(scope="module")
def client():
    return LlamaStackClient(
        base_url=os.environ["LLAMA_STACK_CLIENT_BASE_URL"],
        api_key=os.environ["LLAMA_STACK_CLIENT_API_KEY"],
    )


@pytest.fixture(scope="module")
def documents():
    documents_path = Path(os.path.join(DATA_PATH, "documents"))
    file_store = FileStore(documents_path)
    return file_store.load_as_documents()


@pytest.fixture(scope="module")
def foundation_model(client):
    model_id = os.environ.get("AI4RAG_TEST_FOUNDATION_MODEL", "vllm-inference-llama-3-1/redhataillama-31-8b-instruct")
    return LSFoundationModel(model_id=model_id, client=client)


@pytest.fixture(scope="module")
def base_chunker():
    return LangChainChunker(method="recursive", chunk_size=1024, chunk_overlap=128)


class TestContextualChunkerSingleMode:
    """Test ContextualChunker with batch_size=1 (one LLM call per chunk)."""

    def test_single_chunk_context_generation(self, documents, foundation_model, base_chunker):
        chunker = ContextualChunker(
            base_chunker=base_chunker,
            context_model=foundation_model,
            batch_size=1,
            max_workers=1,
        )

        # Use only first document to keep the test fast
        test_docs = documents[:1]
        result = chunker.split_documents(test_docs)

        assert len(result) > 0

        for chunk in result:
            assert "contextualized" in chunk.metadata
            assert "original_page_content" in chunk.metadata

        contextualized = [c for c in result if c.metadata["contextualized"]]
        assert len(contextualized) > 0, "At least some chunks should be contextualized"

        for chunk in contextualized:
            assert chunk.page_content.startswith("[Context:")
            assert len(chunk.page_content) > len(chunk.metadata["original_page_content"])
            # Original content should still be present after the context prefix
            assert chunk.metadata["original_page_content"] in chunk.page_content


class TestContextualChunkerBatchMode:
    """Test ContextualChunker with batched LLM calls."""

    def test_batch_context_generation(self, documents, foundation_model, base_chunker):
        chunker = ContextualChunker(
            base_chunker=base_chunker,
            context_model=foundation_model,
            batch_size=5,
            max_workers=1,
        )

        test_docs = documents[:1]
        result = chunker.split_documents(test_docs)

        assert len(result) > 0

        contextualized = [c for c in result if c.metadata["contextualized"]]
        assert len(contextualized) > 0, "At least some chunks should be contextualized via batch"

        for chunk in contextualized:
            assert chunk.page_content.startswith("[Context:")
            assert chunk.metadata["original_page_content"] in chunk.page_content

    def test_batch_preserves_metadata(self, documents, foundation_model, base_chunker):
        chunker = ContextualChunker(
            base_chunker=base_chunker,
            context_model=foundation_model,
            batch_size=5,
            max_workers=1,
        )

        test_docs = documents[:1]
        result = chunker.split_documents(test_docs)

        for chunk in result:
            assert "document_id" in chunk.metadata
            assert "sequence_number" in chunk.metadata
            assert "start_index" in chunk.metadata


class TestContextualChunkerParallel:
    """Test ContextualChunker with parallel processing across documents."""

    def test_parallel_multiple_documents(self, documents, foundation_model, base_chunker):
        chunker = ContextualChunker(
            base_chunker=base_chunker,
            context_model=foundation_model,
            batch_size=3,
            max_workers=4,
        )

        test_docs = documents[:3] if len(documents) >= 3 else documents
        result = chunker.split_documents(test_docs)

        assert len(result) > 0

        doc_ids = {c.metadata["document_id"] for c in result}
        assert len(doc_ids) == len(test_docs), "Chunks should come from all input documents"

        contextualized = [c for c in result if c.metadata["contextualized"]]
        assert len(contextualized) > 0


class TestContextualChunkerDocumentSizeGuard:
    """Test that oversized documents are skipped gracefully."""

    def test_large_document_skipped_small_limit(self, documents, foundation_model):
        small_chunker = LangChainChunker(method="recursive", chunk_size=128, chunk_overlap=16)
        chunker = ContextualChunker(
            base_chunker=small_chunker,
            context_model=foundation_model,
            batch_size=1,
            max_workers=1,
            max_document_size=10,  # Very small limit — all docs should be skipped
        )

        test_docs = documents[:1]
        result = chunker.split_documents(test_docs)

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["contextualized"] is False
            assert not chunk.page_content.startswith("[Context:")
