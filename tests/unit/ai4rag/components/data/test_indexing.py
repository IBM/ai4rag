# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from pathlib import Path

import pytest

from ai4rag.components.data.documents_indexing import (
    SUPPORTED_CHUNK_SIZE_RANGE,
    SUPPORTED_DISTANCE_METRICS,
    index_documents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_json_files(directory: Path, count: int) -> list[Path]:
    """Write *count* dummy ``.json`` files into *directory* and return their paths."""
    paths = []
    for i in range(count):
        p = directory / f"doc_{i}.json"
        p.write_text("{}")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestIndexDocumentsValidation:
    """Validation of user-facing parameters in ``index_documents``."""

    @pytest.fixture
    def populated_dir(self, tmp_path) -> Path:
        """Directory with a single dummy JSON file so we reach validation."""
        _create_json_files(tmp_path, 1)
        return tmp_path

    @pytest.fixture
    def mock_ogx_client(self, mocker):
        """Minimal mock OgxClient."""
        return mocker.MagicMock()

    def test_empty_embedding_model_id_raises(self, populated_dir, mock_ogx_client):
        """An empty ``embedding_model_id`` must be rejected."""
        with pytest.raises(ValueError, match="embedding_model_id"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="",
                vector_io_provider_id="provider",
                ogx_client=mock_ogx_client,
            )

    def test_empty_vector_io_provider_id_raises(self, populated_dir, mock_ogx_client):
        """An empty ``vector_io_provider_id`` must be rejected."""
        with pytest.raises(ValueError, match="vector_io_provider_id"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="model-x",
                vector_io_provider_id="",
                ogx_client=mock_ogx_client,
            )

    def test_whitespace_vector_io_provider_id_raises(self, populated_dir, mock_ogx_client):
        """A whitespace-only ``vector_io_provider_id`` must be rejected."""
        with pytest.raises(ValueError, match="vector_io_provider_id"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="model-x",
                vector_io_provider_id="   ",
                ogx_client=mock_ogx_client,
            )

    @pytest.mark.parametrize("bad_metric", ["manhattan", "dot", ""])
    def test_unsupported_distance_metric_raises(self, populated_dir, mock_ogx_client, bad_metric):
        """Unsupported distance metrics must be rejected."""
        with pytest.raises(ValueError, match="distance metric"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="model-x",
                vector_io_provider_id="provider",
                ogx_client=mock_ogx_client,
                distance_metric=bad_metric,
            )

    @pytest.mark.parametrize("bad_method", ["sliding_window", "semantic", ""])
    def test_unsupported_chunking_method_raises(self, populated_dir, mock_ogx_client, bad_method):
        """Unsupported chunking methods must be rejected."""
        with pytest.raises(ValueError, match="chunking_method"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="model-x",
                vector_io_provider_id="provider",
                ogx_client=mock_ogx_client,
                chunking_method=bad_method,
            )

    @pytest.mark.parametrize("bad_size", [0, 64, 127, 2049, 5000])
    def test_out_of_range_chunk_size_raises(self, populated_dir, mock_ogx_client, bad_size):
        """Chunk sizes outside the allowed range must be rejected."""
        with pytest.raises(ValueError, match="chunk_size"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="model-x",
                vector_io_provider_id="provider",
                ogx_client=mock_ogx_client,
                chunk_size=bad_size,
            )

    def test_non_int_chunk_size_raises(self, populated_dir, mock_ogx_client):
        """A non-integer ``chunk_size`` must raise ``TypeError``."""
        with pytest.raises(TypeError, match="chunk_size"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="model-x",
                vector_io_provider_id="provider",
                ogx_client=mock_ogx_client,
                chunk_size=512.5,
            )

    def test_non_numeric_chunk_overlap_raises(self, populated_dir, mock_ogx_client):
        """A non-numeric ``chunk_overlap`` must raise ``TypeError``."""
        with pytest.raises(TypeError, match="chunk_overlap"):
            index_documents(
                extracted_text_dir=populated_dir,
                embedding_model_id="model-x",
                vector_io_provider_id="provider",
                ogx_client=mock_ogx_client,
                chunk_overlap="abc",
            )

    @pytest.mark.parametrize("valid_metric", list(SUPPORTED_DISTANCE_METRICS))
    def test_valid_distance_metrics_accepted(self, populated_dir, mock_ogx_client, mocker, valid_metric):
        """All supported distance metrics must pass validation (verified by reaching next stage)."""
        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingParams")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingModel")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXVectorStore")
        mocker.patch("ai4rag.components.data.documents_indexing.DoclingDocument")
        mocker.patch("ai4rag.components.data.documents_indexing.LangChainChunker")

        index_documents(
            extracted_text_dir=populated_dir,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mock_ogx_client,
            distance_metric=valid_metric,
        )

    @pytest.mark.parametrize("valid_size", [SUPPORTED_CHUNK_SIZE_RANGE[0], 512, 1024, SUPPORTED_CHUNK_SIZE_RANGE[1]])
    def test_boundary_chunk_sizes_accepted(self, populated_dir, mock_ogx_client, mocker, valid_size):
        """Chunk sizes at the boundaries must be accepted."""
        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingParams")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingModel")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXVectorStore")
        mocker.patch("ai4rag.components.data.documents_indexing.DoclingDocument")
        mocker.patch("ai4rag.components.data.documents_indexing.LangChainChunker")

        index_documents(
            extracted_text_dir=populated_dir,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mock_ogx_client,
            chunk_size=valid_size,
        )


# ---------------------------------------------------------------------------
# Batch processing and empty-directory handling
# ---------------------------------------------------------------------------


class TestIndexDocumentsProcessing:
    """Tests for the indexing pipeline's processing logic."""

    @pytest.fixture
    def _patch_dependencies(self, mocker):
        """Patch all heavy dependencies so tests stay fast and isolated.

        Returns a dict with the mocks for assertions.
        """
        mock_params_cls = mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingParams")
        mock_emb_cls = mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingModel")
        mock_vs_cls = mocker.patch("ai4rag.components.data.documents_indexing.OGXVectorStore")
        mock_docling = mocker.patch("ai4rag.components.data.documents_indexing.DoclingDocument")
        mock_lang_chunker_cls = mocker.patch("ai4rag.components.data.documents_indexing.LangChainChunker")

        mock_doc = mocker.MagicMock()
        mock_docling.load_from_json.return_value = mock_doc

        mock_chunker = mock_lang_chunker_cls.return_value
        mock_chunker.split_documents.return_value = [mocker.MagicMock(), mocker.MagicMock()]

        return {
            "params_cls": mock_params_cls,
            "embedding_cls": mock_emb_cls,
            "vector_store_cls": mock_vs_cls,
            "docling": mock_docling,
            "chunker": mock_chunker,
        }

    def test_empty_directory_returns_zero(self, tmp_path, mocker):
        """An empty directory must return 0 chunks without raising."""
        mock_client = mocker.MagicMock()

        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingParams")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingModel")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXVectorStore")

        result = index_documents(
            extracted_text_dir=tmp_path,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mock_client,
        )

        assert result == 0

    def test_processes_all_documents_single_batch(self, tmp_path, mocker, _patch_dependencies):
        """A small set of documents should be processed in one batch."""
        _create_json_files(tmp_path, 3)
        mocks = _patch_dependencies

        total = index_documents(
            extracted_text_dir=tmp_path,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mocker.MagicMock(),
            batch_size=0,
        )

        # Each call to split_documents returns 2 mock chunks; 1 batch call
        assert mocks["chunker"].split_documents.call_count == 1
        assert mocks["docling"].load_from_json.call_count == 3
        assert total == 2

    def test_processes_documents_in_batches(self, tmp_path, mocker, _patch_dependencies):
        """Documents must be split into multiple batches when ``batch_size`` is smaller."""
        _create_json_files(tmp_path, 5)
        mocks = _patch_dependencies

        total = index_documents(
            extracted_text_dir=tmp_path,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mocker.MagicMock(),
            batch_size=2,
        )

        # 5 docs, batch_size=2 -> 3 batches (2+2+1)
        assert mocks["chunker"].split_documents.call_count == 3
        # 3 batches * 2 chunks each = 6 total
        assert total == 6

    def test_non_json_files_ignored(self, tmp_path, mocker, _patch_dependencies):
        """Only ``.json`` files should be processed; others must be skipped."""
        _create_json_files(tmp_path, 2)
        (tmp_path / "readme.txt").write_text("ignore me")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        total = index_documents(
            extracted_text_dir=tmp_path,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mocker.MagicMock(),
            batch_size=0,
        )

        assert _patch_dependencies["docling"].load_from_json.call_count == 2
        assert total == 2

    def test_vector_store_receives_collection_name(self, tmp_path, mocker, _patch_dependencies):
        """When ``collection_name`` is given, it must be forwarded to OGXVectorStore."""
        _create_json_files(tmp_path, 1)

        index_documents(
            extracted_text_dir=tmp_path,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mocker.MagicMock(),
            collection_name="my-collection",
        )

        call_kwargs = _patch_dependencies["vector_store_cls"].call_args[1]
        assert call_kwargs["reuse_collection_name"] == "my-collection"

    def test_hybrid_chunker_selected(self, tmp_path, mocker):
        """``chunking_method='hybrid'`` must instantiate ``DoclingChunker``."""
        _create_json_files(tmp_path, 1)

        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingParams")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXEmbeddingModel")
        mocker.patch("ai4rag.components.data.documents_indexing.OGXVectorStore")
        mocker.patch("ai4rag.components.data.documents_indexing.DoclingDocument")
        mock_docling_chunker = mocker.patch("ai4rag.components.data.documents_indexing.DoclingChunker")
        mock_docling_chunker.return_value.split_documents.return_value = []

        index_documents(
            extracted_text_dir=tmp_path,
            embedding_model_id="model-x",
            vector_io_provider_id="provider",
            ogx_client=mocker.MagicMock(),
            chunking_method="hybrid",
            chunk_size=256,
        )

        mock_docling_chunker.assert_called_once_with(max_tokens=256)
