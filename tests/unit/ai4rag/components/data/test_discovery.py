# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

import json

import pytest

from ai4rag.components.data.documents_discovery import (
    DOCUMENTS_DESCRIPTOR_FILENAME,
    DiscoveryResult,
    DocumentDescriptor,
    discover_documents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _s3_object(key: str, size: int) -> dict:
    """Build a minimal S3 ``Contents`` entry."""
    return {"Key": key, "Size": size}


def _make_mock_s3_client(mocker, contents: list[dict]):
    """Return a mock S3 client whose ``list_objects_v2`` yields *contents*."""
    mock = mocker.MagicMock()
    mock.list_objects_v2.return_value = {"Contents": contents}
    return mock


# ---------------------------------------------------------------------------
# DocumentDescriptor
# ---------------------------------------------------------------------------


class TestDocumentDescriptor:
    """Tests for the ``DocumentDescriptor`` frozen dataclass."""

    def test_attributes(self):
        """Verify field values are stored correctly."""
        dd = DocumentDescriptor(key="docs/report.pdf", size_bytes=42)
        assert dd.key == "docs/report.pdf"
        assert dd.size_bytes == 42

    def test_frozen(self):
        """Frozen dataclass must reject attribute mutation."""
        dd = DocumentDescriptor(key="a.pdf", size_bytes=1)
        with pytest.raises(AttributeError):
            dd.key = "b.pdf"


# ---------------------------------------------------------------------------
# DiscoveryResult
# ---------------------------------------------------------------------------


class TestDiscoveryResult:
    """Tests for the ``DiscoveryResult`` dataclass."""

    @pytest.fixture
    def result(self) -> DiscoveryResult:
        """Minimal discovery result for reuse across tests."""
        docs = [
            DocumentDescriptor(key="a.pdf", size_bytes=100),
            DocumentDescriptor(key="b.docx", size_bytes=200),
        ]
        return DiscoveryResult(
            bucket="test-bucket",
            prefix="docs/",
            documents=docs,
            total_size_bytes=300,
            count=2,
        )

    def test_to_dict_structure(self, result: DiscoveryResult):
        """``to_dict`` must produce JSON-serialisable output with correct keys."""
        d = result.to_dict()
        assert d["bucket"] == "test-bucket"
        assert d["prefix"] == "docs/"
        assert d["total_size_bytes"] == 300
        assert d["count"] == 2
        assert len(d["documents"]) == 2
        assert d["documents"][0] == {"key": "a.pdf", "size_bytes": 100}
        assert d["documents"][1] == {"key": "b.docx", "size_bytes": 200}

    def test_to_dict_is_json_serialisable(self, result: DiscoveryResult):
        """``to_dict`` output must survive a JSON round-trip."""
        serialised = json.dumps(result.to_dict())
        assert json.loads(serialised) == result.to_dict()

    def test_save_creates_file(self, result: DiscoveryResult, tmp_path):
        """``save`` must write the descriptor JSON to the target directory."""
        result.save(tmp_path)
        descriptor_path = tmp_path / DOCUMENTS_DESCRIPTOR_FILENAME
        assert descriptor_path.exists()

        with open(descriptor_path) as fh:
            data = json.load(fh)
        assert data == result.to_dict()

    def test_save_creates_missing_directory(self, result: DiscoveryResult, tmp_path):
        """``save`` must create parent directories when they don't exist."""
        nested = tmp_path / "a" / "b" / "c"
        result.save(nested)
        assert (nested / DOCUMENTS_DESCRIPTOR_FILENAME).exists()


# ---------------------------------------------------------------------------
# discover_documents
# ---------------------------------------------------------------------------


class TestDiscoverDocuments:
    """Tests for the ``discover_documents`` function."""

    def test_happy_path_returns_all_supported(self, mocker):
        """All files with supported extensions are discovered."""
        contents = [
            _s3_object("docs/report.pdf", 500),
            _s3_object("docs/notes.md", 200),
            _s3_object("docs/slide.pptx", 300),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefix="docs/",
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 3
        assert result.total_size_bytes == 1000
        assert result.bucket == "bucket"
        assert result.prefix == "docs/"
        keys = [d.key for d in result.documents]
        assert "docs/report.pdf" in keys
        assert "docs/notes.md" in keys
        assert "docs/slide.pptx" in keys

    def test_unsupported_extensions_filtered_out(self, mocker):
        """Files with unsupported extensions must be excluded."""
        contents = [
            _s3_object("docs/report.pdf", 100),
            _s3_object("docs/image.png", 200),
            _s3_object("docs/archive.zip", 300),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 1
        assert result.documents[0].key == "docs/report.pdf"

    def test_sampling_respects_size_limit(self, mocker):
        """Size-based sampling must stop adding files once the limit is reached."""
        one_gb = int(1024**3)
        contents = [
            _s3_object("a.pdf", one_gb - 1),
            _s3_object("b.pdf", one_gb - 1),
            _s3_object("c.pdf", one_gb - 1),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            sampling_enabled=True,
            sampling_max_size_gb=1.0,
            s3_client=mock_client,
        )

        assert result.count == 1
        assert result.documents[0].key == "a.pdf"

    def test_sampling_disabled_returns_all(self, mocker):
        """With sampling disabled, all supported files are returned regardless of size."""
        one_gb = int(1024**3)
        contents = [
            _s3_object("a.pdf", one_gb * 2),
            _s3_object("b.pdf", one_gb * 3),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 2

    def test_test_data_doc_names_prioritised(self, mocker):
        """Documents referenced by ``test_data_doc_names`` must be sorted first."""
        contents = [
            _s3_object("docs/other.pdf", 100),
            _s3_object("docs/benchmark.pdf", 100),
            _s3_object("docs/important.md", 100),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefix="docs/",
            test_data_doc_names=["benchmark.pdf", "important.md"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        prioritised_keys = [d.key for d in result.documents[:2]]
        assert "docs/benchmark.pdf" in prioritised_keys
        assert "docs/important.md" in prioritised_keys

    def test_test_data_prioritised_under_sampling(self, mocker):
        """Prioritised docs should survive size-based sampling."""
        contents = [
            _s3_object("docs/large.pdf", 500),
            _s3_object("docs/benchmark.pdf", 400),
            _s3_object("docs/other.pdf", 500),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefix="docs/",
            test_data_doc_names=["benchmark.pdf"],
            sampling_enabled=True,
            sampling_max_size_gb=900 / 1024**3,
            s3_client=mock_client,
        )

        keys = [d.key for d in result.documents]
        assert "docs/benchmark.pdf" in keys

    def test_no_supported_files_raises_runtime_error(self, mocker):
        """RuntimeError must be raised when the bucket has no supported files."""
        contents = [
            _s3_object("data/image.png", 100),
            _s3_object("data/archive.tar.gz", 200),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        with pytest.raises(RuntimeError, match="No supported documents found"):
            discover_documents(
                bucket_name="bucket",
                s3_client=mock_client,
            )

    def test_empty_bucket_raises_runtime_error(self, mocker):
        """RuntimeError for an entirely empty bucket."""
        mock_client = _make_mock_s3_client(mocker, [])

        with pytest.raises(RuntimeError, match="No supported documents found"):
            discover_documents(bucket_name="bucket", s3_client=mock_client)

    def test_all_files_exceed_sampling_budget_raises_value_error(self, mocker):
        """ValueError when every file individually exceeds the size budget."""
        two_gb = int(2 * 1024**3)
        contents = [
            _s3_object("a.pdf", two_gb),
            _s3_object("b.pdf", two_gb),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        with pytest.raises(ValueError, match="No documents to process"):
            discover_documents(
                bucket_name="bucket",
                sampling_enabled=True,
                sampling_max_size_gb=1.0,
                s3_client=mock_client,
            )

    def test_custom_supported_extensions(self, mocker):
        """A custom ``supported_extensions`` set should override the defaults."""
        contents = [
            _s3_object("a.csv", 100),
            _s3_object("b.pdf", 200),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            sampling_enabled=False,
            supported_extensions={".csv"},
            s3_client=mock_client,
        )

        assert result.count == 1
        assert result.documents[0].key == "a.csv"

    def test_list_objects_called_correctly(self, mocker):
        """``list_objects_v2`` must receive the correct bucket and prefix."""
        contents = [_s3_object("prefix/x.pdf", 10)]
        mock_client = _make_mock_s3_client(mocker, contents)

        discover_documents(
            bucket_name="my-bucket",
            prefix="prefix/",
            sampling_enabled=False,
            s3_client=mock_client,
        )

        mock_client.list_objects_v2.assert_called_once_with(Bucket="my-bucket", Prefix="prefix/")
