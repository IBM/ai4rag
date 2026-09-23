# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json

import pytest

from ai4rag.utils.data.documents_discovery import (
    DOCUMENTS_DESCRIPTOR_FILENAME,
    BenchmarkKeyError,
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
    """Return a mock S3 client whose paginator yields the objects under a prefix.

    Mirrors the real API: ``paginate`` returns only the objects whose key starts
    with the requested prefix, in a single page.
    """
    mock = mocker.MagicMock()

    def _paginate(Bucket, Prefix, **_kwargs):  # noqa: N803 - boto3 kwarg names
        return iter([{"Contents": [c for c in contents if c["Key"].startswith(Prefix)]}])

    mock.get_paginator.return_value.paginate.side_effect = _paginate
    return mock


def _make_paginated_mock_s3_client(mocker, pages: list[list[dict]]):
    """Return a mock S3 client whose paginator yields *pages* verbatim."""
    mock = mocker.MagicMock()
    mock.get_paginator.return_value.paginate.side_effect = lambda **_kwargs: iter(
        [{"Contents": page} for page in pages]
    )
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
            prefixes=("docs/",),
            documents=docs,
            total_size_bytes=300,
            count=2,
        )

    def test_to_dict_structure(self, result: DiscoveryResult):
        """``to_dict`` must produce JSON-serialisable output with correct keys."""
        d = result.to_dict()
        assert d["bucket"] == "test-bucket"
        assert d["prefixes"] == ["docs/"]
        assert d["total_size_bytes"] == 300
        assert d["count"] == 2
        assert len(d["documents"]) == 2
        assert d["documents"][0] == {"key": "a.pdf", "size_bytes": 100}
        assert d["documents"][1] == {"key": "b.docx", "size_bytes": 200}

    def test_to_dict_is_json_serialisable(self, result: DiscoveryResult):
        """``to_dict`` output must survive a JSON round-trip."""
        serialised = json.dumps(result.to_dict())
        assert json.loads(serialised) == result.to_dict()

    def test_prefixes_are_immutable_when_constructed_with_a_list(self):
        """A frozen result must not expose a mutable prefix collection."""
        result = DiscoveryResult(
            bucket="test-bucket",
            prefixes=["docs/"],
            documents=[],
            total_size_bytes=0,
            count=0,
        )

        assert result.prefixes == ("docs/",)
        with pytest.raises(AttributeError):
            result.prefixes.append("evil/")

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
            _s3_object("docs/file.odt", 100),
            _s3_object("docs/file.odp", 100),
            _s3_object("docs/file.adoc", 100),
            _s3_object("docs/file.tex", 100),
            _s3_object("docs/file.epub", 100),
            _s3_object("docs/file.eml", 100),
            _s3_object("docs/file.qmd", 100),
            _s3_object("docs/file.rmd", 100),
            _s3_object("docs/file.xhtml", 100),
            _s3_object("docs/file.msg", 100),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["docs/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 13
        assert result.total_size_bytes == 2000
        assert result.bucket == "bucket"
        assert result.prefixes == ("docs/",)
        keys = [d.key for d in result.documents]
        assert "docs/report.pdf" in keys
        assert "docs/notes.md" in keys
        assert "docs/slide.pptx" in keys
        assert "docs/file.odt" in keys
        assert "docs/file.odp" in keys
        assert "docs/file.adoc" in keys
        assert "docs/file.tex" in keys
        assert "docs/file.epub" in keys
        assert "docs/file.eml" in keys
        assert "docs/file.qmd" in keys
        assert "docs/file.rmd" in keys
        assert "docs/file.xhtml" in keys
        assert "docs/file.msg" in keys

    def test_unsupported_extensions_filtered_out(self, mocker):
        """Files with unsupported extensions must be excluded."""
        contents = [
            _s3_object("docs/report.pdf", 100),
            _s3_object("docs/photo.jpg", 200),
            _s3_object("docs/archive.zip", 300),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            sampling_enabled=False,
            s3_client=mock_client,
        )

        keys = [d.key for d in result.documents]
        assert result.count == 2
        assert "docs/report.pdf" in keys
        assert "docs/photo.jpg" in keys
        assert "docs/archive.zip" not in keys

    def test_image_extensions_are_supported(self, mocker):
        """JPEG/PNG/TIFF image keys must be discovered for OCR extraction."""
        contents = [
            _s3_object("docs/a.png", 10),
            _s3_object("docs/b.jpeg", 20),
            _s3_object("docs/c.tif", 30),
            _s3_object("docs/d.tiff", 40),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 4
        assert {d.key for d in result.documents} == {
            "docs/a.png",
            "docs/b.jpeg",
            "docs/c.tif",
            "docs/d.tiff",
        }

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
            prefixes=["docs/"],
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
            prefixes=["docs/"],
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
            _s3_object("data/notes.xlsx", 100),
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
        """The paginator must receive the correct bucket and prefix."""
        contents = [_s3_object("prefix/x.pdf", 10)]
        mock_client = _make_mock_s3_client(mocker, contents)

        discover_documents(
            bucket_name="my-bucket",
            prefixes=["prefix/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        mock_client.get_paginator.assert_called_with("list_objects_v2")
        mock_client.get_paginator.return_value.paginate.assert_called_once_with(Bucket="my-bucket", Prefix="prefix/")

    def test_listing_follows_pagination(self, mocker):
        """Listings longer than one page must not be truncated."""
        pages = [
            [_s3_object(f"docs/page1-{i}.pdf", 10) for i in range(1000)],
            [_s3_object("docs/page2-0.pdf", 10)],
        ]
        mock_client = _make_paginated_mock_s3_client(mocker, pages)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["docs/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 1001
        assert "docs/page2-0.pdf" in {d.key for d in result.documents}

    def test_audio_extensions_discovered(self, mocker):
        """Audio files with supported extensions must be discovered."""
        contents = [
            _s3_object("audio/meeting.wav", 1000),
            _s3_object("audio/podcast.mp3", 2000),
            _s3_object("audio/recording.m4a", 1500),
            _s3_object("audio/clip.aac", 800),
            _s3_object("audio/voice.ogg", 600),
            _s3_object("audio/sample.flac", 3000),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["audio/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 6
        keys = [d.key for d in result.documents]
        assert "audio/meeting.wav" in keys
        assert "audio/podcast.mp3" in keys
        assert "audio/recording.m4a" in keys
        assert "audio/clip.aac" in keys
        assert "audio/voice.ogg" in keys
        assert "audio/sample.flac" in keys

    def test_mixed_audio_and_document_extensions(self, mocker):
        """Audio and document files should both be discovered together."""
        contents = [
            _s3_object("data/report.pdf", 500),
            _s3_object("data/meeting.mp3", 2000),
            _s3_object("data/notes.md", 100),
            _s3_object("data/recording.wav", 3000),
            _s3_object("data/image.png", 400),
            _s3_object("data/archive.zip", 400),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["data/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.count == 5
        keys = [d.key for d in result.documents]
        assert "data/report.pdf" in keys
        assert "data/meeting.mp3" in keys
        assert "data/notes.md" in keys
        assert "data/recording.wav" in keys
        assert "data/image.png" in keys  # images are supported (OCR)
        assert "data/archive.zip" not in keys  # unsupported extension is filtered out


# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------


class TestDocumentIdentity:
    """The full object key is what names a document downstream."""

    def test_nested_keys_are_prioritised(self, mocker):
        """Benchmark data names documents by their full key, so sampling must match on it."""
        contents = [
            _s3_object("docs/other.pdf", 400),
            _s3_object("docs/manuals/xr-200/setup.pdf", 400),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["docs"],
            test_data_doc_names=["docs/manuals/xr-200/setup.pdf"],
            sampling_enabled=True,
            sampling_max_size_gb=400 / 1024**3,
            s3_client=mock_client,
        )

        assert [d.key for d in result.documents] == ["docs/manuals/xr-200/setup.pdf"]

    def test_bare_filename_benchmark_keys_still_prioritised(self, mocker):
        """A flat corpus may reference documents by file name alone."""
        contents = [
            _s3_object("docs/other.pdf", 400),
            _s3_object("docs/benchmark.pdf", 400),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["docs"],
            test_data_doc_names=["benchmark.pdf"],
            sampling_enabled=True,
            sampling_max_size_gb=400 / 1024**3,
            s3_client=mock_client,
        )

        assert [d.key for d in result.documents] == ["docs/benchmark.pdf"]

    def test_same_basename_in_different_folders_stays_distinct(self, mocker):
        """The collision this naming scheme exists to prevent."""
        contents = [
            _s3_object("docs/a/setup.txt", 100),
            _s3_object("docs/b/setup.txt", 100),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["docs"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        keys = [d.key for d in result.documents]
        assert keys == ["docs/a/setup.txt", "docs/b/setup.txt"]
        assert len(set(keys)) == 2


# ---------------------------------------------------------------------------
# Multi-location discovery
# ---------------------------------------------------------------------------


class TestMultiplePrefixes:
    """Documents from every selected location form one corpus."""

    CONTENTS = [
        _s3_object("manuals/setup.pdf", 100),
        _s3_object("manuals/nested/spec.pdf", 100),
        _s3_object("reports/q1.pdf", 100),
        _s3_object("archive/old.pdf", 100),
    ]

    def test_union_covers_every_prefix(self, mocker):
        """Every listed prefix contributes its documents."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["manuals/", "reports/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert [d.key for d in result.documents] == [
            "manuals/nested/spec.pdf",
            "manuals/setup.pdf",
            "reports/q1.pdf",
        ]
        assert result.prefixes == ("manuals/", "reports/")

    def test_overlapping_prefixes_are_deduplicated(self, mocker):
        """An object matched by two prefixes is kept once."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["manuals/", "manuals/nested/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        keys = [d.key for d in result.documents]
        assert keys == ["manuals/nested/spec.pdf", "manuals/setup.pdf"]
        assert result.count == 2

    def test_sampling_budget_is_shared_across_prefixes(self, mocker):
        """The size cap applies to the union, not to each location separately."""
        contents = [
            _s3_object("a/one.pdf", 100),
            _s3_object("b/two.pdf", 100),
            _s3_object("c/three.pdf", 100),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["a/", "b/", "c/"],
            sampling_enabled=True,
            sampling_max_size_gb=200 / 1024**3,
            s3_client=mock_client,
        )

        assert result.total_size_bytes == 200
        assert result.count == 2

    def test_benchmark_priority_spans_prefixes(self, mocker):
        """A benchmark document in the last location still wins the budget."""
        contents = [
            _s3_object("a/filler.pdf", 100),
            _s3_object("z/benchmark.pdf", 100),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["a/", "z/"],
            test_data_doc_names=["z/benchmark.pdf"],
            sampling_enabled=True,
            sampling_max_size_gb=100 / 1024**3,
            s3_client=mock_client,
        )

        assert [d.key for d in result.documents] == ["z/benchmark.pdf"]

    def test_bare_string_prefix_is_coerced(self, mocker):
        """A single string is accepted for backward compatibility."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        result = discover_documents(
            bucket_name="bucket",
            prefixes="reports/",
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.prefixes == ("reports/",)
        assert [d.key for d in result.documents] == ["reports/q1.pdf"]

    @pytest.mark.parametrize("prefixes", [None, [], [""], ["", "manuals/"]])
    def test_empty_prefix_lists_whole_bucket(self, mocker, prefixes):
        """No prefix -- or an empty one among others -- means the whole bucket."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=prefixes,
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.prefixes == ("",)
        assert result.count == len(self.CONTENTS)

    def test_prefixes_are_normalised(self, mocker):
        """Leading slashes are stripped and duplicates dropped, order preserved."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["/reports/", "manuals/", "reports/"],
            sampling_enabled=False,
            s3_client=mock_client,
        )

        assert result.prefixes == ("reports/", "manuals/")
        assert result.count == 3

    def test_no_documents_in_any_prefix_names_the_locations(self, mocker):
        """The error tells the user which locations were searched."""
        mock_client = _make_mock_s3_client(mocker, [_s3_object("other/x.pdf", 10)])

        with pytest.raises(RuntimeError, match=r"s3://bucket/a/, s3://bucket/b/"):
            discover_documents(
                bucket_name="bucket",
                prefixes=["a/", "b/"],
                sampling_enabled=False,
                s3_client=mock_client,
            )


# ---------------------------------------------------------------------------
# Benchmark key validation
# ---------------------------------------------------------------------------


class TestBenchmarkKeyValidation:
    """A benchmark key that matches no ingested object must fail loudly."""

    CONTENTS = [
        _s3_object("manuals/setup.pdf", 100),
        _s3_object("reports/setup.pdf", 100),
        _s3_object("reports/q1.pdf", 100),
    ]

    def test_unknown_key_raises(self, mocker):
        """A key absent from the corpus is reported by name."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        with pytest.raises(BenchmarkKeyError, match="'manuals/missing.pdf'"):
            discover_documents(
                bucket_name="bucket",
                prefixes=["manuals/", "reports/"],
                test_data_doc_names=["manuals/setup.pdf", "manuals/missing.pdf"],
                sampling_enabled=False,
                s3_client=mock_client,
            )

    def test_ambiguous_basename_raises(self, mocker):
        """A file name shared by two locations cannot identify one document."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        with pytest.raises(BenchmarkKeyError, match="shared by 2 objects"):
            discover_documents(
                bucket_name="bucket",
                prefixes=["manuals/", "reports/"],
                test_data_doc_names=["setup.pdf"],
                sampling_enabled=False,
                s3_client=mock_client,
            )

    def test_exact_key_wins_over_basename(self, mocker):
        """A full object key is unambiguous even when the file name collides."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["manuals/", "reports/"],
            test_data_doc_names=["reports/setup.pdf"],
            sampling_enabled=True,
            sampling_max_size_gb=100 / 1024**3,
            s3_client=mock_client,
        )

        assert [d.key for d in result.documents] == ["reports/setup.pdf"]

    def test_error_message_explains_the_key_format(self, mocker):
        """The message has to be actionable for someone editing the benchmark JSON."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        with pytest.raises(BenchmarkKeyError) as exc_info:
            discover_documents(
                bucket_name="bucket",
                prefixes=["reports/"],
                test_data_doc_names=["q1.pdf", "nope.pdf"],
                sampling_enabled=False,
                s3_client=mock_client,
            )

        message = str(exc_info.value)
        assert "correct_answer_document_keys" in message
        assert "full object key" in message
        assert "s3://bucket/reports/" in message

    def test_validation_can_be_disabled(self, mocker, caplog):
        """With validation off an unmatched key only warns."""
        mock_client = _make_mock_s3_client(mocker, self.CONTENTS)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["reports/"],
            test_data_doc_names=["nope.pdf"],
            sampling_enabled=False,
            validate_test_data_keys=False,
            s3_client=mock_client,
        )

        assert result.count == 2
        assert "not part of the discovered corpus" in caplog.text

    def test_benchmark_document_dropped_by_budget_warns(self, mocker, caplog):
        """Benchmark docs that alone exceed the budget are flagged, not swallowed."""
        contents = [
            _s3_object("reports/small.pdf", 100),
            _s3_object("reports/huge.pdf", 1000),
        ]
        mock_client = _make_mock_s3_client(mocker, contents)

        result = discover_documents(
            bucket_name="bucket",
            prefixes=["reports/"],
            test_data_doc_names=["reports/small.pdf", "reports/huge.pdf"],
            sampling_enabled=True,
            sampling_max_size_gb=500 / 1024**3,
            s3_client=mock_client,
        )

        assert [d.key for d in result.documents] == ["reports/small.pdf"]
        assert "reports/huge.pdf" in caplog.text
        assert "sampling budget" in caplog.text


# ---------------------------------------------------------------------------
# S3 client creation
# ---------------------------------------------------------------------------


class TestS3ClientCreation:
    """A client is built on demand, retrying without TLS verification when needed."""

    CONTENTS = [_s3_object("docs/report.pdf", 100)]

    def test_client_is_created_when_none_is_supplied(self, mocker):
        """Without an explicit client, discovery builds a verified one."""
        client = _make_mock_s3_client(mocker, self.CONTENTS)
        create = mocker.patch(
            "ai4rag.utils.data.documents_discovery.create_s3_client",
            return_value=client,
        )

        result = discover_documents(bucket_name="bucket", prefixes=["docs/"], sampling_enabled=False)

        create.assert_called_once_with()
        client.list_objects_v2.assert_called_once_with(Bucket="bucket", Prefix="docs/", MaxKeys=1)
        assert result.count == 1

    def test_ssl_error_retries_without_verification(self, mocker):
        """A self-signed endpoint falls back to an unverified client."""
        from botocore.exceptions import SSLError

        failing = mocker.MagicMock()
        failing.list_objects_v2.side_effect = SSLError(endpoint_url="https://s3.example", error="self-signed")
        working = _make_mock_s3_client(mocker, self.CONTENTS)
        create = mocker.patch(
            "ai4rag.utils.data.documents_discovery.create_s3_client",
            side_effect=[failing, working],
        )

        result = discover_documents(bucket_name="bucket", prefixes=["docs/"], sampling_enabled=False)

        assert create.call_args_list == [mocker.call(), mocker.call(verify=False)]
        assert result.count == 1
