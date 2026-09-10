# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""End-to-end checks that a document's key is the identity carried through ingestion.

The unit suites cover each hop in isolation (prefix stripping in
``test_discovery``, path/key pairing in ``test_extraction``).  This module wires
the real components together -- discovery, download, extraction, and benchmark
validation -- against an in-memory bucket, and asserts the property those hops
exist to guarantee: the key a document is discovered under is the name it is
extracted as, and therefore the name benchmark data must reference.

Object storage is faked at the smallest possible seam (the boto3 client), so
``discover_documents``, ``_download_document`` and ``_worker_process_document``
all run for real, including key normalisation and path-traversal checks.  Only
``.txt`` documents are used, which the worker handles without a Docling
``DocumentConverter`` -- that keeps the suite free of model downloads.
"""

from pathlib import Path

import pandas as pd
import pytest
from docling_core.types.doc.document import DoclingDocument

from ai4rag.core.experiment.benchmark_data import BenchmarkData, BenchmarkDataValueError
from ai4rag.core.experiment.results import EvaluationResult, ExperimentResults
from ai4rag.evaluator.base_evaluator import EvaluationData
from ai4rag.utils.data import text_extraction
from ai4rag.utils.data.documents_discovery import discover_documents

# A corpus built around the collision this naming scheme exists to prevent:
# two documents sharing a basename under different folders.
PREFIX = "datasets/rag/docs"
CORPUS = {
    f"{PREFIX}/manuals/xr-200/setup.txt": "Set up the XR-200.",
    f"{PREFIX}/manuals/xr-300/setup.txt": "Set up the XR-300.",
    f"{PREFIX}/overview.txt": "Product overview.",
}


# ---------------------------------------------------------------------------
# In-memory object storage
# ---------------------------------------------------------------------------


class _FakeS3Client:
    """Minimal stand-in for the boto3 S3 client used by discovery and download."""

    def __init__(self, objects: dict[str, str]):
        self._objects = objects

    def list_objects_v2(self, Bucket: str, Prefix: str = ""):  # noqa: N803 - boto3 casing
        del Bucket
        return {"Contents": [{"Key": k, "Size": len(v)} for k, v in self._objects.items() if k.startswith(Prefix)]}

    def download_file(self, Bucket: str, Key: str, Filename: str):  # noqa: N803 - boto3 casing
        del Bucket
        Path(Filename).write_text(self._objects[Key], encoding="utf-8")


class _InlinePool:
    """Process-pool stand-in that runs the worker synchronously in-process.

    ``extract_text`` uses a ``spawn`` pool whose initializer builds a Docling
    ``DocumentConverter``; running the worker inline exercises the same function
    with the same arguments without paying for model loading.
    """

    def apply_async(self, func, args):
        result = func(*args)
        return type("_Result", (), {"ready": staticmethod(lambda: True), "get": staticmethod(lambda: result)})()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_bucket(monkeypatch):
    """Serve ``CORPUS`` to both discovery and download."""
    client = _FakeS3Client(CORPUS)
    monkeypatch.setattr(text_extraction, "_make_s3_client", lambda *_args, **_kwargs: client)
    return client


@pytest.fixture
def extracted(fake_bucket, tmp_path) -> Path:
    """Run the real discovery -> download -> extraction chain, return the output dir.

    This is the ingestion path a pipeline run takes: ``discover_documents``
    produces the descriptor, whose ``documents`` list is handed to extraction
    exactly as ``pipelines-components`` hands it over.
    """
    return _ingest(discover_documents(bucket_name="bucket", prefix=PREFIX, s3_client=fake_bucket), tmp_path)


def _ingest(discovery, tmp_path: Path) -> Path:
    """Download and extract the documents of a discovery result."""
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    out_dir = tmp_path / "extracted"

    tasks, errors = text_extraction._download_and_submit(
        docs=discovery.to_dict()["documents"],
        bucket="bucket",
        download_path=download_dir,
        process_pool=_InlinePool(),
        out_dir=out_dir,
        s3_creds={},
    )

    assert not errors, f"Downloads failed: {errors}"
    for file_path, task in tasks:
        success, error = task.get()
        assert success, f"Extraction failed for {file_path}: {error}"

    return out_dir


def _extracted_names(out_dir: Path) -> set[str]:
    """Names of every ``DoclingDocument`` written under *out_dir*."""
    return {DoclingDocument.load_from_json(str(p)).name for p in out_dir.rglob("*.json")}


# ---------------------------------------------------------------------------
# Key preservation
# ---------------------------------------------------------------------------


class TestObjectKeyPreservation:
    """A document's prefix-relative key must survive ingestion as its name."""

    def test_relative_keys_survive_discovery_to_extraction(self, extracted):
        """Documents are named by their position inside the discovery prefix."""
        assert _extracted_names(extracted) == {
            "manuals/xr-200/setup.txt",
            "manuals/xr-300/setup.txt",
            "overview.txt",
        }

    def test_output_layout_mirrors_the_keys(self, extracted):
        """The on-disk layout mirrors the keys, so nothing overwrites anything."""
        written = sorted(str(p.relative_to(extracted)) for p in extracted.rglob("*.json"))
        assert written == [
            "manuals/xr-200/setup.txt.json",
            "manuals/xr-300/setup.txt.json",
            "overview.txt.json",
        ]

    def test_same_basename_documents_keep_their_own_content(self, extracted):
        """The collision regression: two ``setup.txt`` files must not merge.

        Naming by basename alone gave both documents the name ``setup.txt`` and
        the same output path, so one silently overwrote the other.
        """
        xr200 = DoclingDocument.load_from_json(str(extracted / "manuals/xr-200/setup.txt.json"))
        xr300 = DoclingDocument.load_from_json(str(extracted / "manuals/xr-300/setup.txt.json"))

        assert xr200.name != xr300.name
        assert xr200.texts[0].text == "Set up the XR-200."
        assert xr300.texts[0].text == "Set up the XR-300."

    def test_full_keys_are_preserved_when_discovery_has_no_prefix(self, fake_bucket, tmp_path):
        """Without a prefix to strip, the full object key is the document name."""
        out_dir = _ingest(discover_documents(bucket_name="bucket", s3_client=fake_bucket), tmp_path)

        assert _extracted_names(out_dir) == set(CORPUS)

    def test_leading_slash_key_still_names_the_document(self, monkeypatch, tmp_path):
        """A key needing normalisation must not fall back to the bare filename.

        ``_download_document`` strips the leading slash before building the local
        path, so the key has to travel alongside the path rather than being
        recovered from it.
        """
        client = _FakeS3Client({"/docs/a/setup.txt": "content"})
        monkeypatch.setattr(text_extraction, "_make_s3_client", lambda *_a, **_k: client)

        download_dir = tmp_path / "download"
        download_dir.mkdir()
        out_dir = tmp_path / "extracted"

        tasks, errors = text_extraction._download_and_submit(
            docs=[{"key": "/docs/a/setup.txt", "size_bytes": 7, "relative_key": "a/setup.txt"}],
            bucket="bucket",
            download_path=download_dir,
            process_pool=_InlinePool(),
            out_dir=out_dir,
            s3_creds={},
        )

        assert not errors
        assert all(task.get()[0] for _, task in tasks)
        assert _extracted_names(out_dir) == {"a/setup.txt"}


# ---------------------------------------------------------------------------
# The contract with benchmark data
# ---------------------------------------------------------------------------


class TestBenchmarkKeysMatchExtractedDocuments:
    """Benchmark keys are matched against extracted document names, so they must agree."""

    @staticmethod
    def _benchmark(document_keys: list[str]) -> BenchmarkData:
        return BenchmarkData(
            benchmark_data=pd.DataFrame(
                [
                    {
                        "question": "How do I set up the XR-300?",
                        "correct_answers": ["Follow the manual."],
                        "correct_answer_document_keys": document_keys,
                    }
                ]
            )
        )

    def test_relative_keys_select_the_intended_document(self, extracted):
        """A benchmark written against relative keys resolves to exactly one document."""
        benchmark = self._benchmark(["manuals/xr-300/setup.txt"])
        referenced = {key for keys in benchmark.document_keys for key in keys}

        assert referenced & _extracted_names(extracted) == {"manuals/xr-300/setup.txt"}

    def test_full_object_keys_match_nothing(self, extracted):
        """The misconfiguration to recognise: keys that still carry the prefix.

        Nothing raises here -- the documents simply never match, which is why
        ``ModelsPreSelector`` reports unmatched keys rather than scoring zero.
        """
        benchmark = self._benchmark([f"{PREFIX}/manuals/xr-300/setup.txt"])
        referenced = {key for keys in benchmark.document_keys for key in keys}

        assert not referenced & _extracted_names(extracted)

    def test_bare_filenames_are_ambiguous_across_folders(self, extracted):
        """A basename cannot address a document once folders are in play."""
        benchmark = self._benchmark(["setup.txt"])
        referenced = {key for keys in benchmark.document_keys for key in keys}

        assert not referenced & _extracted_names(extracted)


# ---------------------------------------------------------------------------
# Field naming
# ---------------------------------------------------------------------------


class TestDocumentKeyFieldNaming:
    """The rename from ``document_ids`` to ``document_keys`` must hold at both ends."""

    def test_old_benchmark_field_name_is_rejected(self):
        """``correct_answer_document_ids`` no longer satisfies the schema."""
        df = pd.DataFrame(
            [
                {
                    "question": "What is Python?",
                    "correct_answers": ["A programming language"],
                    "correct_answer_document_ids": ["guides/python.md"],
                }
            ]
        )

        with pytest.raises(BenchmarkDataValueError, match="correct_answer_document_keys"):
            BenchmarkData(benchmark_data=df)

    def test_new_benchmark_field_name_works(self):
        """``correct_answer_document_keys`` is read into ``document_keys``."""
        df = pd.DataFrame(
            [
                {
                    "question": "What is Python?",
                    "correct_answers": ["A programming language"],
                    "correct_answer_document_keys": ["guides/python.md"],
                }
            ]
        )

        assert BenchmarkData(benchmark_data=df).document_keys[0] == ["guides/python.md"]

    def test_evaluation_results_report_document_key(self):
        """Results carry the retrieved document's key under ``document_key``.

        This is the field a user reads to see *which* document backed an answer,
        so it has to hold the same key their benchmark data references.
        """
        eval_data = EvaluationData(
            question="How do I set up the XR-300?",
            question_id="q0",
            answer="Follow the manual.",
            contexts=["Set up the XR-300."],
            context_ids=["manuals/xr-300/setup.txt"],
            ground_truths=["Follow the manual."],
        )
        evaluation_result = EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params={},
            rag_params={},
            scores={
                "metrics": [],
                "question_scores": [
                    {"question_id": "q0", "metrics": [{"name": "faithfulness", "evaluator": "ragas", "value": 0.9}]}
                ],
            },
            execution_time=1.0,
            final_score=0.9,
        )

        results = ExperimentResults.create_evaluation_results_json([eval_data], evaluation_result)

        assert results[0]["answer_contexts"] == [
            {"text": "Set up the XR-300.", "document_key": "manuals/xr-300/setup.txt"}
        ]
