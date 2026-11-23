# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Public extraction contracts and bounded staging across batch boundaries."""

import errno
import os
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from torch import OutOfMemoryError

from ai4rag.utils.clients import s3
from ai4rag.utils.data import _extraction_pool
from ai4rag.utils.data import text_extraction as extraction


class _Client:
    def __init__(self, failures=()):
        self.failures = failures
        self.paths = []
        self.transfer_configs = []
        self.peak_inputs = 0

    def download_file(self, _bucket, key, filename, **kwargs):
        assert kwargs["Config"].use_threads is False
        self.transfer_configs.append(kwargs["Config"])
        self.paths.append(Path(filename))
        self.peak_inputs = max(self.peak_inputs, sum(path.exists() for path in self.paths))
        if key in self.failures:
            Path(filename).write_text("partial")
            raise RuntimeError("download failed")
        Path(filename).write_text("content", encoding="utf-8")


@pytest.fixture
def inline_pool(monkeypatch):
    """Exercise the scheduler and real TXT conversion without interpreter startup."""

    class Pool:
        instances = []

        def __init__(self, **kwargs):
            self.settings = kwargs
            self.batches = []
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def convert(self, documents, output_dir):
            self.batches.append(documents)
            for path, key in documents:
                success, error = extraction._worker_process_document(str(path), str(output_dir), key)
                yield path, success, error

    monkeypatch.setattr(_extraction_pool, "ConversionPool", Pool)
    return Pool


def test_batches_bound_inputs_and_preserve_names(inline_pool, tmp_path):
    client = _Client()
    keys = [f"docs/folder{i}/same.txt" for i in range(9)]
    result = extraction.extract_text(
        [{"key": key, "size_bytes": 10} for key in keys],
        "bucket",
        tmp_path,
        s3_client=client,
        max_extraction_workers=2,
    )
    assert result == extraction.ExtractionResult(9, 9, 0)
    assert [len(batch) for batch in inline_pool.instances[0].batches] == [4, 4, 1]
    assert len({id(config) for config in client.transfer_configs}) == 1
    assert client.peak_inputs <= 4
    assert all(not path.exists() for path in client.paths)
    assert {DoclingDocument.load_from_json(path).name for path in tmp_path.rglob("*.json")} == set(keys)
    assert not list(tmp_path.rglob("*.tmp"))


def test_tolerance_counts_failures_across_batches(inline_pool, tmp_path):
    keys = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
    client = _Client(failures={"a.txt", "e.txt"})
    result = extraction.extract_text(
        [{"key": key} for key in keys],
        "bucket",
        tmp_path,
        s3_client=client,
        max_extraction_workers=1,
        error_tolerance=0.4,
    )
    assert result == extraction.ExtractionResult(3, 5, 2)
    assert len(list(tmp_path.rglob("*.json"))) == 3
    assert all(not path.exists() for path in client.paths)


def test_conversion_failure_within_tolerance(inline_pool, tmp_path):
    result = extraction.extract_text(
        [{"key": "a.pdf"}, {"key": "b.txt"}],
        "bucket",
        tmp_path,
        s3_client=_Client(),
        error_tolerance=0.5,
        max_extraction_workers=1,
    )
    assert result == extraction.ExtractionResult(1, 2, 1)
    assert not (tmp_path / "a.pdf.json").exists()


def test_exceeded_tolerance_cleans_downloads(inline_pool, tmp_path):
    client = _Client(failures={"bad.txt"})
    with pytest.raises(RuntimeError, match="Text extraction failed"):
        extraction.extract_text([{"key": "bad.txt"}], "bucket", tmp_path, s3_client=client)
    assert all(not path.exists() for path in client.paths)


def test_small_corpus_caps_explicit_workers(inline_pool, tmp_path):
    extraction.extract_text([{"key": "one.txt"}], "bucket", tmp_path, s3_client=_Client(), max_extraction_workers=48)
    assert inline_pool.instances[0].settings["workers"] == 1


def test_no_workers_for_unsupported_only_corpus(inline_pool, tmp_path):
    result = extraction.extract_text([{"key": "skip.xyz"}], "bucket", tmp_path, s3_client=_Client())
    assert result == extraction.ExtractionResult(0, 1, 0)
    assert not inline_pool.instances


def test_empty_input_needs_no_credentials_or_workers(inline_pool, tmp_path):
    assert extraction.extract_text([], "bucket", tmp_path) == extraction.ExtractionResult(0, 0, 0)
    assert not inline_pool.instances


@pytest.mark.parametrize("name", ["document_timeout", "worker_startup_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_timeouts_are_positive_and_finite(inline_pool, tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        extraction.extract_text([{"key": "one.txt"}], "bucket", tmp_path, s3_client=_Client(), **{name: value})
    assert not inline_pool.instances


def test_atomic_json_failure_preserves_existing_output(tmp_path):
    output = tmp_path / "document.json"
    output.write_text("original")

    class BrokenDocument:
        def save_as_json(self, path):
            path.write_text("partial")
            raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        extraction._save_document(BrokenDocument(), output)
    assert output.read_text() == "original"
    assert list(tmp_path.iterdir()) == [output]


def test_worker_failure_is_fatal_and_cleans_staging(inline_pool, monkeypatch, tmp_path):
    """Infrastructure failures remain fatal at full document-error tolerance."""

    def broken_conversion(_self, _documents, output_dir):
        (output_dir / "unfinished.json").write_text("partial")
        raise RuntimeError("worker stopped")

    monkeypatch.setattr(inline_pool, "convert", broken_conversion)
    client = _Client()
    with pytest.raises(RuntimeError, match="worker stopped"):
        extraction.extract_text(
            [{"key": "file.txt"}],
            "bucket",
            tmp_path,
            s3_client=client,
            error_tolerance=1.0,
        )
    assert not list(tmp_path.iterdir())
    assert all(not path.exists() for path in client.paths)


def test_download_supports_older_classic_only_boto3(monkeypatch, tmp_path):
    """The declared boto3 minimum predates the CRT backend selector."""

    class ClassicConfig:
        def __init__(self, use_threads):
            self.use_threads = use_threads

    monkeypatch.setattr(s3, "TransferConfig", ClassicConfig)
    client = _Client()
    path = extraction._download_document(
        {"key": "file.txt"}, "bucket", tmp_path, client, extraction._download_transfer_config()
    )
    assert path.read_text() == "content"


def test_cpu_quota_and_affinity_bound_automatic_workers(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 96)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(12)), raising=False)
    monkeypatch.setattr(extraction, "_container_cpu_quota", lambda: 3.5)
    assert extraction._effective_worker_count(None) == 1
    assert extraction._effective_worker_count(48) == 48


@pytest.mark.parametrize("v2", [True, False])
def test_container_cpu_quota_reads_process_and_ancestor_limits(monkeypatch, v2):
    membership = "0::/parent/child" if v2 else "1:cpu,cpuacct:/parent/child"
    files = {"/proc/self/cgroup": f"malformed\n{membership}\nmalformed:line"}
    if v2:
        files.update(
            {
                "/sys/fs/cgroup/cpu.max": "max 100000",
                "/sys/fs/cgroup/parent/cpu.max": "400000 100000",
                "/sys/fs/cgroup/parent/child/cpu.max": "max 100000",
            }
        )
    else:
        files.update(
            {
                "/sys/fs/cgroup/cpu/parent/cpu.cfs_quota_us": "400000",
                "/sys/fs/cgroup/cpu/parent/cpu.cfs_period_us": "100000",
                "/sys/fs/cgroup/cpu/parent/child/cpu.cfs_quota_us": "200000",
                "/sys/fs/cgroup/cpu/parent/child/cpu.cfs_period_us": "100000",
            }
        )

    def read(path, **_kwargs):
        if str(path) not in files:
            raise FileNotFoundError(path)
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    assert _extraction_pool._container_cpu_quota() == (4 if v2 else 2)


def test_real_worker_extracts_txt_without_models_or_main_guard(tmp_path):
    """The installed module entry point works without pickling the caller or client."""
    result = extraction.extract_text(
        [{"key": "/nested/file.txt"}],
        "bucket",
        tmp_path,
        s3_client=_Client(),
        worker_startup_timeout=60,
        document_timeout=30,
    )
    assert result == extraction.ExtractionResult(1, 1, 0)
    assert DoclingDocument.load_from_json(tmp_path / "nested/file.txt.json").name == "/nested/file.txt"


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("out of memory"),
        OutOfMemoryError("out of memory"),
        OSError(errno.ENOMEM, "Cannot allocate memory"),
        RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 1024 bytes"),
    ],
)
def test_conversion_memory_errors_reach_the_worker_supervisor(monkeypatch, tmp_path, error):
    """Memory exhaustion must not become an ordinary tolerated document error."""
    source = tmp_path / "file.txt"
    source.write_text("content")

    def fail(*_args):
        raise error

    monkeypatch.setattr(extraction, "_save_document", fail)
    with pytest.raises(MemoryError):
        extraction._worker_process_document(str(source), str(tmp_path / "output"), "file.txt")


@pytest.mark.parametrize("error", [OSError(errno.ENOSPC, "No space left on device"), RuntimeError("invalid input")])
def test_other_conversion_errors_remain_document_failures(monkeypatch, tmp_path, error):
    source = tmp_path / "file.txt"
    source.write_text("content")

    def fail(*_args):
        raise error

    monkeypatch.setattr(extraction, "_save_document", fail)
    success, traceback = extraction._worker_process_document(str(source), str(tmp_path / "output"), "file.txt")
    assert not success
    assert str(error) in traceback


@pytest.mark.parametrize("explicit_cause", [True, False])
def test_wrapped_docling_memory_errors_reach_the_supervisor(monkeypatch, tmp_path, explicit_cause):
    """Docling wraps model errors in RuntimeError while retaining the cause."""

    class Converter:
        def convert(self, _path):
            try:
                raise OutOfMemoryError("model allocation failed")
            except OutOfMemoryError as exc:
                if explicit_cause:
                    raise RuntimeError("Pipeline failed") from exc
                raise RuntimeError("Pipeline failed")

    monkeypatch.setattr(extraction, "_mp_worker_converter", Converter())
    source = tmp_path / "file.pdf"
    source.write_bytes(b"placeholder")
    with pytest.raises(MemoryError):
        extraction._worker_process_document(str(source), str(tmp_path / "output"), "file.pdf")


def test_real_worker_memory_recovery_preserves_outputs_and_downloads(monkeypatch, tmp_path, capfd):
    """Recover through the production entry point, then reuse it for another batch."""
    import sys

    marker = tmp_path / "failed-once"
    bootstrap = """
import runpy, sys
from pathlib import Path
from ai4rag.utils.data import text_extraction
original = text_extraction._worker_process_document
marker = Path(sys.argv[2])
def convert(path, output_dir, key):
    if key == 'bad.txt' and not marker.exists():
        marker.touch()
        # Leave staged output behind to verify it never leaks into the artifact.
        (Path(output_dir) / 'unfinished.tmp').write_text('partial')
        raise MemoryError('simulated conversion memory exhaustion')
    return original(path, output_dir, key)
text_extraction._worker_process_document = convert
sys.argv = [sys.argv[0], sys.argv[1]]
runpy.run_module('ai4rag.utils.data._extraction_worker', run_name='__main__')
"""
    command = lambda fd: [sys.executable, "-c", bootstrap, str(fd), str(marker)]
    monkeypatch.setattr(_extraction_pool, "_worker_command", command)
    client = _Client()
    output = tmp_path / "output"
    result = extraction.extract_text(
        [{"key": key} for key in ("done.txt", "bad.txt", "next-batch.txt")],
        "bucket",
        output,
        s3_client=client,
        max_extraction_workers=1,
        worker_startup_timeout=60,
        document_timeout=30,
    )
    assert result == extraction.ExtractionResult(3, 3, 0)
    assert "Text extraction worker caught MemoryError." in capfd.readouterr().err
    assert marker.exists()
    assert len(client.paths) == 3
    assert all(not path.exists() for path in client.paths)
    assert {path.name for path in output.iterdir()} == {"done.txt.json", "bad.txt.json", "next-batch.txt.json"}
    assert {DoclingDocument.load_from_json(path).name for path in output.iterdir()} == {
        "done.txt",
        "bad.txt",
        "next-batch.txt",
    }
