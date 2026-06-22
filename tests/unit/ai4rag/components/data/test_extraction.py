# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai4rag.components.data.text_extraction import (
    ExtractionResult,
    _effective_worker_count,
    _raise_if_threshold_exceeded,
    _resolve_artifacts_path,
    _resolve_s3_credentials,
)

# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


class TestExtractionResult:
    """Tests for the ``ExtractionResult`` frozen dataclass."""

    def test_attributes(self):
        """Verify field values are stored correctly."""
        result = ExtractionResult(processed_count=8, total_documents=10, error_count=2)
        assert result.processed_count == 8
        assert result.total_documents == 10
        assert result.error_count == 2

    def test_frozen(self):
        """Frozen dataclass must reject attribute mutation."""
        result = ExtractionResult(processed_count=1, total_documents=1, error_count=0)
        with pytest.raises(AttributeError):
            result.processed_count = 99


# ---------------------------------------------------------------------------
# _resolve_s3_credentials
# ---------------------------------------------------------------------------


class TestResolveS3Credentials:
    """Tests for ``_resolve_s3_credentials``."""

    def test_all_explicit_values(self):
        """Explicit arguments should populate the credentials dict."""
        creds = _resolve_s3_credentials(
            endpoint="https://s3.example.com",
            access_key="AKIA_FAKE",
            secret_key="secret123",
            region="us-east-1",
        )
        assert creds["AWS_S3_ENDPOINT"] == "https://s3.example.com"
        assert creds["AWS_ACCESS_KEY_ID"] == "AKIA_FAKE"
        assert creds["AWS_SECRET_ACCESS_KEY"] == "secret123"
        assert creds["AWS_DEFAULT_REGION"] == "us-east-1"

    def test_env_var_fallback(self, monkeypatch):
        """Missing explicit arguments should fall back to environment variables."""
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://env-endpoint.com")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

        creds = _resolve_s3_credentials(
            endpoint=None,
            access_key=None,
            secret_key=None,
            region=None,
        )

        assert creds["AWS_S3_ENDPOINT"] == "https://env-endpoint.com"
        assert creds["AWS_ACCESS_KEY_ID"] == "ENV_KEY"
        assert creds["AWS_SECRET_ACCESS_KEY"] == "ENV_SECRET"
        assert creds["AWS_DEFAULT_REGION"] == "eu-west-1"

    def test_explicit_takes_precedence_over_env(self, monkeypatch):
        """Explicit arguments must take precedence over environment variables."""
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://env-endpoint.com")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENV_KEY")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ENV_SECRET")

        creds = _resolve_s3_credentials(
            endpoint="https://explicit.com",
            access_key="EXPLICIT_KEY",
            secret_key="EXPLICIT_SECRET",
            region=None,
        )

        assert creds["AWS_S3_ENDPOINT"] == "https://explicit.com"
        assert creds["AWS_ACCESS_KEY_ID"] == "EXPLICIT_KEY"
        assert creds["AWS_SECRET_ACCESS_KEY"] == "EXPLICIT_SECRET"

    def test_missing_endpoint_raises(self, monkeypatch):
        """Missing endpoint (neither explicit nor env) must raise ``ValueError``."""
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        with pytest.raises(ValueError, match="AWS_S3_ENDPOINT"):
            _resolve_s3_credentials(endpoint=None, access_key="k", secret_key="s", region=None)

    def test_missing_access_key_raises(self, monkeypatch):
        """Missing access key must raise ``ValueError``."""
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            _resolve_s3_credentials(endpoint="https://ep.com", access_key=None, secret_key="s", region=None)

    def test_missing_secret_key_raises(self, monkeypatch):
        """Missing secret key must raise ``ValueError``."""
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY"):
            _resolve_s3_credentials(endpoint="https://ep.com", access_key="k", secret_key=None, region=None)

    def test_missing_all_required_raises(self, monkeypatch):
        """Missing all required credentials must raise ``ValueError``."""
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        with pytest.raises(ValueError, match="Missing S3 credential"):
            _resolve_s3_credentials(endpoint=None, access_key=None, secret_key=None, region=None)

    def test_region_is_optional(self):
        """Region may be ``None`` without causing an error."""
        creds = _resolve_s3_credentials(
            endpoint="https://ep.com",
            access_key="k",
            secret_key="s",
            region=None,
        )
        assert creds["AWS_DEFAULT_REGION"] is None


# ---------------------------------------------------------------------------
# _resolve_artifacts_path
# ---------------------------------------------------------------------------


class TestResolveArtifactsPath:
    """Tests for ``_resolve_artifacts_path``."""

    def test_explicit_existing_dir_with_files(self, tmp_path):
        """An explicit path to a non-empty directory should be returned."""
        (tmp_path / "model.bin").write_bytes(b"data")

        result = _resolve_artifacts_path(str(tmp_path))

        assert result == tmp_path

    def test_explicit_empty_dir_returns_none(self, tmp_path):
        """An empty directory should return ``None`` (fallback to HF download)."""
        result = _resolve_artifacts_path(str(tmp_path))

        assert result is None

    def test_explicit_nonexistent_dir_returns_none(self):
        """A path that does not exist should return ``None``."""
        result = _resolve_artifacts_path("/nonexistent/path/to/artifacts")

        assert result is None

    def test_env_var_fallback(self, tmp_path, monkeypatch):
        """When explicit is ``None``, the env var ``DOCLING_ARTIFACTS_PATH`` should be used."""
        (tmp_path / "weights.pt").write_bytes(b"data")
        monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path))

        result = _resolve_artifacts_path(None)

        assert result == tmp_path

    def test_env_var_empty_dir_returns_none(self, tmp_path, monkeypatch):
        """Env var pointing to an empty directory should return ``None``."""
        monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path))

        result = _resolve_artifacts_path(None)

        assert result is None

    def test_no_explicit_no_env_returns_none(self, monkeypatch):
        """No explicit path and no env var should return ``None``."""
        monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)

        result = _resolve_artifacts_path(None)

        assert result is None

    def test_explicit_overrides_env(self, tmp_path, monkeypatch):
        """Explicit path must take precedence over the env var."""
        env_dir = tmp_path / "env_artifacts"
        env_dir.mkdir()
        (env_dir / "model.bin").write_bytes(b"env")
        monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(env_dir))

        explicit_dir = tmp_path / "explicit_artifacts"
        explicit_dir.mkdir()
        (explicit_dir / "model.bin").write_bytes(b"explicit")

        result = _resolve_artifacts_path(str(explicit_dir))

        assert result == explicit_dir


# ---------------------------------------------------------------------------
# _effective_worker_count
# ---------------------------------------------------------------------------


class TestEffectiveWorkerCount:
    """Tests for ``_effective_worker_count``."""

    def test_explicit_positive_value(self):
        """An explicit positive integer should be returned as-is."""
        assert _effective_worker_count(4) == 4

    def test_explicit_one(self):
        """Explicit value of 1 should be returned."""
        assert _effective_worker_count(1) == 1

    def test_explicit_zero_clamped_to_one(self):
        """Explicit value of 0 should be clamped to 1."""
        assert _effective_worker_count(0) == 1

    def test_explicit_negative_clamped_to_one(self):
        """Negative values should be clamped to 1."""
        assert _effective_worker_count(-5) == 1

    def test_none_uses_cpu_count(self, monkeypatch):
        """``None`` should compute workers from ``os.cpu_count()``."""
        monkeypatch.setattr(os, "cpu_count", lambda: 16)

        result = _effective_worker_count(None)

        # min(max(1, 16 // 2), 8) = min(8, 8) = 8
        assert result == 8

    def test_none_with_low_cpu_count(self, monkeypatch):
        """Low CPU counts should still yield at least 1 worker."""
        monkeypatch.setattr(os, "cpu_count", lambda: 1)

        result = _effective_worker_count(None)

        # min(max(1, 1 // 2), 8) = min(max(1, 0), 8) = min(1, 8) = 1
        assert result == 1

    def test_none_with_none_cpu_count(self, monkeypatch):
        """``os.cpu_count()`` returning ``None`` should yield 1 worker."""
        monkeypatch.setattr(os, "cpu_count", lambda: None)

        result = _effective_worker_count(None)

        # min(max(1, 1 // 2), 8) = 1
        assert result == 1

    def test_none_with_many_cpus_capped_at_eight(self, monkeypatch):
        """Worker count is capped at 8 regardless of CPU count."""
        monkeypatch.setattr(os, "cpu_count", lambda: 64)

        result = _effective_worker_count(None)

        assert result == 8


# ---------------------------------------------------------------------------
# _raise_if_threshold_exceeded
# ---------------------------------------------------------------------------


class TestRaiseIfThresholdExceeded:
    """Tests for ``_raise_if_threshold_exceeded``."""

    def test_no_errors_does_not_raise(self):
        """Zero errors should never raise, regardless of tolerance."""
        _raise_if_threshold_exceeded(error_details=[], total_docs=100, tolerance=None)
        _raise_if_threshold_exceeded(error_details=[], total_docs=100, tolerance=0.0)

    def test_errors_within_tolerance(self):
        """Errors at or below the allowed count should not raise."""
        errors = [{"file": "a.pdf", "traceback": "err"}]
        # tolerance=0.1 with 10 docs -> 1 allowed
        _raise_if_threshold_exceeded(error_details=errors, total_docs=10, tolerance=0.1)

    def test_errors_at_exact_tolerance_boundary(self):
        """Errors exactly at the tolerance boundary should not raise."""
        errors = [
            {"file": "a.pdf", "traceback": "err"},
            {"file": "b.pdf", "traceback": "err"},
        ]
        # tolerance=0.2 with 10 docs -> 2 allowed
        _raise_if_threshold_exceeded(error_details=errors, total_docs=10, tolerance=0.2)

    def test_errors_exceed_tolerance_raises(self):
        """Errors exceeding the tolerance must raise ``RuntimeError``."""
        errors = [
            {"file": "a.pdf", "traceback": "err a"},
            {"file": "b.pdf", "traceback": "err b"},
        ]
        # tolerance=0.1 with 10 docs -> only 1 allowed, but we have 2
        with pytest.raises(RuntimeError, match="Text extraction failed"):
            _raise_if_threshold_exceeded(error_details=errors, total_docs=10, tolerance=0.1)

    def test_none_tolerance_means_zero_allowed(self):
        """``tolerance=None`` means zero errors are tolerated."""
        errors = [{"file": "a.pdf", "traceback": "err"}]

        with pytest.raises(RuntimeError, match="Text extraction failed"):
            _raise_if_threshold_exceeded(error_details=errors, total_docs=10, tolerance=None)

    def test_zero_tolerance_means_zero_allowed(self):
        """``tolerance=0.0`` means zero errors are tolerated."""
        errors = [{"file": "x.pdf", "traceback": "boom"}]

        with pytest.raises(RuntimeError, match="Text extraction failed"):
            _raise_if_threshold_exceeded(error_details=errors, total_docs=5, tolerance=0.0)

    def test_full_tolerance_allows_all_errors(self):
        """``tolerance=1.0`` should allow up to ``total_docs`` errors."""
        errors = [{"file": f"f{i}.pdf", "traceback": "err"} for i in range(10)]

        _raise_if_threshold_exceeded(error_details=errors, total_docs=10, tolerance=1.0)

    def test_error_message_includes_file_info(self):
        """The raised error message should include file-level details."""
        errors = [{"file": "broken.pdf", "traceback": "Traceback: some error"}]

        with pytest.raises(RuntimeError, match="broken.pdf"):
            _raise_if_threshold_exceeded(error_details=errors, total_docs=1, tolerance=None)

    def test_error_message_truncates_to_ten(self):
        """At most 10 error details should be shown in the message."""
        errors = [{"file": f"doc_{i}.pdf", "traceback": f"err {i}"} for i in range(15)]

        with pytest.raises(RuntimeError, match="Showing 10 of 15 error"):
            _raise_if_threshold_exceeded(error_details=errors, total_docs=20, tolerance=None)
