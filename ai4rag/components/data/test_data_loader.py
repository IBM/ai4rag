# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import logging
import random
from dataclasses import dataclass
from typing import Any

from ai4rag import handler
from ai4rag.components.utils.s3 import create_s3_client

_logger = logging.getLogger("test-data-loader")
_logger.addHandler(handler)

BENCHMARK_RECORD_KEYS_NEW: frozenset[str] = frozenset({"question", "correct_answers", "correct_answer_document_keys"})
BENCHMARK_RECORD_KEYS_OLD: frozenset[str] = frozenset({"question", "correct_answers", "correct_answer_document_ids"})
BENCHMARK_SAMPLE_SIZE: int = 25


class TestDataLoaderError(Exception):
    """Raised when test data cannot be loaded or validated."""


@dataclass(frozen=True)
class TestDataResult:
    """Outcome of loading (and optionally sampling) benchmark test data.

    Attributes
    ----------
    data : list[dict]
        Benchmark records, each containing *question*,
        *correct_answers*, and *correct_answer_document_keys* (or deprecated *correct_answer_document_ids*).
    record_count : int
        Number of records in ``data``.
    sampled : bool
        ``True`` if the data was randomly sampled down.
    """

    data: list[dict]
    record_count: int
    sampled: bool


def load_test_data(
    bucket_name: str,
    key: str,
    benchmark_sample_size: int = BENCHMARK_SAMPLE_SIZE,
    s3_client: Any | None = None,
) -> TestDataResult:
    """Download benchmark test data from S3 and optionally sample it.

    Parameters
    ----------
    bucket_name : str
        S3-compatible bucket containing the test data file.
    key : str
        Full S3 object key to the JSON test data file.
    benchmark_sample_size : int, default=25
        Maximum number of records to keep.  When the dataset exceeds this
        limit a reproducible random sample is drawn (seed 42).  Set to
        ``0`` to disable sampling and keep all records.
    s3_client : Any | None, default=None
        Pre-configured ``boto3`` S3 client.  When ``None``, one is created
        via :func:`ai4rag.components.s3.create_s3_client`.

    Returns
    -------
    TestDataResult
        Loaded (and optionally sampled) benchmark data.

    Raises
    ------
    FileNotFoundError
        If the object does not exist in S3.
    TestDataLoaderError
        If the file is not valid JSON or the records have an unexpected
        structure.
    """
    if not bucket_name:
        raise TypeError("bucket_name must be a non-empty string")

    if s3_client is None:
        s3_client = _make_s3_client_with_ssl_fallback(bucket_name, key)

    raw_data = _download_object(s3_client, bucket_name, key)
    benchmark_data = _parse_and_validate(raw_data)

    sampled = False
    if 0 < benchmark_sample_size < len(benchmark_data):
        original_count = len(benchmark_data)
        rng = random.Random(42)
        benchmark_data = rng.sample(benchmark_data, benchmark_sample_size)
        sampled = True
        _logger.info("Sampled %d records from %d total.", benchmark_sample_size, original_count)
    else:
        _logger.info("No sampling applied; record count: %d.", len(benchmark_data))

    return TestDataResult(data=benchmark_data, record_count=len(benchmark_data), sampled=sampled)


def _make_s3_client_with_ssl_fallback(bucket_name: str, key: str) -> Any:
    """Create an S3 client, retrying with ``verify=False`` on SSL errors.

    Uses ``head_object`` as a lightweight SSL probe.  Non-SSL errors
    (e.g. 404) are silently ignored here -- they will surface with proper
    context in :func:`_download_object`.
    """
    from botocore.exceptions import SSLError

    client = create_s3_client()
    try:
        client.head_object(Bucket=bucket_name, Key=key)
    except SSLError:
        _logger.warning("SSL error when accessing %s, retrying with verify=False.", key)
        return create_s3_client(verify=False)
    return client


def _download_object(s3_client: Any, bucket_name: str, key: str) -> str:
    """Download an S3 object and return its UTF-8 body."""
    from botocore.exceptions import ClientError, SSLError

    _logger.info("Fetching test data from S3: bucket=%r, key=%r.", bucket_name, key)
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
    except SSLError:
        _logger.warning("SSL error when downloading %s, retrying with verify=False.", key)
        s3_client = create_s3_client(verify=False)
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            raise FileNotFoundError(
                f"Test data object not found in S3. bucket={bucket_name!r}, key={key!r}. "
                "Check that the key points to an existing JSON file."
            ) from exc
        raise TestDataLoaderError(f"Failed to fetch {key}: {exc}") from exc
    except Exception as exc:
        raise TestDataLoaderError(f"Failed to fetch {key}: {exc}") from exc

    return response["Body"].read().decode("utf-8")


def _parse_and_validate(raw: str) -> list[dict]:
    """Parse JSON and validate benchmark record structure."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TestDataLoaderError("Test data must be a valid JSON file.") from exc

    if not isinstance(data, list):
        raise TestDataLoaderError("Test data file content must be a list of benchmark records.")

    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise TestDataLoaderError(f"Expected a dict at index {idx}, got {type(record).__name__}: {record!r}")
        record_keys = set(record.keys())
        if record_keys != BENCHMARK_RECORD_KEYS_NEW and record_keys != BENCHMARK_RECORD_KEYS_OLD:
            raise TestDataLoaderError(
                f"Incorrect or incomplete keys in test data record at index {idx}. "
                f"Each record must contain exactly: {sorted(BENCHMARK_RECORD_KEYS_NEW)} or {sorted(BENCHMARK_RECORD_KEYS_OLD)}."
            )

    return data
