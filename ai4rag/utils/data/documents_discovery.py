# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import logging
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any

from ai4rag import handler
from ai4rag.utils.clients.s3 import create_s3_client
from ai4rag.utils.data.constants import SUPPORTED_EXTENSIONS

_logger = logging.getLogger("documents-discovery")
_logger.addHandler(handler)


DOCUMENTS_DESCRIPTOR_FILENAME = "documents_descriptor.json"
SAMPLING_MAX_SIZE_GB: float = 1


class BenchmarkKeyError(ValueError):
    """Benchmark data references documents that are not part of the discovered corpus."""


@dataclass(frozen=True)
class DocumentDescriptor:
    """Metadata for a single document discovered in an S3 bucket.

    Attributes
    ----------
    key : str
        Full S3 object key.  It both fetches the object and identifies the
        document downstream: it becomes the ``DoclingDocument`` name and is
        what benchmark data must reference.
    size_bytes : int
        Object size in bytes.
    """

    key: str
    size_bytes: int


@dataclass(frozen=True)
class DiscoveryResult:
    """Outcome of a document discovery run.

    Attributes
    ----------
    bucket : str
        S3 bucket name.
    prefixes : tuple[str, ...]
        S3 key prefixes used during listing.  A single empty string means the
        whole bucket was listed.
    documents : list[DocumentDescriptor]
        Discovered (and optionally sampled) documents, deduplicated by object
        key across all prefixes.
    total_size_bytes : int
        Combined size of all discovered documents.
    count : int
        Number of discovered documents.
    """

    bucket: str
    prefixes: tuple[str, ...]
    documents: list[DocumentDescriptor]
    total_size_bytes: int
    count: int

    def __post_init__(self) -> None:
        """Freeze the prefix collection even when callers provide a list."""
        object.__setattr__(self, "prefixes", tuple(self.prefixes))

    def to_dict(self) -> dict:
        """Serialise the result to a JSON-compatible dictionary."""
        return {
            "bucket": self.bucket,
            "prefixes": list(self.prefixes),
            "documents": [{"key": d.key, "size_bytes": d.size_bytes} for d in self.documents],
            "total_size_bytes": self.total_size_bytes,
            "count": self.count,
        }

    def save(self, path: str | Path, filename: str = DOCUMENTS_DESCRIPTOR_FILENAME) -> None:
        """Write ``documents_descriptor.json`` into the given directory.

        Parameters
        ----------
        path : str | Path
            Directory where the descriptor file will be created. The
            directory is created if it does not exist.
        filename : str
            Name of the file to be used within the output directory.
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        descriptor_path = out_dir / filename
        with open(descriptor_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        _logger.info("Documents descriptor written to %s", descriptor_path)


# pylint: disable=too-many-locals
def discover_documents(
    bucket_name: str,
    prefixes: str | list[str] | None = None,
    test_data_doc_names: list[str] | None = None,
    sampling_enabled: bool = True,
    sampling_max_size_gb: float = SAMPLING_MAX_SIZE_GB,
    supported_extensions: set[str] | None = None,
    validate_test_data_keys: bool = True,
    s3_client: Any | None = None,
) -> DiscoveryResult:
    """Discover documents across one or more bucket locations and optionally sample them.

    Lists objects under every entry of *prefixes*, merges the results into a
    single corpus deduplicated by object key, filters by file extension, and
    applies size-based sampling when enabled.  The sampling budget is shared by
    the whole union, not applied per prefix.  Documents referenced by
    ``test_data_doc_names`` are prioritized during sampling so that
    benchmark-relevant files are always included when the budget permits.

    Parameters
    ----------
    bucket_name : str
        S3-compatible bucket name.
    prefixes : str | list[str] | None, default=None
        Object-key prefixes to narrow the listing.  A bare string is treated as
        a one-element list.  ``None``, an empty list, or a list containing an
        empty string lists the whole bucket.  Overlapping prefixes are safe:
        objects matched by more than one are kept once.
    test_data_doc_names : list[str] | None, default=None
        Keys of documents referenced by the benchmark test data.  Each is
        matched against the full object key, or -- when the name resolves to
        exactly one document -- against a bare file name.  Matched documents
        are sorted first so that sampling picks them before other files.
    sampling_enabled : bool, default=True
        When ``True``, only documents up to *sampling_max_size_gb* total
        are returned.
    sampling_max_size_gb : float, default=1.0
        Maximum cumulative size (in gigabytes) when sampling is enabled.
    supported_extensions : set[str] | None, default=None
        File extensions to accept.  Defaults to
        :data:`~ai4rag.utils.data.constants.SUPPORTED_EXTENSIONS`.
    validate_test_data_keys : bool, default=True
        When ``True``, every entry of *test_data_doc_names* must identify
        exactly one discovered document, otherwise :class:`BenchmarkKeyError`
        is raised.  Set to ``False`` to downgrade the failure to a warning.
    s3_client : Any | None, default=None
        Pre-configured ``boto3`` S3 client.  When ``None``, one is created
        via :func:`ai4rag.utils.clients.s3.create_s3_client`.

    Returns
    -------
    DiscoveryResult
        Discovery outcome with document metadata.

    Raises
    ------
    RuntimeError
        If no supported documents are found under any of the prefixes.
    BenchmarkKeyError
        If a benchmark key matches no discovered document, or matches more
        than one, while *validate_test_data_keys* is enabled.
    ValueError
        If sampling produces an empty selection.
    """
    if supported_extensions is None:
        supported_extensions = set(SUPPORTED_EXTENSIONS)

    resolved_prefixes = _normalize_prefixes(prefixes)
    ext_tuple = tuple(supported_extensions)
    max_size_bytes = float(sampling_max_size_gb) * 1024**3 if sampling_enabled else float(inf)

    if s3_client is None:
        s3_client = _create_s3_client_with_ssl_fallback(bucket_name, resolved_prefixes[0])
    contents = _list_objects_union(s3_client, bucket_name, resolved_prefixes)
    supported_files = [c for c in contents if c["Key"].endswith(ext_tuple)]

    if not supported_files:
        raise RuntimeError(f"No supported documents found in {_location(bucket_name, resolved_prefixes)}.")

    test_keys: set[str] = set()
    if test_data_doc_names:
        test_keys = _resolve_test_data_keys(
            test_data_doc_names,
            supported_files,
            bucket_name=bucket_name,
            prefixes=resolved_prefixes,
            strict=validate_test_data_keys,
        )
        supported_files.sort(key=lambda c: c["Key"] not in test_keys)

    total_size = 0
    selected: list[DocumentDescriptor] = []
    for file_info in supported_files:
        size = file_info["Size"]
        if total_size + size > max_size_bytes:
            continue
        selected.append(DocumentDescriptor(key=file_info["Key"], size_bytes=size))
        total_size += size

    if not selected:
        raise ValueError(
            "No documents to process. Check that the bucket/prefixes are correct and contain supported files."
        )

    dropped = sorted(test_keys - {d.key for d in selected})
    if dropped:
        _logger.warning(
            "%d benchmark-referenced document(s) could not fit within the %.2f GB sampling budget and were skipped: %s",
            len(dropped),
            sampling_max_size_gb,
            ", ".join(dropped),
        )

    result = DiscoveryResult(
        bucket=bucket_name,
        prefixes=tuple(resolved_prefixes),
        documents=selected,
        total_size_bytes=total_size,
        count=len(selected),
    )
    _logger.info(
        "Discovered %d document(s) across %d location(s), total size %d bytes",
        result.count,
        len(resolved_prefixes),
        result.total_size_bytes,
    )
    return result


def _normalize_prefixes(prefixes: str | list[str] | None) -> list[str]:
    """Coerce the *prefixes* argument into a deduplicated list of key prefixes.

    A bare string becomes a one-element list.  Entries are stripped of
    surrounding whitespace and of a leading ``/`` (object keys never start with
    one).  Order is preserved.  An empty result, or any entry that is itself
    empty, collapses to ``[""]`` -- the whole bucket, which subsumes every
    other prefix.
    """
    if prefixes is None:
        prefixes = []
    elif isinstance(prefixes, str):
        prefixes = [prefixes]

    normalized = list(dict.fromkeys((prefix or "").strip().lstrip("/") for prefix in prefixes))

    if not normalized or "" in normalized:
        if len(normalized) > 1:
            _logger.info("An empty prefix was given alongside others; listing the whole bucket instead.")
        return [""]
    return normalized


def _location(bucket_name: str, prefixes: list[str]) -> str:
    """Render the listed locations for log and error messages."""
    if prefixes == [""]:
        return f"s3://{bucket_name}"
    return ", ".join(f"s3://{bucket_name}/{prefix}" for prefix in prefixes)


def _list_objects(s3_client: Any, bucket_name: str, prefix: str) -> list[dict]:
    """List every object under *prefix*, following pagination to the end."""
    paginator = s3_client.get_paginator("list_objects_v2")
    contents: list[dict] = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        contents.extend(page.get("Contents", []))
    return contents


def _list_objects_union(s3_client: Any, bucket_name: str, prefixes: list[str]) -> list[dict]:
    """List all *prefixes* and merge them into one key-deduplicated, key-sorted list."""
    merged: dict[str, dict] = {}
    for prefix in prefixes:
        objects = _list_objects(s3_client, bucket_name, prefix)
        _logger.info("Listed %d object(s) under s3://%s/%s", len(objects), bucket_name, prefix)
        for obj in objects:
            merged.setdefault(obj["Key"], obj)
    return sorted(merged.values(), key=lambda obj: obj["Key"])


def _resolve_test_data_keys(
    test_data_doc_names: list[str],
    supported_files: list[dict],
    bucket_name: str,
    prefixes: list[str],
    strict: bool,
) -> set[str]:
    """Map benchmark document names onto discovered object keys.

    An exact object-key match always wins.  A bare file name is accepted only
    when it identifies exactly one document in the corpus -- across several
    locations the same file name can appear more than once, and guessing which
    one the benchmark meant would silently score against the wrong document.

    Raises
    ------
    BenchmarkKeyError
        When *strict* and any name matches no document or more than one.
    """
    all_keys = {c["Key"] for c in supported_files}
    by_basename: dict[str, list[str]] = {}
    for key in all_keys:
        by_basename.setdefault(Path(key).name, []).append(key)

    resolved: set[str] = set()
    missing: list[str] = []
    ambiguous: list[tuple[str, list[str]]] = []

    for name in test_data_doc_names:
        if name in all_keys:
            resolved.add(name)
            continue
        candidates = by_basename.get(name, [])
        if len(candidates) == 1:
            resolved.add(candidates[0])
        elif candidates:
            ambiguous.append((name, sorted(candidates)))
        else:
            missing.append(name)

    if missing or ambiguous:
        message = _format_unresolved_keys(missing, ambiguous, bucket_name, prefixes)
        if strict:
            raise BenchmarkKeyError(message)
        _logger.warning(message)

    return resolved


def _format_unresolved_keys(
    missing: list[str],
    ambiguous: list[tuple[str, list[str]]],
    bucket_name: str,
    prefixes: list[str],
) -> str:
    """Build the error message for benchmark keys that do not identify one document."""
    lines = ["Benchmark data references documents that are not part of the discovered corpus."]
    if missing:
        lines.append("Not found: " + ", ".join(f"'{name}'" for name in sorted(missing)) + ".")
    for name, candidates in sorted(ambiguous):
        lines.append(
            f"'{name}' is a file name shared by {len(candidates)} objects "
            f"({', '.join(candidates)}); use the full object key instead."
        )
    lines.append(
        "Every 'correct_answer_document_keys' entry must be the full object key including its "
        "prefix, for example 'product-manuals/xr-200-manual.pdf'."
    )
    lines.append(f"Searched {_location(bucket_name, prefixes)}.")
    return " ".join(lines)


def _create_s3_client_with_ssl_fallback(bucket_name: str, probe_prefix: str) -> Any:
    """Create an S3 client, retrying with ``verify=False`` on SSL errors.

    Returns
    -------
    Any
        A client that has successfully listed against *bucket_name*.
    """
    from botocore.exceptions import SSLError

    client = create_s3_client()
    try:
        client.list_objects_v2(Bucket=bucket_name, Prefix=probe_prefix, MaxKeys=1)
        return client
    except SSLError:
        _logger.warning(
            "SSL error when listing objects in s3://%s/%s, retrying with verify=False",
            bucket_name,
            probe_prefix,
        )
        return create_s3_client(verify=False)
