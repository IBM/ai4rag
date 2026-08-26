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
from ai4rag.components.data.constants import SUPPORTED_EXTENSIONS
from ai4rag.components.utils.s3 import create_s3_client

_logger = logging.getLogger("documents-discovery")
_logger.addHandler(handler)


DOCUMENTS_DESCRIPTOR_FILENAME = "documents_descriptor.json"
SAMPLING_MAX_SIZE_GB: float = 1


@dataclass(frozen=True)
class DocumentDescriptor:
    """Metadata for a single document discovered in an S3 bucket.

    Attributes
    ----------
    key : str
        Full S3 object key.
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
    prefix : str
        S3 key prefix used during listing.
    documents : list[DocumentDescriptor]
        Discovered (and optionally sampled) documents.
    total_size_bytes : int
        Combined size of all discovered documents.
    count : int
        Number of discovered documents.
    """

    bucket: str
    prefix: str
    documents: list[DocumentDescriptor]
    total_size_bytes: int
    count: int

    def to_dict(self) -> dict:
        """Serialise the result to a JSON-compatible dictionary."""
        return {
            "bucket": self.bucket,
            "prefix": self.prefix,
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


def discover_documents(  # pylint: disable=too-many-locals
    bucket_name: str,
    prefix: str = "",
    test_data_doc_names: list[str] | None = None,
    sampling_enabled: bool = True,
    sampling_max_size_gb: float = SAMPLING_MAX_SIZE_GB,
    supported_extensions: set[str] | None = None,
    s3_client: Any | None = None,
) -> DiscoveryResult:
    """Discover documents in an S3-compatible bucket and optionally sample them.

    Lists objects under *bucket_name*/*prefix*, filters by file extension,
    and applies size-based sampling when enabled.  Documents referenced by
    ``test_data_doc_names`` are prioritized during sampling so that
    benchmark-relevant files are always included when the budget permits.

    Parameters
    ----------
    bucket_name : str
        S3-compatible bucket name.
    prefix : str, default=""
        Object-key prefix to narrow the listing.
    test_data_doc_names : list[str] | None, default=None
        Filenames (stem + extension, no path) of documents referenced by
        the benchmark test data.  These are sorted first so that sampling
        picks them before other files.
    sampling_enabled : bool, default=True
        When ``True``, only documents up to *sampling_max_size_gb* total
        are returned.
    sampling_max_size_gb : float, default=1.0
        Maximum cumulative size (in gigabytes) when sampling is enabled.
    supported_extensions : set[str] | None, default=None
        File extensions to accept.  Defaults to
        :data:`~ai4rag.components.data.constants.SUPPORTED_EXTENSIONS`.
    s3_client : Any | None, default=None
        Pre-configured ``boto3`` S3 client.  When ``None``, one is created
        via :func:`ai4rag.components._s3.create_s3_client`.

    Returns
    -------
    DiscoveryResult
        Discovery outcome with document metadata.

    Raises
    ------
    RuntimeError
        If no supported documents are found in the bucket.
    ValueError
        If sampling produces an empty selection.
    """
    if supported_extensions is None:
        supported_extensions = set(SUPPORTED_EXTENSIONS)

    ext_tuple = tuple(supported_extensions)
    max_size_bytes = float(sampling_max_size_gb) * 1024**3 if sampling_enabled else float(inf)

    if s3_client is None:
        s3_client, contents = _list_objects_with_ssl_fallback(bucket_name, prefix)
    else:
        contents = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", [])
    supported_files = [c for c in contents if c["Key"].endswith(ext_tuple)]

    if not supported_files:
        raise RuntimeError("No supported documents found.")

    if test_data_doc_names:
        test_names_set = set(test_data_doc_names)
        test_keys = {c["Key"] for c in supported_files if Path(c["Key"]).name in test_names_set}
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
            "No documents to process. Check that the bucket/prefix is correct and contains supported files."
        )

    result = DiscoveryResult(
        bucket=bucket_name,
        prefix=prefix,
        documents=selected,
        total_size_bytes=total_size,
        count=len(selected),
    )
    _logger.info("Discovered %d document(s), total size %d bytes", result.count, result.total_size_bytes)
    return result


def _list_objects_with_ssl_fallback(bucket_name: str, prefix: str) -> tuple[Any, list[dict]]:
    """List S3 objects, retrying with ``verify=False`` on SSL errors.

    Returns
    -------
    tuple[Any, list[dict]]
        The S3 client and the ``Contents`` list from ``list_objects_v2``.
    """
    from botocore.exceptions import SSLError

    try:
        client = create_s3_client()
        contents = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", [])
        return client, contents
    except SSLError:
        _logger.warning(
            "SSL error when listing objects in s3://%s/%s, retrying with verify=False",
            bucket_name,
            prefix,
        )
        client = create_s3_client(verify=False)
        contents = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", [])
        return client, contents
