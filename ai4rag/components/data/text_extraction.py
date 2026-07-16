# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PaginatedPipelineOptions, ThreadedPdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
    OdtFormatOption,
    OdpFormatOption,
    LatexFormatOption,
    AsciiDocFormatOption,
    EpubFormatOption,
    EmailFormatOption,
)

from ai4rag import handler

_logger = logging.getLogger("text-extraction")
_logger.addHandler(handler)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".md",
        ".html",
        ".txt",
        # New formats
        ".odt",
        ".odp",
        ".adoc",
        ".tex",
        ".epub",
        ".eml",
        ".qmd",
        ".Rmd",
        ".xhtml",
    }
)
DOWNLOAD_MAX_THREADS = 8

# Module-level global used by multiprocessing workers.  Each spawned worker
# initializes its own ``DocumentConverter`` via the pool initializer and
# stores it here so that ``_worker_process_document`` can retrieve it.
_mp_worker_converter = None  # pylint: disable=invalid-name


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a text extraction run.

    Attributes
    ----------
    processed_count : int
        Number of documents successfully extracted.
    total_documents : int
        Total number of input documents.
    error_count : int
        Number of documents that failed during download or extraction.
    """

    processed_count: int
    total_documents: int
    error_count: int


def extract_text(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    documents: list[dict],
    bucket: str,
    output_dir: str | Path,
    s3_endpoint: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
    s3_region: str | None = None,
    error_tolerance: float | None = None,
    max_extraction_workers: int | None = None,
    docling_artifacts_path: str | None = None,
    do_table_structure: bool = False,
) -> ExtractionResult:
    """Download documents from S3 and extract text using Docling.

    Each input document is downloaded from S3, converted to a
    :class:`DoclingDocument` via the Docling library, and persisted as a
    JSON file in *output_dir*.  Conversion runs in a separate process pool
    (``multiprocess`` library, ``"spawn"`` context) while downloads happen
    concurrently in a thread pool.

    Parameters
    ----------
    documents
        List of document descriptor dicts, each with at least a ``"key"``
        and ``"size_bytes"`` entry (as produced by
        :func:`~ai4rag.components.data.documents_discovery.discover_documents`).
    bucket
        S3-compatible bucket name.
    output_dir
        Local directory where DoclingDocument JSON files are written.
    s3_endpoint
        S3-compatible endpoint URL.  Falls back to ``AWS_S3_ENDPOINT``.
    s3_access_key
        AWS access key.  Falls back to ``AWS_ACCESS_KEY_ID``.
    s3_secret_key
        AWS secret key.  Falls back to ``AWS_SECRET_ACCESS_KEY``.
    s3_region
        AWS region.  Falls back to ``AWS_DEFAULT_REGION``.
    error_tolerance
        Fraction of documents (0.0--1.0) allowed to fail.  ``None`` means
        zero tolerance.
    max_extraction_workers
        Number of parallel worker processes.  Defaults to
        ``min(max(1, cpu_count // 2), 8)``.
    docling_artifacts_path
        Path to pre-downloaded Docling model artifacts for offline use.
        Falls back to ``DOCLING_ARTIFACTS_PATH`` environment variable.
    do_table_structure
        Whether Docling should run TableFormer to reconstruct table
        rows and columns from detected PDF layout.  ``False`` (default)
        disables table structure parsing; ``True`` enables it to achieve
        increased accuracy of extracted data.

    Returns
    -------
    ExtractionResult
        Summary of the extraction run.

    Raises
    ------
    RuntimeError
        If the error count exceeds the allowed tolerance.
    """
    import tempfile

    import multiprocess as multiprocessing

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not documents:
        _logger.info("No documents to process.")
        return ExtractionResult(processed_count=0, total_documents=0, error_count=0)

    s3_creds = _resolve_s3_credentials(s3_endpoint, s3_access_key, s3_secret_key, s3_region)
    artifacts_path = _resolve_artifacts_path(docling_artifacts_path)

    _logger.info("Docling table structure parsing: %s", do_table_structure)

    documents = sorted(documents, key=lambda d: d.get("size_bytes", 0), reverse=True)

    effective_workers = _effective_worker_count(max_extraction_workers)
    _logger.info(
        "Starting text extraction for %d documents. extraction_workers=%d, download_threads=%d.",
        len(documents),
        effective_workers,
        DOWNLOAD_MAX_THREADS,
    )

    if artifacts_path is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    mp_context = multiprocessing.get_context("spawn")  # pylint: disable=no-member
    with (
        tempfile.TemporaryDirectory() as download_dir,
        mp_context.Pool(
            processes=effective_workers,
            initializer=_text_extraction_pool_initializer,
            initargs=(do_table_structure,),
        ) as process_pool,
    ):
        download_start = time.perf_counter()
        extraction_tasks, download_errors = _download_and_submit(
            docs=documents,
            bucket=bucket,
            download_path=Path(download_dir),
            process_pool=process_pool,
            out_dir=out_dir,
            s3_creds=s3_creds,
        )
        _logger.info(
            "Downloads finished in %.1fs; %d file(s) queued for extraction, %d download error(s).",
            time.perf_counter() - download_start,
            len(extraction_tasks),
            len(download_errors),
        )
        _raise_if_threshold_exceeded(download_errors, len(documents), error_tolerance)

        extraction_errors: list[dict] = []
        processed_count = 0
        pending = list(extraction_tasks)
        completed = 0

        while pending:
            still_pending = []
            for file_path, task in pending:
                if task.ready():
                    completed += 1
                    try:
                        success, tb = task.get()
                    except Exception:
                        tb = traceback.format_exc()
                        _logger.error("Worker crashed for %s:\n%s", file_path, tb)
                        success = False
                    Path(file_path).unlink(missing_ok=True)
                    if success:
                        processed_count += 1
                    else:
                        extraction_errors.append({"file": file_path, "traceback": tb})
                    _logger.info("Extraction progress %d/%d", completed, len(extraction_tasks))
                else:
                    still_pending.append((file_path, task))
            pending = still_pending
            if pending:
                time.sleep(0.01)

    all_errors = download_errors + extraction_errors
    total_errors = len(all_errors)
    _logger.info(
        "Text extraction completed. Total processed: %d/%d, Errors: %d",
        processed_count,
        len(documents),
        total_errors,
    )
    _raise_if_threshold_exceeded(
        error_details=all_errors,
        total_docs=len(documents),
        tolerance=error_tolerance,
    )

    return ExtractionResult(
        processed_count=processed_count,
        total_documents=len(documents),
        error_count=total_errors,
    )


def _resolve_s3_credentials(
    endpoint: str | None,
    access_key: str | None,
    secret_key: str | None,
    region: str | None,
) -> dict[str, str | None]:
    """Build an S3 credentials dict, falling back to environment variables."""
    creds = {
        "AWS_S3_ENDPOINT": endpoint or os.environ.get("AWS_S3_ENDPOINT"),
        "AWS_ACCESS_KEY_ID": access_key or os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_DEFAULT_REGION": region or os.environ.get("AWS_DEFAULT_REGION"),
    }
    missing = [k for k in ("AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not creds[k]]
    if missing:
        raise ValueError(
            f"Missing S3 credential(s): {missing}. "
            "Pass them explicitly or set the corresponding environment variables."
        )
    return creds


def _make_s3_client(s3_creds: dict[str, str | None], verify: bool = True) -> Any:
    """Create a fresh ``boto3`` S3 client from explicit credentials.

    A fresh session is created on every call so the client is safe to use
    from multiple threads without sharing state.
    """
    import boto3

    session = boto3.session.Session(
        aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
        region_name=s3_creds.get("AWS_DEFAULT_REGION"),
    )
    return session.client(
        service_name="s3",
        endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
        verify=verify,
    )


def _download_document(
    doc: dict,
    bucket: str,
    base_path: Path,
    s3_creds: dict[str, str | None],
) -> Path:
    """Download a single document from S3 with path-traversal protection.

    On an ``SSLError`` the download is retried once with certificate
    verification disabled.

    Parameters
    ----------
    doc
        Document descriptor dict with at least a ``"key"`` field.
    bucket
        S3 bucket name.
    base_path
        Local directory under which the file is saved, preserving the S3
        key as a relative sub-path.
    s3_creds
        Credentials dict for creating per-thread S3 clients.

    Returns
    -------
    Path
        Path to the downloaded local file.
    """
    raw_key = doc["key"]
    safe_key = raw_key.strip().lstrip("/")
    rel = Path(safe_key)
    if not safe_key or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe S3 key (path traversal): {raw_key!r}")

    local_path = (base_path / rel).resolve()
    base_resolved = base_path.resolve()
    if not local_path.is_relative_to(base_resolved):
        raise ValueError(f"Unsafe S3 key (escapes download directory): {raw_key!r}")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    dl_start = time.perf_counter()
    _logger.info("Downloading %s", raw_key)

    from botocore.exceptions import SSLError

    try:
        _make_s3_client(s3_creds).download_file(bucket, raw_key, str(local_path))
    except SSLError:
        _logger.warning("SSL error when downloading %s, retrying with verify=False", raw_key)
        _make_s3_client(s3_creds, verify=False).download_file(bucket, raw_key, str(local_path))

    _logger.info("Download finished %s (%.1fs)", raw_key, time.perf_counter() - dl_start)
    return local_path


def _resolve_artifacts_path(explicit: str | None) -> Path | None:
    """Resolve the Docling artifacts directory.

    Returns ``None`` when no usable artifacts directory is available,
    causing Docling to download models from HuggingFace at runtime.
    """
    raw = explicit or os.environ.get("DOCLING_ARTIFACTS_PATH")
    if not raw:
        _logger.info("DOCLING_ARTIFACTS_PATH not set -- models will be downloaded from HuggingFace.")
        return None
    p = Path(raw)
    if not p.is_dir() or not any(p.iterdir()):
        _logger.warning(
            "DOCLING_ARTIFACTS_PATH=%s is set but the directory is missing or empty "
            "-- falling back to HuggingFace model download.",
            raw,
        )
        return None
    _logger.info("Using local Docling artifacts from %s", p)
    return p


def _build_docling_format_options(do_table_structure: bool = False) -> dict:
    """Build Docling pipeline format options for each supported input format."""
    ap = _resolve_artifacts_path(None)
    accel = AcceleratorOptions(device="cpu", num_threads=2)

    pdf_pipeline_options = ThreadedPdfPipelineOptions(
        artifacts_path=ap,
        do_ocr=False,
        do_table_structure=do_table_structure,
        accelerator_options=accel,
    )
    paginated_pipeline_options = PaginatedPipelineOptions(
        artifacts_path=ap,
        generate_page_images=False,
        accelerator_options=accel,
    )

    return {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
        InputFormat.DOCX: WordFormatOption(pipeline_options=paginated_pipeline_options),
        InputFormat.PPTX: PowerpointFormatOption(pipeline_options=paginated_pipeline_options),
        InputFormat.HTML: HTMLFormatOption(),
        InputFormat.MD: MarkdownFormatOption(),
        InputFormat.ODT: OdtFormatOption(),
        InputFormat.ODP: OdpFormatOption(),
        InputFormat.ASCIIDOC: AsciiDocFormatOption(),
        InputFormat.LATEX: LatexFormatOption(),
        InputFormat.EPUB: EpubFormatOption(),
        InputFormat.EMAIL: EmailFormatOption(),
    }


def _text_extraction_pool_initializer(do_table_structure: bool = False) -> None:
    """Pool initializer that creates a ``DocumentConverter`` per worker process."""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if _resolve_artifacts_path(None) is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    worker_log = logging.getLogger("text_extraction_worker")
    worker_log.setLevel(logging.INFO)
    worker_log.propagate = False
    if not worker_log.handlers:
        worker_log.addHandler(logging.StreamHandler(sys.stdout))

    worker_pid = os.getpid()
    init_start = time.perf_counter()
    worker_log.debug("Worker pid=%s: loading DocumentConverter.", worker_pid)

    mod = sys.modules[__name__]
    # pylint: disable=protected-access
    mod._mp_worker_converter = DocumentConverter(format_options=_build_docling_format_options(do_table_structure))
    worker_log.debug(
        "Worker pid=%s: DocumentConverter ready (%.1fs)",
        worker_pid,
        time.perf_counter() - init_start,
    )


def _worker_process_document(file_path_str: str, output_dir_str: str) -> tuple[bool, str | None]:
    """Convert a single document to a DoclingDocument JSON file.

    Plain-text (``.txt``) files are wrapped in a minimal
    ``DoclingDocument``.  All other supported formats are converted via
    the ``DocumentConverter`` created by the pool initializer.

    Parameters
    ----------
    file_path_str
        Absolute path to the local input file.
    output_dir_str
        Absolute path to the directory where the resulting JSON file
        will be written (named ``<original_filename>.json``).

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` on success; ``(False, error_message)`` on failure.
    """
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    worker_log = logging.getLogger("text_extraction_worker")
    start = time.perf_counter()

    try:
        input_file = Path(file_path_str)
        output_dir = Path(output_dir_str)
        output_file = output_dir / f"{input_file.name}.json"

        if input_file.suffix.lower() == ".txt":
            doc = DoclingDocument(name=input_file.name)
            doc.add_text(label=DocItemLabel.TEXT, text=input_file.read_text(encoding="utf-8"))
            doc.save_as_json(output_file)
            return True, None

        converter = getattr(sys.modules[__name__], "_mp_worker_converter", None)
        if converter is None:
            return False, (
                f"Worker pid={os.getpid()} has no DocumentConverter. "
                "Pool initializer did not run or failed before setting _mp_worker_converter."
            )

        file_size_mib = input_file.stat().st_size / (1024 * 1024) if input_file.exists() else 0.0
        worker_log.info(
            "pid=%s docling convert start: %s (%.1f MiB on disk)",
            os.getpid(),
            input_file.name,
            file_size_mib,
        )
        conversion_result = converter.convert(input_file)
        conversion_result.document.name = input_file.name
        conversion_result.document.save_as_json(output_file)
        worker_log.info(
            "pid=%s docling convert done: %s -> %s (%.1fs)",
            os.getpid(),
            input_file.name,
            output_file.name,
            time.perf_counter() - start,
        )
        return True, None

    except Exception:
        error_tb = traceback.format_exc()
        worker_log.error("Failed to process %s:\n%s", file_path_str, error_tb)
        return False, error_tb


def _download_and_submit(  # pylint: disable=too-many-locals
    docs: list[dict],
    bucket: str,
    download_path: Path,
    process_pool: Any,
    out_dir: Path,
    s3_creds: dict[str, str | None],
) -> tuple[list[tuple[str, Any]], list[dict]]:
    """Download all documents from S3, then submit for extraction largest-first.

    Documents with unsupported extensions are filtered out before any
    downloads begin.  Supported documents are downloaded concurrently,
    then sorted by size descending before being submitted to the process
    pool to avoid the straggler problem.

    Parameters
    ----------
    docs
        Document descriptor dicts.
    bucket
        S3 bucket name.
    download_path
        Local temporary directory for downloaded files.
    process_pool
        Active multiprocessing pool.
    out_dir
        Directory where extracted DoclingDocument JSONs are written.
    s3_creds
        S3 credentials dict for per-thread client creation.

    Returns
    -------
    tuple[list[tuple[str, Any]], list[dict]]
        Extraction tasks (path, AsyncResult) and download error dicts.
    """
    download_errors: list[dict] = []
    downloaded_paths: list[Path] = []

    supported = [d for d in docs if Path(d["key"]).suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped = [d for d in docs if Path(d["key"]).suffix.lower() not in SUPPORTED_EXTENSIONS]
    if skipped:
        skipped_keys = ", ".join(d["key"] for d in skipped)
        _logger.warning("Skipping %d document(s) with unsupported extensions: %s", len(skipped), skipped_keys)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_MAX_THREADS) as dl_pool:
        dl_futures = {
            dl_pool.submit(_download_document, doc, bucket, download_path, s3_creds): doc for doc in supported
        }
        for dl_future in as_completed(dl_futures):
            doc = dl_futures[dl_future]
            key = doc.get("key", "?") if isinstance(doc, dict) else "?"
            try:
                local_path = dl_future.result()
            except Exception as exc:
                exc_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                _logger.warning("Download failed for key=%s: %s", key, exc)
                download_errors.append({"file": key, "traceback": exc_tb})
                continue
            downloaded_paths.append(local_path)

    downloaded_paths.sort(key=lambda p: p.stat().st_size, reverse=True)
    extraction_tasks = [
        (str(lp), process_pool.apply_async(_worker_process_document, (str(lp), str(out_dir))))
        for lp in downloaded_paths
    ]
    return extraction_tasks, download_errors


def _raise_if_threshold_exceeded(
    error_details: list[dict],
    total_docs: int,
    tolerance: float | None,
) -> None:
    """Raise if the error count exceeds the allowed tolerance.

    Parameters
    ----------
    error_details
        Accumulated error dicts with ``"file"`` and ``"traceback"`` keys.
    total_docs
        Total number of input documents.
    tolerance
        Fraction (0.0--1.0) that may fail.  ``None`` means zero tolerance.

    Raises
    ------
    RuntimeError
        When errors exceed the allowed count.
    """
    n_errors = len(error_details)
    if n_errors == 0:
        return

    allowed = 0 if tolerance is None else int(tolerance * total_docs)
    if n_errors <= allowed:
        return

    tolerance_str = "0 (none allowed)" if tolerance is None else f"{tolerance:.0%} of {total_docs}"
    shown = error_details[:10]
    lines = [
        f"Text extraction failed: {n_errors}/{total_docs} document(s) failed (tolerance: {tolerance_str}).",
        f"Showing {len(shown)} of {n_errors} error(s):",
    ]
    for i, err in enumerate(shown, 1):
        tb_lines = err["traceback"].strip().splitlines()
        snippet = "\n    ".join(tb_lines[-5:])
        lines.append(f"\n  [{i}] {err['file']}\n    {snippet}")
    raise RuntimeError("\n".join(lines))


def _effective_worker_count(requested: int | None) -> int:
    """Determine the number of extraction worker processes."""
    if requested is not None:
        return max(1, requested)
    return min(max(1, (os.cpu_count() or 1) // 2), 8)
