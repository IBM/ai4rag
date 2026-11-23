# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import errno
import importlib
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from boto3.s3.transfer import TransferConfig
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PaginatedPipelineOptions,
    RapidOcrOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import (
    AsciiDocFormatOption,
    AudioFormatOption,
    DocumentConverter,
    EmailFormatOption,
    EpubFormatOption,
    HTMLFormatOption,
    ImageFormatOption,
    LatexFormatOption,
    MarkdownFormatOption,
    OdpFormatOption,
    OdtFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)

from ai4rag import handler
from ai4rag.utils.clients.s3 import _download_transfer_config, create_s3_client

from ._extraction_pool import _container_cpu_quota
from .constants import SUPPORTED_EXTENSIONS
from .local_hf_asr_pipeline import (
    LocalHuggingFaceAsrPipeline,
    LocalHuggingFaceAsrPipelineOptions,
    _validate_model_directory,
)

_logger = logging.getLogger("text-extraction")
_logger.addHandler(handler)

DOWNLOAD_MAX_THREADS = 8

# OCR language handling
# ---------------------
# There is *no* automatic language detection: OCR is a per-run choice driven by
# the ``ocr_lang`` value supplied by the caller (defaulting to English).  RapidOCR
# ships two model bundles that we support out of the box -- a Latin/English set
# and a Chinese set -- and ``ocr_lang`` selects which one is used (see
# ``_rapidocr_artifacts_rel_paths``).  Latin-script languages are all served by
# the English models, so passing e.g. ``"french"`` still resolves to the English
# bundle; only Chinese switches to the dedicated Chinese models.
DEFAULT_OCR_LANG: tuple[str, ...] = ("english",)
OCR_EXTENSIONS = frozenset({".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"})
_CHINESE_OCR_LANGS = frozenset({"chinese", "ch", "zh", "zho", "chi", "zh-cn", "zh-tw"})
_OCR_ARTIFACT_REPO = "superlinked/docling-artifacts"
_OCR_ARTIFACT_REVISION = "310479be56f3a99a0fea75d930e11eea38bf5ef7"
_ASR_MODEL_REPO = "openai/whisper-tiny"
_ASR_MODEL_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"

# Relative paths Docling expects under ``$DOCLING_ARTIFACTS_PATH/RapidOcr/`` for
# the onnxruntime backend (see docling.models.stages.ocr.rapid_ocr_model).  These
# mirror Docling's own on-disk layout, so they live here alongside the code that
# resolves them rather than in ``pipelines-components``.
_RAPIDOCR_BUNDLES = {
    "english": {
        "ocr_det_model_path": "onnx/PP-OCRv4/det/en_PP-OCRv3_det_mobile.onnx",
        "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "ocr_rec_model_path": "onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx",
        "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt",
    },
    "chinese": {
        "ocr_det_model_path": "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
        "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "ocr_rec_model_path": "onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx",
        "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
    },
}
# Optional small models that some rapidocr wheels still ship under ``rapidocr/models/``.
_BUNDLED_RAPIDOCR_DET = "PP-OCRv6_det_small.onnx"
_BUNDLED_RAPIDOCR_CLS = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
_BUNDLED_RAPIDOCR_REC = "PP-OCRv6_rec_small.onnx"

# Module-level state private to each fresh conversion interpreter. Each worker
# initializes its own ``DocumentConverter`` before accepting documents and
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


@dataclass(frozen=True)
class DoclingExtractionConfig:
    """Docling converter settings shared with extraction worker processes.

    This config holds the converter settings callers (e.g.
    ``pipelines-components``) use to control conversion behaviour.  It is
    constructed once and passed to :func:`extract_text`, which copies its
    settings into every isolated worker so each builds an identically
    configured ``DocumentConverter``. The caller's config remains immutable.

    Attributes
    ----------
    do_table_structure
        What: run Docling's TableFormer to reconstruct rows/columns from the
        detected PDF layout.  Why: table structure parsing is comparatively
        expensive, so it stays off by default and is opted into only when the
        corpus contains tables worth reconstructing.
    do_ocr
        What: run RapidOCR on pages Docling flags as needing OCR (scanned PDFs,
        images).  Why: born-digital documents already carry a text layer, so
        OCR is off by default to avoid the runtime cost; enable it for scanned
        or image inputs.
    ocr_lang
        What: the RapidOCR language selection (e.g. ``("english",)`` or
        ``("english", "chinese")``).  Why: there is no auto-detection -- this
        value picks which bundled model set is loaded.  Latin-script languages
        map to the English models; only Chinese switches to the Chinese models
        (see :func:`_rapidocr_artifacts_rel_paths`).  Ignored when ``do_ocr``
        is ``False``. Explicit OCR model paths take precedence over bundled
        wheel models, followed by validated ``DOCLING_ARTIFACTS_PATH``
        artifacts. Without configured artifacts, connected environments
        can use Docling's normal model downloader.
    ocr_det_model_path
        What/Why: optional path to a custom RapidOCR text-*detection* ONNX
        model, for disconnected clusters or specialised model sets that differ
        from the bundled defaults.
    ocr_cls_model_path
        What/Why: optional path to a custom RapidOCR angle-*classification*
        ONNX model (same rationale as ``ocr_det_model_path``).
    ocr_rec_model_path
        What/Why: optional path to a custom RapidOCR text-*recognition* ONNX
        model (same rationale as ``ocr_det_model_path``).
    ocr_rec_keys_path
        What/Why: optional path to the character-keys dictionary matching the
        custom recognition model; required when the recognition model uses a
        non-default character set.
    asr_model_path
        Directory containing the approved local Hugging Face Transformers
        Whisper model. When ``None``, the audio pipeline resolves
        ``HF_MODEL_DIR`` in its worker process. Audio extraction fails closed
        when neither location provides a complete local model. ASR runs on
        CPU independently of ``device`` and never downloads model weights.
    device
        Device for Docling's PDF, image, DOCX, and PPTX pipelines. Defaults to
        ``"cpu"`` so existing extraction runs keep their device selection.
        CUDA OCR requires a GPU-enabled ONNX Runtime.
    """

    do_table_structure: bool = False
    do_ocr: bool = False
    ocr_lang: tuple[str, ...] = DEFAULT_OCR_LANG
    ocr_det_model_path: str | None = None
    ocr_cls_model_path: str | None = None
    ocr_rec_model_path: str | None = None
    ocr_rec_keys_path: str | None = None
    asr_model_path: str | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        # Normalize ``ocr_lang`` in one place so callers may pass a single
        # string or any sequence and always end up with a non-empty tuple.
        object.__setattr__(self, "ocr_lang", _normalize_ocr_lang(self.ocr_lang))


@dataclass(frozen=True)
class PreparedExtraction:
    """Validated model settings for one discovered corpus.

    Attributes
    ----------
    config
        Configuration to pass to :func:`extract_text`.
    artifacts_path
        Local Docling artifacts directory, when one is configured.
    needs_ocr
        Whether the selected documents include a PDF or image.
    needs_asr
        Whether the selected documents include audio.
    """

    config: DoclingExtractionConfig
    artifacts_path: str | None
    needs_ocr: bool
    needs_asr: bool


def prepare_extraction(
    documents: list[dict],
    language: str | None = None,
    *,
    disconnected: bool = False,
    docling_artifacts_path: str | None = None,
    asr_model_path: str | None = None,
) -> PreparedExtraction:
    """Select and validate extraction models for discovered documents.

    Connected runs download the pinned PP-OCRv4 files when no local artifacts
    directory is configured. Disconnected runs require those files locally.
    Audio uses a local Whisper model when configured. Connected runs download
    the pinned Whisper snapshot when no local model is configured.
    Text-only runs require neither model bundle.

    Parameters
    ----------
    documents
        Selected document descriptors containing a ``key``.
    language
        Document language or pattern language code. Chinese selects the Chinese
        OCR bundle; every other value selects English.
    disconnected
        Whether downloads are unavailable.
    docling_artifacts_path
        Local Docling artifacts directory. Defaults to ``DOCLING_ARTIFACTS_PATH``.
    asr_model_path
        Local Whisper directory. Defaults to ``HF_MODEL_DIR`` when set.

    Returns
    -------
    PreparedExtraction
        The validated configuration and local artifacts path for extraction.
    """
    extensions = {Path(document["key"]).suffix.lower() for document in documents}
    needs_ocr = bool(extensions & OCR_EXTENSIONS)
    needs_asr = bool(extensions & AUDIO_EXTENSIONS)
    bundle = "chinese" if (language or "").strip().lower() in _CHINESE_OCR_LANGS else "english"
    model_paths, local_path = (
        _prepare_ocr_model_paths(bundle, disconnected, docling_artifacts_path) if needs_ocr else ({}, None)
    )
    local_asr_path = _prepare_asr_model_path(asr_model_path, disconnected) if needs_asr else None
    config = DoclingExtractionConfig(do_ocr=needs_ocr, ocr_lang=bundle, asr_model_path=local_asr_path, **model_paths)
    return PreparedExtraction(
        config=config,
        artifacts_path=local_path,
        needs_ocr=needs_ocr,
        needs_asr=needs_asr,
    )


def _prepare_ocr_model_paths(
    bundle: str, disconnected: bool, docling_artifacts_path: str | None
) -> tuple[dict[str, str], str | None]:
    """Resolve and validate the selected RapidOCR files."""
    local_path = docling_artifacts_path or os.getenv("DOCLING_ARTIFACTS_PATH")
    relative_paths = _RAPIDOCR_BUNDLES[bundle]
    if local_path:
        root = Path(local_path).expanduser().resolve()
        model_paths = {name: str(root / "RapidOcr" / relative) for name, relative in relative_paths.items()}
        local_path = str(root)
    elif disconnected:
        raise FileNotFoundError(
            "PDF or image documents need DOCLING_ARTIFACTS_PATH on a disconnected cluster. "
            "Mount the approved Docling/RapidOCR bundle before running the notebook."
        )
    else:
        from huggingface_hub import hf_hub_download

        model_paths = {
            name: hf_hub_download(
                repo_id=_OCR_ARTIFACT_REPO,
                filename=f"RapidOcr/{relative}",
                revision=_OCR_ARTIFACT_REVISION,
            )
            for name, relative in relative_paths.items()
        }
    missing = sorted(path for path in model_paths.values() if not Path(path).is_file())
    if missing:
        source = "RapidOCR" if local_path else "Downloaded RapidOCR"
        raise FileNotFoundError(f"{source} {bundle} models are missing:\n  - " + "\n  - ".join(missing))
    return model_paths, local_path


def _prepare_asr_model_path(asr_model_path: str | None, disconnected: bool) -> str:
    """Use a local Whisper model or download the pinned connected snapshot."""
    local_path = asr_model_path or os.getenv("HF_MODEL_DIR")
    if not local_path and not disconnected:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=_ASR_MODEL_REPO,
            revision=_ASR_MODEL_REVISION,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
    return _validate_model_directory(local_path)


def extract_text(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-statements
    documents: list[dict],
    bucket: str,
    output_dir: str | Path,
    s3_endpoint: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
    s3_region: str | None = None,
    ssl_cert_path: str | None = None,
    error_tolerance: float | None = None,
    max_extraction_workers: int | None = None,
    docling_artifacts_path: str | None = None,
    docling_config: DoclingExtractionConfig | None = None,
    s3_client: Any | None = None,
    *,
    worker_startup_timeout: float = 300.0,
    document_timeout: float = 600.0,
) -> ExtractionResult:
    """Download documents from S3 and extract text using Docling.

    Each input document is downloaded from S3, converted to a
    :class:`DoclingDocument` via the Docling library, and persisted as a
    JSON file in *output_dir*. Downloads use a parent-owned thread pool.
    Fresh-interpreter conversion workers are reused across bounded batches
    of at most twice the worker count, with larger files scheduled first.

    Parameters
    ----------
    documents
        List of document descriptor dicts, each with at least a ``"key"``
        and ``"size_bytes"`` entry (as produced by
        :func:`~ai4rag.utils.data.documents_discovery.discover_documents`).
        The ``"key"`` entry also names the extracted document.
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
    ssl_cert_path
        Optional path to a PEM CA bundle used to verify the S3 endpoint. When
        provided, it takes precedence over ``AWS_CA_BUNDLE``. The file must be
        readable; certificate validation remains enabled.
    error_tolerance
        Fraction of documents (0.0--1.0) allowed to fail.  ``None`` means
        zero tolerance.
    max_extraction_workers
        Number of parallel conversion processes, limited to the supported document
        count. By default uses half the available CPUs (including affinity and
        container quota), at least one and at most eight. Positive explicit
        overrides may exceed the CPU budget; each worker loads its own models.
        Zero and negative overrides retain their historical meaning of one.
    docling_artifacts_path
        Path to pre-downloaded Docling model artifacts for offline use.
        Falls back to ``DOCLING_ARTIFACTS_PATH`` environment variable.
    docling_config
        Ready :class:`DoclingExtractionConfig` controlling table-structure,
        OCR, and device selection. Callers (e.g. ``pipelines-components``) construct
        it once and pass it here; it is forwarded unchanged to every worker
        process.  ``None`` (default) uses ``DoclingExtractionConfig()`` --
        table structure and OCR both disabled.  See
        :class:`DoclingExtractionConfig` for the per-field ``what``/``why``
        and for how ``ocr_lang`` maps to bundled OCR models.
    s3_client
        Pre-configured S3 client used for every download. When omitted, one
        client is created from the S3 credentials supplied to this function or
        from the environment. The client is shared only by download threads;
        it is never sent to Docling worker processes.

    worker_startup_timeout
        Maximum seconds for each worker to initialize, including imports.
        Defaults to 300 seconds. Must be positive and finite.
    document_timeout
        Maximum seconds for a worker to convert one document, including lazy
        model loading. Defaults to 600 seconds. Must be positive and finite.
        A worker MemoryError or SIGKILL allows one fresh-pool retry of unfinished
        work per extraction run. SIGKILL is treated as possible memory exhaustion.
        Timeouts, other worker failures, and exhausted recovery abort the run
        regardless of error tolerance. Recovery requires the parent to survive.

    Returns
    -------
    ExtractionResult
        Summary of the extraction run.

    Raises
    ------
    RuntimeError
        If CUDA OCR lacks an ONNX Runtime GPU provider, or the error count
        exceeds the allowed tolerance.
    """
    import tempfile

    from ._extraction_pool import ConversionPool

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not documents:
        _logger.info("No documents to process.")
        return ExtractionResult(processed_count=0, total_documents=0, error_count=0)

    for name, value in (("worker_startup_timeout", worker_startup_timeout), ("document_timeout", document_timeout)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    pipeline_config = docling_config or DoclingExtractionConfig()
    _require_cuda_ocr_provider(pipeline_config)

    if s3_client is None:
        s3_creds = _resolve_s3_credentials(s3_endpoint, s3_access_key, s3_secret_key, s3_region)
        s3_client = _make_s3_client(s3_creds, ssl_cert_path=ssl_cert_path)
    artifacts_path = _resolve_artifacts_path(docling_artifacts_path)

    has_custom_models = bool(
        pipeline_config.ocr_det_model_path
        or pipeline_config.ocr_cls_model_path
        or pipeline_config.ocr_rec_model_path
        or pipeline_config.ocr_rec_keys_path
    )
    _logger.info("Docling table structure parsing: %s", pipeline_config.do_table_structure)
    _logger.info(
        "Docling OCR (RapidOCR): enabled=%s lang=%s custom_models=%s",
        pipeline_config.do_ocr,
        pipeline_config.ocr_lang if pipeline_config.do_ocr else (),
        has_custom_models,
    )

    documents = sorted(documents, key=lambda d: d.get("size_bytes", 0), reverse=True)

    supported = [doc for doc in documents if Path(doc["key"]).suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped = len(documents) - len(supported)
    if skipped:
        _logger.warning("Skipping %d document(s) with unsupported extensions.", skipped)
    if not supported:
        return ExtractionResult(processed_count=0, total_documents=len(documents), error_count=0)
    effective_workers = min(len(supported), _effective_worker_count(max_extraction_workers))
    _logger.info(
        "Starting text extraction for %d documents. extraction_workers=%d, download_threads=%d.",
        len(documents),
        effective_workers,
        DOWNLOAD_MAX_THREADS,
    )

    all_errors: list[dict] = []
    processed_count = 0
    transfer_config = _download_transfer_config()
    # Separate, bounded download/conversion phases keep ownership simple and
    # preserve largest-first conversion within each batch. No unbounded task
    # queue or full-corpus staging is needed.
    with (
        tempfile.TemporaryDirectory() as download_dir,
        tempfile.TemporaryDirectory(prefix=".text-extraction-", dir=out_dir) as conversion_dir,
        ConversionPool(
            workers=effective_workers,
            config=asdict(pipeline_config),
            artifacts_path=artifacts_path,
            startup_timeout=worker_startup_timeout,
            conversion_timeout=document_timeout,
        ) as process_pool,
    ):
        input_dir = Path(download_dir).resolve() / "inputs"
        # Stage JSON on the artifact filesystem so publishing stays atomic even
        # when the download directory and KFP artifact are on different mounts.
        staged_output = Path(conversion_dir).resolve()
        batch_size = 2 * effective_workers
        for offset in range(0, len(supported), batch_size):
            downloaded, download_errors = _download_documents(
                docs=supported[offset : offset + batch_size],
                bucket=bucket,
                download_path=input_dir,
                s3_client=s3_client,
                transfer_config=transfer_config,
            )
            all_errors.extend(download_errors)
            _raise_if_threshold_exceeded(all_errors, len(documents), error_tolerance)
            for file_path, success, error in process_pool.convert(downloaded, staged_output):
                file_path.unlink(missing_ok=True)
                if success:
                    relative = file_path.relative_to(input_dir).with_suffix(file_path.suffix + ".json")
                    destination = out_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    (staged_output / relative).replace(destination)
                    processed_count += 1
                else:
                    all_errors.append({"file": str(file_path), "traceback": error})
                _logger.info("Extraction progress %d/%d", processed_count + len(all_errors), len(supported))

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


def _make_s3_client(s3_creds: dict[str, str | None], ssl_cert_path: str | None = None) -> Any:
    """Create the S3 client used by an extraction run.

    Keeping client construction in :func:`create_s3_client` makes extraction
    use the same TLS and certificate-discovery behaviour as document
    discovery. The returned client is shared by all download threads in the
    run; boto3 clients support concurrent use by threads.
    """
    verify: bool | str = True
    if ssl_cert_path:
        ca_bundle = Path(ssl_cert_path)
        if not ca_bundle.is_file() or not os.access(ca_bundle, os.R_OK):
            raise FileNotFoundError(
                f"ssl_cert_path={ssl_cert_path!r} is not a readable PEM CA bundle. "
                "Mount the trusted CA bundle into the workbench and pass its path, "
                "or configure AWS_CA_BUNDLE."
            )
        verify = str(ca_bundle)

    return create_s3_client(
        endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
        access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
        secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
        region_name=s3_creds.get("AWS_DEFAULT_REGION"),
        verify=verify,
    )


def _download_document(
    doc: dict,
    bucket: str,
    base_path: Path,
    s3_client: Any,
    transfer_config: TransferConfig,
) -> Path:
    """Download one S3 object under ``base_path``, preserving its nested key.

    The configured client is shared by the run's download threads. Reject
    traversal keys and return the absolute local path to the downloaded file.
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

    try:
        s3_client.download_file(bucket, raw_key, str(local_path), Config=transfer_config)
    except BaseException:
        local_path.unlink(missing_ok=True)
        raise

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


def _normalize_ocr_lang(ocr_lang: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize OCR language input to a non-empty tuple of language codes."""
    if ocr_lang is None:
        return DEFAULT_OCR_LANG
    if isinstance(ocr_lang, str):
        langs = (ocr_lang.strip(),) if ocr_lang.strip() else DEFAULT_OCR_LANG
    else:
        langs = tuple(str(lang).strip() for lang in ocr_lang if str(lang).strip())
    return langs or DEFAULT_OCR_LANG


def _rapidocr_artifacts_rel_paths(ocr_lang: tuple[str, ...]) -> tuple[str, ...]:
    """Pick Docling artifact-relative OCR model paths for the requested language."""
    normalized = {lang.strip().lower() for lang in ocr_lang}
    has_chinese = bool(normalized & _CHINESE_OCR_LANGS)
    has_english = bool(normalized & {"english", "en", "eng", "latin"})
    bundle = "chinese" if has_chinese and not has_english else "english"
    model_names = ("ocr_det_model_path", "ocr_cls_model_path", "ocr_rec_model_path")
    return tuple(_RAPIDOCR_BUNDLES[bundle][name] for name in model_names)


def _try_resolve_wheel_rapidocr_model_paths() -> dict[str, str] | None:
    """Return RapidOCR wheel model paths when the installed package still ships ONNX files.

    Only called on the OCR path (``do_ocr=True``).  ``rapidocr`` is a hard
    transitive dependency of ``docling-slim[standard]``, so it must be importable
    whenever OCR is requested; a failed import means a broken install and OCR
    genuinely cannot run, so we raise an actionable error rather than degrade
    silently (a bare ``ImportError`` would otherwise surface deep inside a worker
    process).

    ``None`` is returned only for the *expected* case where rapidocr imports but
    no longer bundles ONNX files (current wheels), letting the caller fall back
    to the ``DOCLING_ARTIFACTS_PATH`` resolution path.
    """
    try:
        import rapidocr
    except ImportError as exc:
        raise RuntimeError(
            "OCR was requested (do_ocr=True) but the 'rapidocr' package is not importable. "
            "It ships with 'docling-slim[standard]', so this indicates a broken install; "
            "reinstall ai4rag (or docling) to restore it, or disable OCR."
        ) from exc

    models_dir = Path(rapidocr.__file__).resolve().parent / "models"
    paths = {
        "det_model_path": models_dir / _BUNDLED_RAPIDOCR_DET,
        "cls_model_path": models_dir / _BUNDLED_RAPIDOCR_CLS,
        "rec_model_path": models_dir / _BUNDLED_RAPIDOCR_REC,
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    return {key: str(path) for key, path in paths.items()}


def _validate_rapidocr_artifacts(ocr_lang: tuple[str, ...]) -> None:
    """Fail fast unless configured artifacts provide RapidOCR models.

    This is called only for disconnected-style runs with a configured artifacts
    path, after both explicit model paths and bundled RapidOCR wheel models
    have been ruled out.
    """
    artifacts = _resolve_artifacts_path(None)
    if artifacts is None:
        raise FileNotFoundError(
            "RapidOCR model files are unavailable. Set DOCLING_ARTIFACTS_PATH to an image-baked "
            "RapidOCR bundle or pass ocr_*_model_path explicitly."
        )
    ocr_root = artifacts / "RapidOcr"
    missing = [str(ocr_root / rel) for rel in _rapidocr_artifacts_rel_paths(ocr_lang) if not (ocr_root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            "RapidOCR models are missing under DOCLING_ARTIFACTS_PATH. Expected files:\n  - "
            + "\n  - ".join(missing)
            + "\nBake them into the AutoRAG image (see tmp/Containerfile.autorag-dev) or pass "
            "ocr_*_model_path explicitly. Current PyPI rapidocr wheels no longer ship ONNX models."
        )


def _require_cuda_ocr_provider(config: DoclingExtractionConfig) -> None:
    """Fail before spawning workers if CUDA OCR cannot use ONNX Runtime's GPU provider."""
    if not config.do_ocr or not config.device.startswith("cuda"):
        return

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("CUDA OCR requires ONNX Runtime with CUDA support, but it is not installed.") from exc

    if "CUDAExecutionProvider" not in ort.get_available_providers() or ort.get_device() != "GPU":
        raise RuntimeError(
            "CUDA OCR was requested, but ONNX Runtime cannot use CUDAExecutionProvider. "
            "Install a compatible GPU-enabled ONNX Runtime in the extraction image."
        )


def _build_rapidocr_options(config: DoclingExtractionConfig) -> RapidOcrOptions:
    """Build Docling ``RapidOcrOptions`` from extraction config.

    Resolution order when custom paths are omitted:

    1. ONNX files shipped inside the ``rapidocr`` package (older / some local installs)
    2. When ``DOCLING_ARTIFACTS_PATH`` is configured, validate the models under
       ``DOCLING_ARTIFACTS_PATH/RapidOcr`` before allowing Docling to load them.
    3. Without configured artifacts, leave model paths unset so connected
       environments can use Docling's normal model downloader.
    """
    kwargs: dict[str, Any] = {
        "lang": list(config.ocr_lang),
        "force_full_page_ocr": False,
        "rapidocr_params": {"EngineConfig.onnxruntime.inter_op_num_threads": 1},
    }
    custom_paths = {
        "det_model_path": config.ocr_det_model_path,
        "cls_model_path": config.ocr_cls_model_path,
        "rec_model_path": config.ocr_rec_model_path,
        "rec_keys_path": config.ocr_rec_keys_path,
    }
    if any(custom_paths.values()):
        for key, value in custom_paths.items():
            if value:
                kwargs[key] = value
        return RapidOcrOptions(**kwargs)

    wheel_paths = _try_resolve_wheel_rapidocr_model_paths()
    if wheel_paths is not None:
        kwargs.update(wheel_paths)
        return RapidOcrOptions(**kwargs)

    if _resolve_artifacts_path(None) is not None:
        _validate_rapidocr_artifacts(config.ocr_lang)
    return RapidOcrOptions(**kwargs)


def _build_docling_format_options(
    do_table_structure: bool = False,
    config: DoclingExtractionConfig | None = None,
) -> dict:
    """Build Docling pipeline format options for each supported input format.

    Parameters
    ----------
    do_table_structure
        Legacy convenience flag used by existing unit tests.  Ignored when
        *config* is provided.
    config
        Full extraction config.  When ``None``, a config is built from
        ``do_table_structure`` with OCR disabled.
    """
    cfg = config or DoclingExtractionConfig(do_table_structure=do_table_structure)
    ap = _resolve_artifacts_path(None)
    accel = AcceleratorOptions(device=cfg.device, num_threads=2)
    ocr_options = _build_rapidocr_options(cfg) if cfg.do_ocr else None

    pdf_kwargs: dict[str, Any] = {
        "artifacts_path": ap,
        "do_ocr": cfg.do_ocr,
        "do_table_structure": cfg.do_table_structure,
        "accelerator_options": accel,
    }
    if ocr_options is not None:
        pdf_kwargs["ocr_options"] = ocr_options

    pdf_pipeline_options = ThreadedPdfPipelineOptions(**pdf_kwargs)

    asr_pipeline_options = LocalHuggingFaceAsrPipelineOptions(
        asr_model_path=cfg.asr_model_path or os.environ.get("HF_MODEL_DIR"),
    )

    paginated_pipeline_options = PaginatedPipelineOptions(
        artifacts_path=ap,
        generate_page_images=False,
        accelerator_options=accel,
    )

    format_options: dict = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pdf_pipeline_options,
            backend=PyPdfiumDocumentBackend,
        ),
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
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=LocalHuggingFaceAsrPipeline,
            pipeline_options=asr_pipeline_options,
        ),
    }
    # Images always go through the PDF/image pipeline so RapidOCR can run when enabled.
    format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=pdf_pipeline_options)
    return format_options


def _text_extraction_pool_initializer(
    config: DoclingExtractionConfig | bool = False,
) -> None:
    """Create a ``DocumentConverter`` in a fresh, thread-limited worker.

    Accepts either a :class:`DoclingExtractionConfig` or a legacy boolean
    ``do_table_structure`` flag for backward compatibility with older call sites.
    """
    if isinstance(config, bool):
        pipeline_config = DoclingExtractionConfig(do_table_structure=config)
    else:
        pipeline_config = config

    # These optional modules are provided by the text-extraction extra, not
    # the lightweight installation used for static analysis and core callers.
    cv2 = importlib.import_module("cv2")
    torch = importlib.import_module("torch")

    # Environment limits were installed before this interpreter imported any
    # native library. Docling's accelerator uses two intra-op threads; bound
    # Torch's independent inter-op pool as well, including the lazy ASR path.
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)  # pylint: disable=no-member

    if _resolve_artifacts_path(None) is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    worker_log = logging.getLogger("text_extraction_worker")
    worker_log.setLevel(logging.INFO)
    worker_log.propagate = False
    if not worker_log.handlers:
        worker_log.addHandler(logging.StreamHandler(sys.stdout))

    mod = sys.modules[__name__]
    # pylint: disable=protected-access
    mod._mp_worker_converter = DocumentConverter(format_options=_build_docling_format_options(config=pipeline_config))


def _worker_process_document(  # pylint: disable=too-many-locals
    file_path_str: str, output_dir_str: str, doc_key: str = ""
) -> tuple[bool, str | None]:
    """Convert an absolute input path to a DoclingDocument JSON under the output directory.

    TXT files are wrapped directly; other formats use the worker's converter.
    The original ``doc_key`` names the document and preserves its nested output
    path, falling back to the input filename when empty. Returns ``(True, None)``
    on success or ``(False, traceback)`` for ordinary document errors. Memory
    exhaustion escapes as ``MemoryError`` so the parent can replace the pool.
    """
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    worker_log = logging.getLogger("text_extraction_worker")
    start = time.perf_counter()

    try:
        input_file = Path(file_path_str)
        output_dir = Path(output_dir_str)

        # The document is named by its key so that files sharing a basename in
        # different folders stay distinct downstream.
        doc_name = doc_key or input_file.name

        # Preserve directory structure in output to prevent collisions when multiple
        # files have the same basename from different S3 subdirectories.  The key is
        # normalized the same way ``_download_document`` normalizes it, so a leading
        # slash cannot turn the output path absolute and escape *output_dir*.
        safe_name = doc_name.strip().lstrip("/")
        output_rel_path = Path(safe_name).with_suffix(Path(safe_name).suffix + ".json")
        output_file = output_dir / output_rel_path

        if input_file.suffix.lower() == ".txt":
            doc = DoclingDocument(name=doc_name)
            doc.add_text(label=DocItemLabel.TEXT, text=input_file.read_text(encoding="utf-8"))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            _save_document(doc, output_file)
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
            doc_name,
            file_size_mib,
        )
        conversion_result = converter.convert(input_file)
        conversion_result.document.name = doc_name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        _save_document(conversion_result.document, output_file)
        worker_log.info(
            "pid=%s docling convert done: %s -> %s (%.1fs)",
            os.getpid(),
            doc_name,
            output_file,
            time.perf_counter() - start,
        )
        return True, None

    except Exception as exc:
        if _is_memory_error(exc):
            raise MemoryError("The conversion runtime exhausted worker memory.") from exc
        error_tb = traceback.format_exc()
        worker_log.error("Failed to process %s:\n%s", file_path_str, error_tb)
        return False, error_tb


def _is_memory_error(error: BaseException | None) -> bool:
    """Recognize allocation failures, including causes wrapped by Docling."""
    # A Torch exception implies Torch is already loaded in this worker.
    torch_memory_error = getattr(sys.modules.get("torch"), "OutOfMemoryError", MemoryError)
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (MemoryError, torch_memory_error)):
            return True
        if isinstance(error, OSError) and getattr(error, "errno", None) == errno.ENOMEM:
            return True
        # Torch's CPU allocator uses RuntimeError in some versions.
        if isinstance(error, RuntimeError) and "DefaultCPUAllocator: can't allocate memory" in str(error):
            return True
        error = error.__cause__ or error.__context__
    return False


def _save_document(document: Any, output_file: Path) -> None:
    """Publish complete JSON only; temporary files are never named ``*.json``."""
    import tempfile

    with tempfile.NamedTemporaryFile(dir=output_file.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        document.save_as_json(temporary_path)
        temporary_path.replace(output_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def _download_documents(  # pylint: disable=too-many-locals
    docs: list[dict],
    bucket: str,
    download_path: Path,
    s3_client: Any,
    transfer_config: TransferConfig | None = None,
) -> tuple[list[tuple[Path, str]], list[dict]]:
    """Download one bounded batch and return (local path, original key) pairs.

    The supplied client stays in the parent process and is shared only by
    the download threads. Results are sorted by actual size, largest first.
    """
    download_errors: list[dict] = []
    downloaded: list[tuple[Path, str]] = []

    supported = [d for d in docs if Path(d["key"]).suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped = [d for d in docs if Path(d["key"]).suffix.lower() not in SUPPORTED_EXTENSIONS]
    if skipped:
        skipped_keys = ", ".join(d["key"] for d in skipped)
        _logger.warning(
            "Skipping %d document(s) with unsupported extensions: %s",
            len(skipped),
            skipped_keys,
        )

    if transfer_config is None:
        transfer_config = _download_transfer_config()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_MAX_THREADS) as dl_pool:
        dl_futures = {
            dl_pool.submit(_download_document, doc, bucket, download_path, s3_client, transfer_config): doc
            for doc in supported
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
            # Carry the name alongside the path: ``_download_document`` normalizes
            # the key before building the local path, so the path cannot be mapped
            # back to its descriptor afterwards without duplicating that logic.
            downloaded.append((local_path, key))

    downloaded.sort(key=lambda item: item[0].stat().st_size, reverse=True)
    return downloaded, download_errors


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
    available = float(os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        try:
            available = min(available, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    quota = _container_cpu_quota()
    if quota is not None:
        available = min(available, quota)
    return min(max(1, int(available // 2)), 8)
