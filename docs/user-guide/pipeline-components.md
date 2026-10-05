# Pipeline Components

`ai4rag.utils.data` and `ai4rag.assets_generator` provide reusable building blocks for RAG
pipeline workflows. These functions encapsulate the business logic that was previously inlined in
Kubeflow Pipeline components, making it available for use in any context — KFP pipelines, standalone
scripts, notebooks, or tests.

## Architecture

```
┌──────────────────────────────────────────────┐
│  pipelines-components  (KFP wrappers +       │
│  RAG optimization orchestration)             │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ @dsl.   │ │ @dsl.    │ │ @dsl.        │  │
│  │component│ │component │ │ component    │  │
│  └────┬────┘ └─────┬────┘ └──────┬───────┘  │
│       │            │             │           │
└───────┼────────────┼─────────────┼───────────┘
        │            │             │
┌───────▼────────────▼─────────────▼───────────┐
│  ai4rag  (business logic)                    │
│  ┌─────────────┐ ┌─────────────────────────┐ │
│  │ utils/                                    │ │
│  │  data/                                    │ │
│  │  clients/ — s3, maas_client               │ │
│  │ assets_generator/                         │ │
│  │  notebook, leaderboard, templates         │ │
│  └───────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────┐ │
│  │ core/ — experiment, HPO                  │ │
│  │ search_space/ — search space, prepare    │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

KFP wrappers handle artifact I/O (reading `dsl.Input[Artifact]`, writing `dsl.Output[Artifact]`),
Kubernetes-specific concerns (secrets, resource limits), and full RAG-optimization orchestration.
Data-stage and asset-generation business logic lives in `ai4rag`, so it can be exercised outside KFP.

## Installation

S3 support (`boto3`), multiprocessing (`multiprocess`), and text extraction for born-digital documents (`docling-slim[feat-chunking]`) are all included in the core `ai4rag` install. OCR (scanned PDFs/images) and audio transcription additionally require the `text-extraction` extra — see [Installation](../getting-started/installation.md#basic-installation).

## Private CA certificates on OpenShift AI

For disconnected clusters or private endpoints that use a self-signed or private CA,
configure the CA once through the OpenShift AI `DSCInitialization` (DSCI) object. Do
not disable TLS verification with `verify=False`.

1. Create or update a ConfigMap in the OpenShift AI applications namespace. The
   `ca-bundle.crt` value can contain one or more PEM certificates.

   ```sh
   oc -n redhat-ods-applications create configmap custom-ca-bundle \
     --from-file=ca-bundle.crt=/path/to/ca-bundle.crt \
     --dry-run=client -o yaml | oc apply -f -
   ```

2. Reference it from the DSCI object. Replace `default-dsci` if your cluster uses
   another DSCI resource name.

   ```sh
   oc patch dscinitialization default-dsci --type=merge \
     -p '{"spec":{"trustedCABundle":{"customCABundle":"custom-ca-bundle"}}}'
   ```

OpenShift AI automatically propagates the configured certificate into the managed
trust bundles used by workbenches and by other OpenShift AI services that consume the
custom CA bundle. Restart affected workbenches after an update. The DSCI setting does
not alter trust for arbitrary workloads that do not mount the OpenShift AI custom CA
bundle.

In a managed workbench, set `AWS_CA_BUNDLE` to the mounted bundle before constructing
the S3 client when it is not already set:

```python
import os

os.environ.setdefault("AWS_CA_BUNDLE", "/etc/pki/tls/custom-certs/ca-bundle.crt")
```

The MaaS client (`create_maas_client()`) honors the same pattern via `MAAS_CA_BUNDLE`
(or an explicit `ca_bundle=` argument); it never falls back to unverified TLS:

```python
os.environ.setdefault("MAAS_CA_BUNDLE", "/etc/pki/tls/custom-certs/ca-bundle.crt")
```

## Data Components

### Document Discovery

List and sample documents from one or more locations in an S3-compatible bucket:

```python
from ai4rag.utils.data import discover_documents

result = discover_documents(
    bucket_name="my-bucket",
    prefixes=["documents/", "manuals/"],
    sampling_enabled=True,
    sampling_max_size_gb=1.0,
)
print(f"Found {result.count} documents ({result.total_size_bytes} bytes)")
result.save("/tmp/discovery_output")
```

Every prefix is listed and merged into a single corpus deduplicated by object key, so overlapping
selections (`docs/` and `docs/manuals/`) are safe. The sampling budget applies to that union, not to
each location. Omitting `prefixes` — or passing an empty list — lists the whole bucket; a bare string
is accepted as a single location.

When `test_data_doc_names` is given, every entry must identify exactly one discovered document.
A key that matches nothing, or a bare file name shared by documents in two locations, raises
`BenchmarkKeyError` naming the offending keys rather than leaving the question silently ungrounded:

```python
from ai4rag.utils.data import BenchmarkKeyError, discover_documents

try:
    result = discover_documents(
        bucket_name="my-bucket",
        prefixes=["documents/"],
        test_data_doc_names=["documents/report.pdf"],
    )
except BenchmarkKeyError as exc:
    print(exc)  # names the keys and the locations that were searched
```

### Text Extraction

Download documents from S3 and extract text using Docling:

```python
from ai4rag.utils.data import extract_text

result = extract_text(
    documents=[{"key": "docs/report.pdf", "size_bytes": 1024}],
    bucket="my-bucket",
    output_dir="/tmp/extracted",
    max_extraction_workers=4,
)
print(f"Processed {result.processed_count}/{result.total_documents}")
```

By default `extract_text` builds its own S3 client from the `s3_*` arguments or the environment,
using `AWS_CA_BUNDLE` for TLS verification as described above. Pass `ssl_cert_path` to point at a
CA bundle for this call only (it takes precedence over `AWS_CA_BUNDLE`), or pass a pre-configured
`s3_client` to reuse one you already built (e.g. via `create_s3_client`) instead of having
`extract_text` construct its own:

```python
from ai4rag.utils.clients import create_s3_client
from ai4rag.utils.data import extract_text

s3_client = create_s3_client(verify="/etc/pki/tls/custom-certs/ca-bundle.crt")
result = extract_text(
    documents=[{"key": "docs/report.pdf", "size_bytes": 1024}],
    bucket="my-bucket",
    output_dir="/tmp/extracted",
    s3_client=s3_client,
)
```

Supported document extensions include PDF, DOCX, PPTX, Markdown, HTML, TXT, ODT/ODP, AsciiDoc, LaTeX, EPUB, email (`.eml`, `.msg`), Quarto/R Markdown, XHTML, images (JPEG, PNG, TIFF), and audio (WAV, MP3, M4A, AAC, OGG, FLAC).

OCR is **off by default**. Conversion behaviour (table structure and OCR) is
controlled through a single `DoclingExtractionConfig` instance. To enable
RapidOCR (for scanned PDFs / images):

```python
from ai4rag.utils.data import DoclingExtractionConfig, extract_text

result = extract_text(
    documents=[{"key": "scans/page.png", "size_bytes": 2048}],
    bucket="my-bucket",
    output_dir="/tmp/extracted",
    docling_config=DoclingExtractionConfig(
        do_ocr=True,  # RapidOCR via Docling
        ocr_lang="english",  # default when OCR is enabled
        # Optional custom ONNX models for disconnected / specialized deployments:
        # ocr_det_model_path="/models/det.onnx",
        # ocr_cls_model_path="/models/cls.onnx",
        # ocr_rec_model_path="/models/rec.onnx",
        # ocr_rec_keys_path="/models/keys.txt",
    ),
)
```

**Language handling.** There is no automatic language detection — `ocr_lang`
selects which bundled RapidOCR model set is loaded. Latin-script languages all
resolve to the English models; only Chinese switches to the dedicated Chinese
models. `ocr_lang` accepts a single string (`"english"`) or a sequence
(`["english", "chinese"]`) and defaults to `["english"]` when OCR is enabled.

Default RapidOCR models are **not** in current PyPI `rapidocr` wheels, and `extract_text` no longer falls back to Docling's runtime model downloader. Whenever `do_ocr=True`, `DOCLING_ARTIFACTS_PATH` must be set to a non-empty Docling artifacts directory — this is checked before anything else, **even when `ocr_*_model_path` is also set** — or `extract_text` raises `FileNotFoundError` immediately. Bake ONNX models under `$DOCLING_ARTIFACTS_PATH/RapidOcr/` at image build time (see `tmp/Containerfile.autorag-dev`); use `ocr_*_model_path` only to point at a different ONNX set once `DOCLING_ARTIFACTS_PATH` is set. Docling auto-detects pages that need OCR when `do_ocr=True`.

#### Audio Transcription

Audio files (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`) are transcribed automatically — no configuration flag is needed, unlike OCR. `extract_text` routes them through Docling's ASR pipeline using a Whisper (`whisper-tiny`) model, with the spoken language auto-detected per file:

```python
from ai4rag.utils.data import extract_text

result = extract_text(
    documents=[{"key": "recordings/call.mp3", "size_bytes": 4096}],
    bucket="my-bucket",
    output_dir="/tmp/extracted",
)
```

When using the hybrid/Docling chunker, chunks produced from ASR transcript
segments also carry `audio_start_seconds` and `audio_end_seconds` in their
vector metadata.  These are the earliest and latest Docling segment times in
the chunk, so consumers can link a retrieval result back to the corresponding
audio range.  They are not word-level seek positions.

### Test Data Loading

Load benchmark test data from S3:

```python
from ai4rag.utils.data import load_test_data

result = load_test_data(
    bucket_name="my-bucket",
    key="benchmarks/test_data.json",
    benchmark_sample_size=25,
)
print(f"Loaded {result.record_count} records (sampled: {result.sampled})")
```

## Search Space Preparation

Build and validate a search space, then serialize it to a report. Model
pre-selection is a separate step (see `ModelsPreSelector`); the report written
here is the full search space:

```python
from ai4rag.search_space.prepare import (
    build_search_space_report,
    prepare_search_space_with_maas,
)

search_space = prepare_search_space_with_maas(
    payload={
        "foundation_models": [{"model_id": "qwen3-8b-fp8-dynamic"}],
        "embedding_models": [{"model_id": "bge-m3"}],
        "chunking_methods": ["recursive"],  # optional: constrain chunking methods
        "chunk_sizes": [256, 512, 1024],    # optional: constrain chunk sizes
        "chunk_overlaps": [0, 128],         # optional: constrain chunk overlaps
    },
    client=client,
    benchmark_data=benchmark_df,  # optional: used for language detection
    vector_store_type="milvus",   # optional: "milvus", "milvus_lite", "pgvector", or "neo4j"
)
build_search_space_report(search_space).save_json("/tmp/search_space.json")
```

!!! note "Neo4j search space defaults"
    Passing `vector_store_type="neo4j"` fixes `chunk_size`/`chunk_overlap` and `search_mode="graph"`, and
    omits the hybrid ranker parameters, since the Neo4j backend supports graph retrieval only. See
    [Search Space defaults](../user-guide/search-space.md#default-parameters).

!!! note "RAG optimization orchestration"
    Running a full optimization experiment (`run_rag_optimization`) — wiring search-space
    preparation, indexing, retrieval, generation, and evaluation together into RAG
    patterns — is a KFP pipeline concern and lives in the `pipelines-components` repo, on
    top of the `ai4rag.core` and `ai4rag.search_space` APIs.

## Shared Utilities

`ai4rag.utils.clients` provides thin, dependency-injectable adapters over the external
services pipeline steps talk to, and `ai4rag.utils.docling_io` loads persisted
`DoclingDocument` JSON:

| Module | Function | Purpose |
|--------|----------|---------|
| `ai4rag.utils.clients.s3` | `create_s3_client()` | S3 client factory with environment-based credentials and private-CA support via `AWS_CA_BUNDLE` |
| `ai4rag.utils.clients.maas_client` | `create_maas_client()` | Single MaaS client (endpoint from `MAAS_BASE_URL`, normalized to a `/v1`-suffixed URL) for listing, chat, and embeddings, with private-CA support via `MAAS_CA_BUNDLE` |
| `ai4rag.utils.docling_io` | `load_docling_documents()` | Load DoclingDocument JSON files |

`create_s3_client()` and `create_maas_client()` are also re-exported from the `ai4rag.utils.clients`
package for convenience:

```python
from ai4rag.utils.clients import create_maas_client, create_s3_client
from ai4rag.utils.docling_io import load_docling_documents
```

!!! note "Single client for everything"
    `create_maas_client()` builds the one client MaaS needs: it lists available models
    (`models.list()`) and is reused, unchanged, to serve `chat.completions` and `embeddings`
    for every model wrapper. See [Provider-Agnostic Design](provider-agnostic.md) for the full pattern.

## Design Principles

- **No KFP types**: Functions accept plain Python types (`str`, `Path`, `dict`) and return frozen dataclasses.
- **Dependency injection**: All functions accept pre-configured clients (S3, MaaS) as optional parameters — when omitted, clients are created from environment variables.
- **Lazy imports**: Heavy optional dependencies (`boto3`, `multiprocess`, `docling`) are imported only when used.
- **TLS verification**: Configure private CAs through the platform trust bundle and `AWS_CA_BUNDLE`; do not disable certificate verification.
