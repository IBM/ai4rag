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

In DSP task pods, when no explicit `ca_bundle`, `MAAS_CA_BUNDLE`, or
`SSL_CERT_FILE` is configured, `create_maas_client()` also treats a plain
`ca.crt` under the existing `SSL_CERT_DIR` as a safe fallback and adds it to
certifi's public roots. If that mount is absent, HTTPX defaults are unchanged.
An invalid or unreadable mounted bundle raises; verification is never disabled.

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

PDFs are parsed with Docling's PDFium backend to recover text that the native parser
can emit as NUL characters. This applies to `extract_text()`; the development-only
`FileStore` helper also uses PDFium for local files.

#### Preparing extraction models

Call `prepare_extraction()` after discovery to select the models needed by the
corpus. Import it from `ai4rag.utils.data.text_extraction`; it is not re-exported
from `ai4rag.utils.data`.

```python
from ai4rag.utils.data.text_extraction import extract_text, prepare_extraction

documents = [{"key": "docs/report.pdf", "size_bytes": 1024}]
prepared = prepare_extraction(
    documents,
    language="en",
    disconnected=False,
)
result = extract_text(
    documents=documents,
    bucket="my-bucket",
    output_dir="/tmp/extracted",
    docling_config=prepared.config,
    docling_artifacts_path=prepared.artifacts_path,
    max_extraction_workers=1,
)
```

The returned `PreparedExtraction` exposes `config`, `artifacts_path`, `needs_ocr`,
and `needs_asr`. PDFs/images enable OCR, audio requires Whisper, and text-only
corpora need neither bundle. The `language` argument selects Chinese OCR for
`chinese`, `ch`, `zh`, `zho`, `chi`, `zh-cn`, or `zh-tw` (case-insensitive), and
English otherwise; it does not detect document language.

Connected preparation downloads pinned PP-OCRv4 files from
`superlinked/docling-artifacts` and a pinned `openai/whisper-tiny` snapshot only
when the corresponding local path is unset. Pass `docling_artifacts_path` or
`asr_model_path` to override `DOCLING_ARTIFACTS_PATH` or `HF_MODEL_DIR`, respectively.
Configured local paths are validated rather than replaced by downloads. With
`disconnected=True`, required local bundles must exist before preparation: OCR needs
the selected detection, classification, recognition, and matching dictionary files
under `RapidOcr/`; audio needs a complete Transformers-format Whisper directory.

Generated MaaS and knowledge-graph indexing notebooks use this workflow, select
`language` from the pattern, and use one extraction worker to reduce memory pressure.
They default to connected mode; set `RHOAI_NOTEBOOK_CLUSTER_MODE=disconnected` in a
workbench or `NOTEBOOK_CLUSTER_MODE=disconnected` as an override, including in AutoX
Jobs. Both variables accept `connected` or `disconnected`. Run All uses the chosen
mode without skipping cells. Direct Python callers set `disconnected` explicitly;
the helper does not read notebook mode variables.

Without a prepared configuration, OCR is **off by default**. Conversion behaviour (table structure and OCR) is
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

OCR model resolution first uses explicit `ocr_*_model_path` values, then ONNX models
bundled in a compatible RapidOCR wheel. If neither is available and
`DOCLING_ARTIFACTS_PATH` is set, ai4rag validates the language-specific models under
`$DOCLING_ARTIFACTS_PATH/RapidOcr/` before conversion. Missing offline models raise
`FileNotFoundError`. Without configured artifacts, model paths remain unset so a
connected environment can use Docling's normal model downloader. Explicit model paths
and bundled wheel models do not require `DOCLING_ARTIFACTS_PATH`.

For disconnected OCR runs, stage all required models before extraction. The generated
MaaS and knowledge-graph indexing notebooks use the pinned PP-OCRv4 bundles selected
by `prepare_extraction()`, rather than relying on bundled wheel models or Docling's
default downloader. See the [offline staging instructions](https://github.com/IBM/ai4rag#running-on-disconnected-clusters).
Docling auto-detects pages that need OCR when `do_ocr=True`.

#### Audio Transcription

Audio files (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`) are transcribed automatically
through `LocalHuggingFaceAsrPipeline`, a Docling ASR pipeline using a local Hugging Face
Transformers Whisper model. Set `HF_MODEL_DIR`, or override it with
`DoclingExtractionConfig.asr_model_path` as shown below. The model is loaded lazily on
the first audio input with `local_files_only=True`; non-audio conversion does not
require Whisper artifacts. `extract_text()` and the audio pipeline do not download
ASR weights. Connected `prepare_extraction()` can download the pinned Whisper Tiny
snapshot before extraction when no local model is configured.

Audio decoding requires the `ffmpeg` executable on `PATH`, with support for the
input codec. AI4RAG opens the local file directly with FFmpeg and passes mono
float32 samples at the model's sampling rate to Transformers. This supports M4A
files whose metadata is stored at the end, which require seeking during decoding.
Missing FFmpeg or a decoding failure raises `RuntimeError` with actionable details;
decoding an empty waveform raises `ValueError`.

If FFmpeg reports `Unsupported 'chnl' box`, the pipeline raises `RuntimeError`
explaining that the installed decoder cannot read the M4A file's version-1
channel-layout box. Convert the affected file to standard AAC-LC M4A, MP3, or WAV,
as recommended by the error.

The directory must contain `config.json` with `model_type="whisper"`,
`preprocessor_config.json`, `tokenizer_config.json`, model weights (or a sharded-weight
index), and the processor/tokenizer files needed by Transformers. Missing or incomplete
directories raise `FileNotFoundError`; invalid configurations or other model types
raise `ValueError`. Existing openai-whisper checkpoints need to be replaced with a
Transformers-format model.

```python
from ai4rag.utils.data import DoclingExtractionConfig, extract_text

result = extract_text(
    documents=[{"key": "recordings/call.mp3", "size_bytes": 4096}],
    bucket="my-bucket",
    output_dir="/tmp/extracted",
    docling_config=DoclingExtractionConfig(asr_model_path="/models/whisper-tiny"),
)
```

For an AutoRAG product deployment, `HF_MODEL_DIR` must be supplied by the approved Whisper modelcar.
For other deployments, mount an equivalent locally verified model directory and make it read-only.

ASR runs on CPU using Whisper's native timestamp-aware long-form generation rather
than fixed pipeline-level windows. Multilingual models explicitly transcribe the
source language with automatic language detection unless a language is configured
in the local model's generation configuration. `DoclingExtractionConfig.device`
controls document/OCR pipelines and does not change the audio backend's device.

Some Transformers versions emit a deprecation warning mentioning `suppress_tokens`
and `begin_suppress_tokens` from inside Whisper's generation implementation. AI4RAG
does not pass these overrides or disable the model's token-suppression rules.

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
        "template_ids": ["agentic_rag", "sequential_rag"],  # optional: template search dimension
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

Payload `chunk_sizes` accepts values from 128 through 1,024 tokens inclusive.
Values outside this range raise a Pydantic `ValidationError` before model
discovery. Omitting `chunk_sizes` uses the backend defaults: 512 and 1,024 for
vector backends, or 1,024 for Neo4j.

!!! note "Neo4j search space defaults"
    For Neo4j, set `vector_store_type="neo4j"` and omit `template_ids` (or use
    `["agentic_graph_rag"]`); the vector-template IDs in the example above are incompatible.
    Neo4j defaults use `chunk_size=1024`, `chunk_overlap` values `0` and `64`, and
    `search_mode="graph"`, omitting hybrid ranker parameters. Explicit chunking constraints
    in the payload still override those defaults. See
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
