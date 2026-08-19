# Pipeline Components

The `ai4rag.components` package provides reusable building blocks for RAG pipeline workflows.
These functions encapsulate the business logic that was previously inlined in Kubeflow Pipeline components,
making it available for use in any context — KFP pipelines, standalone scripts, notebooks, or tests.

## Architecture

```
┌──────────────────────────────────────────────┐
│  pipelines-components  (KFP wrappers)        │
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
│  │ components/                               │ │
│  │  data/                                    │ │
│  │  optimization/                            │ │
│  │  assets_generator/                        │ │
│  │   notebook, leaderboard, templates        │ │
│  └───────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────┐ │
│  │ core/ — experiment, HPO, search space    │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

KFP wrappers handle only artifact I/O (reading `dsl.Input[Artifact]`, writing `dsl.Output[Artifact]`)
and Kubernetes-specific concerns (secrets, resource limits).  All business logic lives in `ai4rag`.

## Installation

S3 support (`boto3`), multiprocessing (`multiprocess`), and text extraction (`docling`) are all included in the core `ai4rag` install — no extra dependencies are needed to use pipeline components.

## Data Components

### Document Discovery

List and sample documents from an S3-compatible bucket:

```python
from ai4rag.components.data import discover_documents

result = discover_documents(
    bucket_name="my-bucket",
    prefix="documents/",
    sampling_enabled=True,
    sampling_max_size_gb=1.0,
)
print(f"Found {result.count} documents ({result.total_size_bytes} bytes)")
result.save("/tmp/discovery_output")
```

### Text Extraction

Download documents from S3 and extract text using Docling:

```python
from ai4rag.components.data import extract_text

result = extract_text(
    documents=[{"key": "docs/report.pdf", "size_bytes": 1024}],
    bucket="my-bucket",
    output_dir="/tmp/extracted",
    max_extraction_workers=4,
)
print(f"Processed {result.processed_count}/{result.total_documents}")
```

Supported document extensions include PDF, DOCX, PPTX, Markdown, HTML, TXT, ODT/ODP, AsciiDoc, LaTeX, EPUB, email, Quarto/R Markdown, XHTML, and images (JPEG, PNG, TIFF).

OCR is **off by default**. To enable RapidOCR (for scanned PDFs / images):

```python
result = extract_text(
    documents=[{"key": "scans/page.png", "size_bytes": 2048}],
    bucket="my-bucket",
    output_dir="/tmp/extracted",
    do_ocr=True,                 # RapidOCR via Docling
    ocr_lang="english",          # default when OCR is enabled
    # Optional custom ONNX models for disconnected / specialized deployments:
    # ocr_det_model_path="/models/det.onnx",
    # ocr_cls_model_path="/models/cls.onnx",
    # ocr_rec_model_path="/models/rec.onnx",
    # ocr_rec_keys_path="/models/keys.txt",
)
```

Default RapidOCR models are **not** in current PyPI `rapidocr` wheels. On AutoRAG/OpenShift images with `DOCLING_ARTIFACTS_PATH` set, bake ONNX models under `$DOCLING_ARTIFACTS_PATH/RapidOcr/` at image build time (see `tmp/Containerfile.autorag-dev`). Docling auto-detects pages that need OCR when `do_ocr=True`. Override with `ocr_*_model_path` for custom ONNX sets.

### Test Data Loading

Load benchmark test data from S3:

```python
from ai4rag.components.data import load_test_data

result = load_test_data(
    bucket_name="my-bucket",
    key="benchmarks/test_data.json",
    benchmark_sample_size=25,
)
print(f"Loaded {result.record_count} records (sampled: {result.sampled})")
```

## Optimization Components

### Search Space Preparation

Build a search space report with model pre-selection:

```python
from ai4rag.components.optimization import prepare_search_space_report

report = prepare_search_space_report(
    test_data_path="/tmp/test_data.json",
    extracted_text_path="/tmp/extracted/",
    maas_client=client,
    embedding_models=["bge-m3"],
    generation_models=["qwen3-8b-fp8-dynamic"],
    chunking_methods=["recursive"],   # optional: constrain chunking methods
    chunk_sizes=[256, 512, 1024],     # optional: constrain chunk sizes
    chunk_overlaps=[0, 128],          # optional: constrain chunk overlaps
)
report.save_yaml("/tmp/search_space.yaml")
```

### RAG Optimization

Run a full optimization experiment:

```python
from ai4rag.components.optimization import run_rag_optimization
from ai4rag.rag.vector_store import MilvusConfig

result = run_rag_optimization(
    extracted_text_path="/tmp/extracted/",
    test_data_path="/tmp/test_data.json",
    search_space_report_path="/tmp/search_space.yaml",
    output_dir="/tmp/rag_patterns/",
    maas_client=client,
    vector_store_config=MilvusConfig.from_env(),
    test_data_key="benchmarks/test_data.json",
    input_data_key="documents/",
)
print(f"Generated {len(result.patterns)} patterns")
```

## Shared Utilities

The `ai4rag.components` package provides three shared utility modules used across components:

| Module | Function | Purpose |
|--------|----------|---------|
| `utils.s3` | `create_s3_client()` | S3 client factory with env-var fallback |
| `utils.maas_client` | `create_maas_client()` | Single MaaS client (endpoint from `MAAS_BASE_URL`, used verbatim) for listing, chat, and embeddings, with SSL self-signed cert fallback |
| `utils.docling_io` | `load_docling_documents()` | Load DoclingDocument JSON files |

These are importable from `ai4rag.components` or `ai4rag.components.utils`:

```python
from ai4rag.components import create_s3_client, create_maas_client, load_docling_documents
```

!!! note "Single client for everything"
    `create_maas_client()` builds the one client MaaS needs: it lists available models
    (`models.list()`) and is reused, unchanged, to serve `chat.completions` and `embeddings`
    for every model wrapper. See [Provider-Agnostic Design](provider-agnostic.md) for the full pattern.

## Design Principles

- **No KFP types**: Functions accept plain Python types (`str`, `Path`, `dict`) and return frozen dataclasses.
- **Dependency injection**: All functions accept pre-configured clients (S3, MaaS) as optional parameters — when omitted, clients are created from environment variables.
- **Lazy imports**: Heavy optional dependencies (`boto3`, `multiprocess`, `docling`) are imported only when used.
- **SSL fallback**: S3 operations and the MaaS client automatically retry with `verify=False` when self-signed certificate errors are detected.
