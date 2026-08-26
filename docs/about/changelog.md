# Changelog

All notable changes to ai4rag will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Text extraction** — optional RapidOCR via Docling, configured through a single `DoclingExtractionConfig` passed to `extract_text` (`do_ocr`, `ocr_lang`, custom ONNX model paths); OCR remains off by default
- **Document discovery / extraction** — JPEG, PNG, and TIFF image extensions supported for OCR-capable ingestion
- **Text extraction** — audio ingestion (`.wav`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.flac`) transcribed via Docling's ASR pipeline (Whisper), with automatic language detection

### Fixed
- **Text extraction / OCR** — fail fast with bake instructions when `DOCLING_ARTIFACTS_PATH` is set but RapidOCR ONNX models are missing; current PyPI `rapidocr` wheels no longer ship model files (bake via `tmp/Containerfile.autorag-dev`)

---

## [0.12.0](https://github.com/IBM/ai4rag/releases/tag/v0.12.0)

### Added
- **Vector store** — direct backend clients for Chroma, Milvus, and PostgreSQL/pgvector (`ChromaVectorStore`, `MilvusVectorStore`, `PGVectorStore`), each selected via a typed, frozen config dataclass (`ChromaConfig`, `MilvusConfig`, `PGVectorConfig`) passed as a single `vector_store_config`
- **Vector store** — `reranker` module implementing RRF and weighted fusion for hybrid search
- **Vector store** — `BaseVectorStore` now supports `close()` and the context-manager protocol; each optimization trial scopes its store in a `with` block, so connections and pools are no longer leaked per trial
- **Vector store (pgvector)** — connection pooling via `psycopg_pool.ConnectionPool` with a configurable `PGVectorConfig.pool_max_size` (default 10); `AI4RAGExperiment` sizes the pool from `inference_max_threads` so it tracks real query concurrency
- **Evaluator** — optional `RagasEvaluator` (with RAGAS adapter classes `AI4RAGRagasLLM` / `AI4RAGRagasEmbeddings`) enabling RAGAS-based metrics; `ragas` is now a regular dependency
- **RAG optimization component** — `llm_judge_mode` selector (`base` / `ragas` / `all` / `none`) on `run_rag_optimization()` to choose which LLM-as-a-Judge evaluators run
- **Evaluator** — `build_aggregate_metric()` shared helper on `BaseEvaluator` for constructing aggregate metric payloads
- **Search space preparation** — `build_search_space_report()` and `serialize_model()` in `ai4rag.search_space.prepare`, co-locating the model↔spec round-trip (`serialize_model()` is the write mirror of the model restore path)
- **Model access** — `create_maas_client()` and shared model discovery/restore helpers `get_foundation_models()` / `get_embedding_models()` in `ai4rag.search_space.prepare.models`, accepting either bare model ids (discovery) or serialized report specs (restore)
- **Assets generator** — `get_vector_store_config()` / `get_vector_store_env_vars()` factories that build a backend config from a provider discriminator and expose each backend's required environment variables for documentation
- **Dependencies** — added `openai` as the model-access SDK (replacing `ogx-client`), plus `chromadb`, `pymilvus`, `pgvector`, and `psycopg[binary,pool]` for the direct vector-store clients

### Changed
- **Model provider** — replaced the OGX integration with any OpenAI-compatible endpoint; the shipped integration targets OpenShift AI Models-as-a-Service (MaaS), which serves listing, chat, and embeddings from a single endpoint
- **Vector store** — `get_vector_store()` and `AI4RAGExperiment` now take a single `vector_store_config` and dispatch on `config.provider`, replacing the `vector_store_type` string plus the OGX `vector_io` provider id
- **Vector store** — collection-name resolution centralized in `BaseVectorStore`, enforcing a mandatory `ai4rag` prefix as the cross-backend isolation guard
- **Vector store** — hybrid-search reranking parameter renamed `impact_factor` → `k`
- **Search space** — default `vector_store_type` changed from `ogx` to `milvus`; the default Chroma search space no longer includes the `window` retrieval method
- **Search space preparation** — renamed `prepare_search_space_with_ogx` to `prepare_search_space_with_maas`, now accepting an `openai.OpenAI` client. Because MaaS `models.list()` carries no metadata (model type, embedding dimension, context length), the payload must declare foundation and embedding model IDs explicitly; embedding dimension and context length are auto-detected at construction time
- **Model ids** — model ids are used verbatim, exactly as `models.list()` reports them (including any `/` characters); there is no more model-specific URL derivation or id stripping
- **Client factory** — replaced `create_ogx_client` with `create_maas_client`, a single client that serves listing, chat, and embeddings for every model at the one MaaS endpoint
- **Notebook templates** — renamed the generated `ogx_{indexing,inference}` templates to `maas_{indexing,inference}`, each building a single `OpenAI` client from `MAAS_BASE_URL` / `MAAS_API_KEY` and reusing it for every model; the inference notebook now also rebuilds the pattern's detected generation language and passes it to `OpenAIFoundationModel`, so answers keep the benchmark's language
- **Experiment / evaluator** — `metrics` and `optimization_metric` now require `RAGMetric` instances selected from `Metrics` and reject bare metric-name strings, which are ambiguous now that a name (e.g. `faithfulness`) is shared across the unitxt and RAGAS evaluators
- **Model helpers** — model-instantiation helpers moved to `ai4rag.search_space.prepare.models`, removing the components↔search_space coupling
- **Search space report** — model pre-selection decoupled from report building into an explicit `ModelsPreSelector` step; `SearchSpaceReport` slimmed to the search-space dict and no longer carries `selected_models` or a per-model `base_url`, and `pattern.json` no longer carries `base_url`
- **Leaderboard** — aggregate scores are keyed by a collision-free key (unitxt and custom metrics keep their bare name; other evaluators are prefixed, e.g. `ragas_faithfulness`), so colliding metric names each get their own column instead of overwriting one another

### Fixed
- **Vector store (Milvus)** — forced `consistency_level="Strong"` on vector/hybrid search so a query immediately following an `add_documents()` upsert can no longer race Milvus's default bounded-staleness read and return zero hits against a collection that does contain matching data
- **Vector store (pgvector)** — corrected `inner_product` scoring: the `<#>` operator returns the negative inner product, so the score is now derived by negation (cosine/l2/l1 keep `1/dist`), fixing an inverted ranking
- **Vector store (pgvector)** — guarded lazy index creation with double-checked locking (plus a `UniqueViolation` fallback) so concurrent search threads no longer race on `CREATE INDEX`
- **Experiment** — an optimization metric that is produced but unscored (`None` mean) is now recorded as a failed — not fatal — iteration; a genuinely absent metric still raises a `RAGExperimentError` with an evaluator-qualified message
- **Components** — added `vector_db_secret_name` to the indexing pipeline params
- **Core** — `ensure_ascii=False` when JSON-dumping documents that may reach the end user, preserving non-ASCII characters
- **Benchmark data** — reject `BenchmarkData` records with zero correct answers, preventing a downstream unitxt `TokenOverlap` crash on `max()` of an empty iterable
- **Experiment** — benchmark JSON is now read with an explicit UTF-8 encoding

### Removed
- **OGX** — removed all OGX support: the `ogx-client` dependency, `OGXFoundationModel`, `OGXEmbeddingModel`, `OGXVectorStore`, `OGXModelParameters`, `OGXEmbeddingParams`, `create_ogx_client`, the `ogx_utils` module, the `ogx_inference_base_url` helper, and the `OGX_CLIENT_BASE_URL` / `OGX_CLIENT_API_KEY` environment variables (replaced by `MAAS_BASE_URL` / `MAAS_API_KEY`)
- **Assets generator** — removed the OGX-only `pattern_builder` and `prompt_filters` modules and the `build_pattern_json` export; indexing-spec enrichment is now inlined
- **Search space preparation** — `prepare_search_space_report()` and the `search_space_preparation` module removed from `ai4rag.components.optimization`; build a search space with `prepare_search_space_with_maas()`, then call `build_search_space_report()` from `ai4rag.search_space.prepare`
- **Experiment** — `EvaluationResult` no longer carries a `rag_pattern` field; a trial's vector store is closed once the trial finishes, so read `pattern_name` / `scores` from `EvaluationResult` instead of calling `.generate()` on a previously returned pattern
- **Dependencies** — removed `langchain-chroma`; Chroma is now used directly via `chromadb`
- **Samples** — removed the outdated `samples/run_ai4rag.ipynb` notebook

---

## [0.11.0](https://github.com/IBM/ai4rag/releases/tag/v0.11.0)

### Added
- **Text extraction** — added support for 9 additional document formats (`.odt`, `.odp`, `.adoc`, `.tex`, `.epub`, `.eml`, `.qmd`, `.rmd`, `.xhtml`) in document discovery and text extraction, alongside existing PDF, DOCX, PPTX, Markdown, HTML, and plain-text support

### Changed
- **Data component** — `SUPPORTED_EXTENSIONS` extracted into a shared `ai4rag.components.data.constants` module, removing duplication between document discovery and text extraction
- **Dependencies** — replaced the `docling` meta-package with `docling-slim[standard,feat-chunking,format-opendocument]`, and dropped the standalone `docling-core` dependency, now pulled in transitively via the `feat-chunking` extra

### Fixed
- **Notebooks** — updated the `ogx_inference_template.ipynb` test-data-loading example to call `ai4rag.components.data.test_data_loader.load_test_data()`, replacing a stale reference to the removed `kfp_components` pipeline API

---

## [0.10.4](https://github.com/IBM/ai4rag/releases/tag/v0.10.4)

### Fixed
- **Experiment** — fixed incorrect dictionary key `"method"` used to check the chunking method when determining whether to include metadata; now uses the canonical `AI4RAGParamNames.CHUNKING_METHOD` constant, ensuring `include_metadata` is correctly set for hybrid chunking during experiment execution

---

## [0.10.3](https://github.com/IBM/ai4rag/releases/tag/v0.10.3)

### Fixed
- **RAG optimization component** — `GAMOptimizer` initial random phase now uses stratified sampling to guarantee that every unique value of each string-valued categorical parameter (e.g. `search_mode`, `chunking_method`) is evaluated at least once before GAM training begins, preventing biased exploration when the search space is skewed toward a dominant category; warm-start observations are accounted for so stratification does not waste early slots on already-covered values; a warning is emitted when `n_random_nodes` is too small for full categorical coverage

---

## [0.10.2](https://github.com/IBM/ai4rag/releases/tag/v0.10.2)

### Changed
- **Dependencies** — updated `ogx-client` dependency from `~=1.1.0` to `~=1.2.0`
- **Data component** — `ChunkingConstraints.METHODS` changed from mutable list to immutable tuple for correctness

### Fixed
- **Evaluator** — hardened LLM-as-a-Judge JSON response parsing with lightweight repair for common malformed outputs (single-quoted JSON, markdown-fenced blocks, JSON embedded in surrounding prose); added explicit output format instructions to the judge prompt; separated LLM call failures from JSON parse failures with distinct warning messages and raw response logging
- **Evaluator** — `calculate_overall_score()` now propagates `None` directly for unevaluated metrics instead of converting to `float("nan")`
- **Experiment** — streamed pattern now includes the `include_metadata` chunking field, ensuring metadata-aware chunking configurations are fully captured in pattern output

### Removed
- **Data component** — removed `index_documents()` function and `documents_indexing` module from `ai4rag.components.data`; the component was unused

---

## [0.10.1](https://github.com/IBM/ai4rag/releases/tag/v0.10.1)

### Added
- **OGX client** — timeout fallback for embedding and chat requests: on `APITimeoutError`, retries once with a 20-minute timeout and disabled client-level retries to accommodate slow CPU-deployed models

### Changed
- **Chunking** — `AI4RAGChunk` now carries a deterministic `chunk_id` field (SHA-256 of document ID, sequence number, and text), replacing ad-hoc hash-based ID generation in vector stores
- **Chunking** — hybrid chunking method now automatically includes document metadata during experiment execution
- **Vector store** — `ChromaVectorStore` and `OGXVectorStore` deduplication and chunk identification now use the deterministic `AI4RAGChunk.chunk_id` instead of independent hash computations

### Fixed
- **Prompt templates** — partially reverted default RAG prompt templates for all model families (Granite, Llama, Mistral, OpenAI, default) to use model-native prompting patterns, removing shared instruction boilerplate
- **Prompt filters** — decoupled `HPO_CITATION_FRAGMENTS` from internal `_RAG_CITATION_INSTRUCTION` constant, using inline string literals for portability

---

## [0.10.0](https://github.com/IBM/ai4rag/releases/tag/v0.10.0)

### Added
- **Evaluator** — `LLMaJEvaluator` for LLM-as-a-Judge evaluation, scoring `answer_relevance` on a 1–5 rubric with structured JSON output and bootstrap confidence intervals; scores are normalized to [0.0, 1.0]
- **Evaluator** — automatic judge model selection via `select_judge_model()` — when multiple generation models are available, a calibration round scores each candidate on a benchmark subset and picks the one with the highest spread-and-stability score
- **Evaluator** — `RAGMetric` frozen dataclass and `Metrics` registry replacing raw metric-name strings throughout the evaluator and experiment APIs
- **Evaluator** — `custom_metrics` module with `calculate_overall_score()` — computes a cross-metric mean as a built-in custom metric (`overall_score`)
- **Experiment** — multi-evaluator dispatch: `AI4RAGExperiment` now accepts an `evaluators` list and routes each metric to the evaluator matching its `EVALUATOR_TYPE`
- **Experiment** — `metrics` parameter on `AI4RAGExperiment` for explicit control over which metrics are evaluated; defaults are derived from configured evaluators when omitted
- **RAG optimization component** — `indexing_pipeline_params` parameter on `run_rag_optimization()` for enriching `pattern.json` with indexing pipeline settings
- **OGX client utilities** — `ogx_inference_base_url()` helper for building `/v1`-suffixed inference endpoint URLs

### Changed
- **Evaluator** — `BaseEvaluator.evaluate_metrics()` signature now accepts `Sequence[RAGMetric]` and returns a structured `EvaluationMetricsResult` TypedDict (was `list[str]` → `dict`)
- **Evaluator** — `UnitxtEvaluator` updated to work with the new `RAGMetric`-based metric dispatch and return `EvaluationMetricsResult`
- **Event handler** — `BaseEventHandler.on_pattern_creation()` payload and evaluation results now fully typed via `PatternPayload` and `EvaluationRecord` TypedDicts with nested structured types (`AggregateMetricPayload`, `VectorStoreSettings`, `ChunkingSettings`, `RetrievalSettings`, `GenerationSettings`)
- **Experiment** — `optimization_metric` parameter accepts `RAGMetric | str` (was `str` only); default changed from `faithfulness` to `overall_score`
- **Experiment** — evaluation results internally use structured `EvaluationMetricsResult` throughout the scoring, streaming, and caching pipeline
- **RAG optimization component** — default optimization metric changed from `faithfulness` to `overall_score`; supported metrics now include `overall_score`
- **RAG optimization component** — judge model selection and LLM-as-a-Judge evaluation are now automatically enabled during `run_rag_optimization()`
- **RAG optimization component** — artefact generation extracted into `_generate_output_artifacts()` for clearer separation of concerns
- **Dependencies** — refreshed `uv.lock` and sorted dependency declarations in `pyproject.toml`

---

## [0.9.3](https://github.com/IBM/ai4rag/releases/tag/v0.9.3)

### Added
- **Search space preparation** — `chunk_overlaps` parameter on `prepare_search_space_report()` and `prepare_search_space_with_ogx()` for constraining the chunk-overlap dimension of the search space (e.g. `[0, 128]`), with Pydantic range validation against `ChunkingConstraints` bounds

### Changed
- **RAG optimization component** — renamed `max_threads` keyword argument to `inference_max_threads` on `run_rag_optimization()` to align with `prepare_search_space_report()` naming
- **Search space report** — verbose representation in `prepare_search_space_report()` now derives values from valid (rule-filtered) combinations instead of raw parameter lists, ensuring the report reflects only reachable configurations
- **Search space constraint validation** — removed automatic deduplication of `chunking_methods` and `chunk_sizes` from `AI4RAGConstraints` validators — duplicates are no longer silently collapsed
- **Dependencies** — removed the `langchain` meta-package; only the needed sub-packages (`langchain-chroma`, `langchain-text-splitters`) are retained

---

## [0.9.2](https://github.com/IBM/ai4rag/releases/tag/v0.9.2)

### Added
- `chunk_sizes` parameter on `prepare_search_space_report()` for constraining the chunk-size dimension of the search space (e.g. `[256, 512]`)

### Changed
- Chunking constraint validation (`chunking_methods`, `chunk_sizes`) now happens via Pydantic before any I/O, providing clearer error messages with exact field paths
- Minimum allowed chunk size lowered from 512 to 128, enabling finer-grained chunking strategies
- Duplicate values in `chunking_methods` and `chunk_sizes` are now automatically deduplicated

---

## [0.9.1](https://github.com/IBM/ai4rag/releases/tag/v0.9.1)

### Fixed
- LLM-based language detection now uses JSON-schema structured output (`response_format`) instead of fragile regex extraction, ensuring reliable ISO 639-1 code parsing from model responses

---

## [0.9.0](https://github.com/IBM/ai4rag/releases/tag/v0.9.0)

### Added
- Multilingual support — `Language` dataclass and `language` parameter on `BaseFoundationModel` for language-aware prompt template generation
- `ai4rag.search_space.prepare.language_detection` module for LLM-based benchmark language detection with ISO 639-1 mapping
- `CharApproxTokenizer` — lightweight, model-agnostic tokenizer approximating token count via character ratio, replacing the `tiktoken` dependency
- `ai4rag.components.assets_generator.prompt_filters` module for filtering OGX runtime injection duplicates from HPO prompt templates during Responses API export
- Progressive chunk truncation in `OGXEmbeddingModel` — oversized chunks are truncated before embedding instead of failing
- `max_threads` parameter on `run_rag_optimization()` for controlling concurrent benchmark evaluation threads

### Changed
- **Breaking:** `BaseFoundationModel` constructor accepts a `language` parameter; `user_message_text` and `context_template_text` are now validated properties instead of `RAGPromptTemplateString` descriptors
- **Breaking:** `BaseFoundationModel.chat()` signature now accepts `**kwargs`
- **Breaking:** Search space report format changed from YAML to JSON — `search_space_preparation` and `rag_templates_optimization` no longer use `pyyaml`
- **Breaking:** `OGXEmbeddingModel._embed_text()` renamed to `_call_embedding_api()`
- Replaced `tiktoken` dependency with character-based token approximation across all chunkers
- Upgraded `docling` dependency to `2.107.0` and adapted to new API
- Prompt template system refactored — `RAGPromptTemplateString` descriptor replaced with setter-based validation via `validate_prompt_templates_placeholders()`
- Responses API payload aligned with previous chat/completion format
- Component functions (`text_extraction`, `search_space_preparation`, `rag_templates_optimization`) made more customizable with additional parameters
- Removed `mike` dependency and documentation versioning from CI/CD

### Fixed
- Chunks exceeding embedding model context length now truncated with progressive margins instead of causing API failures
- `random_state` parameter properly wired through `BaseOptimizer` to `GAMOptimizer` and `RandomOptimizer` for deterministic optimization runs
- Removed unnecessary `docling` install cell from indexing notebook template

### Removed
- `tiktoken` dependency — replaced by `CharApproxTokenizer`
- `RAGPromptTemplateString` descriptor class from `ai4rag.rag.foundation_models.utils`
- YAML serialization support for model instances in search space reports

---

## [0.8.1](https://github.com/IBM/ai4rag/releases/tag/v0.8.1)

### Changed
- Downgraded `docling-core` dependency from `~=2.84.0` to `~=2.83.0` to resolve compatibility issues

---

## [0.8.0](https://github.com/IBM/ai4rag/releases/tag/v0.8.0)

### Added
- `ai4rag.components` package — pipeline step business logic consolidated from `pipelines-components`, usable standalone or within KFP wrappers
    - `components.data`: `discover_documents()`, `extract_text()`, `index_documents()`, `load_test_data()`
    - `components.optimization`: `prepare_search_space_report()`, `run_rag_optimization()`, `detect_benchmark_language()`
    - Shared utilities: `create_s3_client()`, `create_ogx_client()`, `load_docling_documents()`
- `ai4rag.components.assets_generator` — notebook, leaderboard, and pattern artefact generation
    - `Notebook` / `NotebookCell` classes with `importlib.resources` template loading
    - `build_leaderboard_html()` for styled HTML leaderboard generation
    - `build_pattern_json()` for RAG pattern definition building
    - `generate_notebook_from_template()` for notebook rendering from templates
    - Bundled notebook and script templates as package data
- `ai4rag.utils.compat.ensure_sqlite3()` — centralized pysqlite3 patch for RHEL 9 / older sqlite
- `boto3` and `multiprocess` added as core dependencies
- `docling` promoted from dev-only to core dependency
- Pipeline Components user guide and API reference documentation

### Changed
- **Breaking:** Event handler `PatternPayload` schema restructured — `pattern_name` → `name`, `execution_time` → `duration_seconds`, `vector_store` → `vector_store_binding`, `datasource_type` → `provider_id`, `collection_name` → `vector_store_id`
- **Breaking:** Removed `schema_version` and `producer` fields from `PatternPayload`
- **Breaking:** Removed `distance_metric` from `EmbeddingSettings` and indexing params
- `VectorStoreSettings` and `EmbeddingSettings` TypedDicts now use `total=False` for optional fields
- `ogx-client` dependency updated from `~=1.0.0` to `~=1.1.0`
- `docling-core` dependency updated from `~=2.74.1` to `~=2.84.0`
- Hybrid search payload now conditionally includes `ranker_k` (only for `rrf` strategy) and `ranker_alpha` (only for `weighted` strategy)
- Vector store provider type is now resolved dynamically from the OGX server when available
- `max_combinations` from search space now included in pattern creation payload

### Removed
- `ai4rag.search_space.src.models` module (`FoundationModels` and `EmbeddingModels` enum classes) — model IDs are now plain strings throughout the codebase
- `EmbeddingModels.get_distance_metric()` utility — distance metric is no longer tracked

---

## [0.7.0](https://github.com/IBM/ai4rag/releases/tag/v0.7.0)

### Added
- `DoclingChunker` — structure-aware, token-aware chunker wrapping docling's `HybridChunker`, operating directly on `DoclingDocument` objects and preserving document hierarchy (headings, tables, figures) during chunking
- `AI4RAGChunk` — framework-agnostic chunk dataclass replacing langchain `Document` as the pipeline's canonical chunk representation
- `"hybrid"` chunking method in the default search space, enabling `DoclingChunker` alongside the existing `"recursive"` method
- Search space validation rule `_rule_chunk_overlap_for_chunking_method` enforcing chunker-specific overlap constraints (`hybrid` requires overlap = 0; `recursive` requires overlap > 0)
- Minimum context length validation for embedding models — models with `context_length` below 700 tokens are now rejected during initialization with a descriptive error

### Changed
- **Breaking:** `BaseChunker.split_documents()` now accepts `Sequence[DoclingDocument]` and returns `list[AI4RAGChunk]` (was `Sequence[Document]` → `list[Document]`)
- **Breaking:** `BaseVectorStore.add_documents()` now accepts `Sequence[AI4RAGChunk]` (was `Sequence[Document]`)
- **Breaking:** `BaseVectorStore.search()` now returns `list[AI4RAGChunk]` (was `list[dict]`)
- `LangChainChunker` updated to accept `DoclingDocument` input (converts to markdown internally) and return `AI4RAGChunk` output
- `ChromaVectorStore` and `OGXVectorStore` updated to work with `AI4RAGChunk`, with internal conversions handled transparently
- `OGXEmbeddingModel` embedding batch size reduced from 2048 to 1024
- Added `docling-core` as a project dependency

---

## [0.6.3](https://github.com/IBM/ai4rag/releases/tag/v0.6.3)

### Fixed
- Duplicate chunk IDs in `OGXVectorStore.add_documents()` now detected and skipped with a warning, preventing insertion failures when documents produce identical chunk hashes

---

## [0.6.2](https://github.com/IBM/ai4rag/releases/tag/v0.6.2)

### Added
- `VectorStoreInitializationError` exception for clearer diagnostics when vector store creation or retrieval fails

### Fixed
- Vector store initialization errors are now caught and wrapped with contextual information (embedding model ID, vector store provider) instead of propagating raw exceptions
- Simplified experiment error summary — removed redundant log-file reminder suffix from error messages

---

## [0.6.1](https://github.com/IBM/ai4rag/releases/tag/v0.6.1)

### Changed
- Upgraded `ogx-client` dependency from `~=0.8.0` to `~=1.0.0`
- Updated documentation to require OGX Server >= 1.0.0

---

## [0.6.0](https://github.com/IBM/ai4rag/releases/tag/v0.6.0)

### Added
- `uv` package manager support as an alternative to `pip` for dependency management and development workflows
- `AGENTS.md` file with AI agent guidelines for contributing to the project

### Changed
- Rebranded all Llama Stack integrations to OGX: `LSEmbeddingModel` → `OGXEmbeddingModel`, `LSFoundationModel` → `OGXFoundationModel`, `LSVectorStore` → `OGXVectorStore`, `prepare_search_space_with_llama_stack` → `prepare_search_space_with_ogx` (and all related classes, modules, and configuration keys)
- Replaced `llama-stack-client` dependency with `ogx-client`
- Improved logging during model selection and validation — clearer messages when models are filtered or skipped
- Updated CI/CD workflows to use `uv` for dependency installation and test execution
- Updated all documentation to reflect the Llama Stack → OGX rebranding

### Removed
- OpenAI model wrappers (`OpenAIEmbeddingModel`, `OpenAIFoundationModel`)
- `dev_utils/run_experiment_with_openai_models.py` example script
- Llama Stack example notebooks from `dev_utils/llama_stack_examples/`

---

## [0.5.5](https://github.com/IBM/ai4rag/releases/tag/v0.5.5)

### Changed
- Improved error messages when models registered in Llama Stack do not respond — errors now distinguish between "not registered" and "registered but not responding" models, with actionable guidance
- Improved pre-selector logging to show total model counts before selection and selected counts after
- Added logging of selected foundation and embedding models during search space preparation
- Removed `pydantic`-based payload validation overhead in `prepare_search_space_with_llama_stack` — replaced with direct dataclass instantiation
- Removed `validation_error_decoder` module and `pydantic` dependency from search space preparation

---

## [0.5.4](https://github.com/IBM/ai4rag/releases/tag/v0.5.4)

### Fixed
- Fixed chunk ID collisions in `LSVectorStore` — chunks from the same document no longer share the same `chunk_id`; IDs are now derived from chunk content via hashing
- Fixed `chunk_metadata` in `LSVectorStore` to only contain `document_id`, with full document metadata preserved in a separate `metadata` field

---

## [0.5.3](https://github.com/IBM/ai4rag/releases/tag/v0.5.3)

### Changed
- Bumped `llama-stack-client` dependency from `~=0.6.0` to `~=0.7.1`
- Updated documentation and installation instructions to require Llama Stack >= 0.7.0

---

## [0.5.2](https://github.com/IBM/ai4rag/releases/tag/v0.5.2)

### Changed
- Vector store type now supports any Llama Stack provider via the `ls_<provider_id>` pattern (e.g., `ls_milvus`, `ls_qdrant`), instead of only the hardcoded `ls_milvus`
- `vector_store_type` parameter on `AI4RAGExperiment` changed from `Literal["chroma", "ls_milvus"]` to `str` for flexibility

### Fixed
- Fixed `provider_id` extraction in `get_vector_store` — previously hardcoded to `"milvus"`, now correctly derived from the `ls_<provider_id>` vector store type

---

## [0.5.1](https://github.com/IBM/ai4rag/releases/tag/v0.5.1)

### Added
- Batch processing for Llama Stack embeddings (2048 chunk limit) and vector store document insertion, preventing failures with large document sets

### Changed
- Hybrid search re-enabled by default in the `ls_milvus` default search space (was disabled in 0.5.0 due to upstream instability)
- Default chunk sizes narrowed from `(512, 1024, 2048, 4096)` to `(1024, 2048)` and overlaps from `(128, 256, 512)` to `(128, 256)` for faster optimization
- Chroma vector store batch size simplified to a fixed default of 2048 instead of querying client internals

---

## [0.5.0](https://github.com/IBM/ai4rag/releases/tag/v0.5.0)

### Added
- `KFPEventHandler`: new event handler for Kubeflow Pipelines (KFP) integration, enabling experiment progress tracking inside KFP pipeline components
- `known_observations` parameter on `GAMOptimiser` and `AI4RAGExperiment`, allowing the optimizer to be pre-seeded with prior evaluation results so redundant evaluations are skipped
- `__hash__` method on `BaseFoundationModel` based on `model_id`
- New functional test suite under `tests/functional/` with end-to-end experiment coverage using mocked models

### Changed
- `GAMOptSettings`: removed lower-bound constraints on `n_random_nodes` and `max_evals`; both now accept `0`, which is required for KFP pipeline component usage
- Bumped `llama-stack-client` dependency from `~=0.5.0` to `~=0.6.0`
- Hybrid search disabled by default in the default search space due to upstream Llama Stack instability
- `BaseEventHandler` payload TypedDicts enriched with full structured types (`MetricCI`, `PatternScores`, `VectorStoreSettings`, `ChunkingSettings`, etc.)
- Tests reorganized into `tests/unit/` and `tests/functional/` subdirectories
- Documentation and development workflow guides updated

### Fixed
- Fixed crash when the `metadata` field is absent from the `models.list()` response returned by Llama Stack

---

## [0.4.2](https://github.com/IBM/ai4rag/releases/tag/v0.4.2)

### Added
- Search space validation rule `_rule_ranker_k_for_rrf_only` ensuring `ranker_k` is only used with `rrf` ranker strategy
- Vector store validation that `ranker_k` is only valid when `ranker_strategy='rrf'`

### Changed
- Removed `numpy` dependency from `UnitxtEvaluator`; replaced with pandas-native `DataFrame.mask()` and `pd.isna()`
- Default search space: added `4096` to default chunk sizes
- Default search space: simplified hybrid search defaults — removed `normalized` ranker strategy, reduced `ranker_k` values to `(0, 60)` and `ranker_alpha` values to `(1, 0.5)`

---

## [0.4.1](https://github.com/IBM/ai4rag/releases/tag/v0.4.1)

### Changed
- Updated hybrid search reranker API to match Llama Stack 0.5.x: `ranker` → `reranker_type`/`reranker_params`, `k` → `impact_factor` (for RRF strategy)
- `ranker_k` parameter is now only passed for `rrf` ranker strategy (previously passed for all strategies)
- Bumped `llama-stack-client` dependency from `~=0.4.2` to `~=0.5.0`
- Updated documentation and installation instructions to require Llama Stack >= 0.5.0

---

## [0.4.0](https://github.com/IBM/ai4rag/releases/tag/v0.4.0)

### Added
- Hybrid search support for `ls_milvus` vector store: new `search_mode` ("vector" or "hybrid"), `ranker_strategy` ("rrf", "weighted", "normalized"), `ranker_k`, and `ranker_alpha` parameters
- Search space validation rules for hybrid search consistency (`_rule_search_mode_ranker_consistency`, `_rule_ranker_alpha_for_weighted_only`)
- `AI4RAGSearchSpace` now accepts `vector_store_type` parameter to tailor default parameters and validation rules per vector store
- Default search space for `chroma` now includes `window` retrieval method and window sizes (0, 1, 3, 5)
- Embedding params are now serialized and included in indexing params passed to the vector store
- `__hash__` method added to `BaseEmbeddingModel` based on `model_id`
- New documentation page for hybrid search (`docs/user-guide/hybrid-search.md`)

### Changed
- `LlamaStackRAG` renamed to `SimpleRAG` and moved from `llama_stack_rag_template.py` to `simple_rag_template.py` to reflect its provider-agnostic nature
- `Retriever` now accepts and forwards `search_mode`, `ranker_strategy`, `ranker_k`, and `ranker_alpha` to the vector store
- `LSVectorStore.search()` now accepts hybrid search parameters and validates their consistency
- Event stream payload restructured: `pattern_name`, `scores`, `execution_time`, `final_score`, `schema_version`, and `producer` are now top-level fields; `settings.retrieval` includes `search_mode` and ranker details for hybrid mode
- `get_default_ai4rag_search_space_parameters()` now accepts `vector_store_type` to control which parameters are included in the default search space

### Fixed
- Fixed incorrect logger call in `LocalEventHandler.on_pattern_creation` (missing format argument)
- Added `encoding="utf-8"` to file open calls in `LocalEventHandler`

---

## [0.3.0](https://github.com/IBM/ai4rag/releases/tag/v0.3.0)

### Added
- Auto-detection of embedding model `embedding_dimension` and `context_length` when not explicitly provided
- Model availability validation against the Llama Stack server during search space preparation
- Search space validation rule ensuring `chunk_size` respects embedding model context length
- New `prepare_search_space_with_llama_stack` utility for streamlined search space setup

### Changed
- Foundation model `chat()` API now accepts structured message list instead of separate system/user message strings
- Default search space expanded with additional `chunk_size` (512) and `chunk_overlap` (128) values
- Chunk size validation rule now requires `chunk_size > 2 * chunk_overlap`
- `LSEmbeddingParams` refactored from `TypedDict` to `@dataclass`

### Fixed
- Embedding model backwards compatibility in vector store for both legacy dict and new dataclass params

## [0.2.1](https://github.com/IBM/ai4rag/releases/tag/v0.2.1)

### Changed
- Default optimizer is now `GAMOptimizer`
- Default retrieval methods no longer contain `window` method, as this is not supported for `ls_milvus` at the moment
- `Parameter` no longer requires to specify `param_type`. `C` type is used as default

### Fixed
- Bug in `GAMOptimizer` that unabled its usage (failing during deepcopy)

## [0.2.0](https://github.com/IBM/ai4rag/releases/tag/v0.2.0)

### Added
- Support for `LocalEventHandler`
- Support for external models introduced via `OpenAI` client
- CI/CD tooling
- Added RAG pattern object streaming with and added it to results, so that pattern can be reused post experiment


### Fixed
- Documentation and `README.md` update
- Updated samples
- Updated docstrings

### Changed
- Loose required parameters for `AI4RAGExperiment`
- Change "Optimiser" to "Optimizer" in all references

## [0.1.0](https://github.com/IBM/ai4rag/releases/tag/v0.1.0)

### Added
- Initial working implementation of `ai4rag` that can be used with `llama-stack` for RAG Template optimization

---

### Version Numbering

ai4rag follows [Semantic Versioning](https://semver.org/):

- **Major.Minor.Patch** (e.g., 1.2.3)
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

---

## Release Process

Releases are created by maintainers by tagging a commit on `main`.

See [Development Workflow](../development/workflow.md#creating-a-release) for detailed release procedures.

---

## Stay Updated

- Watch the [GitHub repository](https://github.com/IBM/ai4rag) for releases
- Subscribe to release notifications
- Check the [releases page](https://github.com/IBM/ai4rag/releases) for version history
