# Changelog

All notable changes to ai4rag will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
