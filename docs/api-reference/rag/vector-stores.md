# Vector Stores API

ai4rag talks to vector databases through **direct clients** selected by a typed
configuration object. A [config](#configuration) carries the connection details
for a single backend, and [`get_vector_store`](#store-selection) instantiates the
matching store — the backend is chosen entirely from `config.provider`, so no
separate type string is needed. Three backends are supported today:

| Backend | Config | Provider | Store | Hybrid search |
|---------|--------|----------|-------|---------------|
| Milvus (remote server only) | `MilvusConfig` | `"milvus"` | `MilvusVectorStore` | ✅ server-side dense + BM25 |
| Milvus Lite (embedded, local file only) | `MilvusLiteConfig` | `"milvus_lite"` | `MilvusVectorStore` | ✅ embedded dense + BM25 |
| PostgreSQL + pgvector | `PGVectorConfig` | `"pgvector"` | `PGVectorStore` | ✅ dense + full-text |

`MilvusConfig` and `MilvusLiteConfig` both construct a `MilvusVectorStore`, but they are separate,
mutually-exclusive config classes rather than two modes of one config:

!!! note "Why `MilvusConfig` and `MilvusLiteConfig` are separate — and why that matters"
    `MilvusConfig.uri` is validated to be an `http(s)://` URL and **raises `ValueError`** for anything else
    (a bare host, a local file path, an empty string). This is a deliberate safety fix: previously, a
    mistyped or unreachable `MILVUS_URI` could be silently interpreted as a local file path, creating an
    unintended local Milvus Lite database instead of failing — dangerous in production, where it could mask a
    misconfigured deployment. That silent fallback is no longer possible: a bad `MILVUS_URI` now fails loudly
    at construction time.

    To use the embedded engine, opt in explicitly with **`MilvusLiteConfig(db_path="./ai4rag.db")`** (or
    `MilvusLiteConfig()` for the default path, `DEFAULT_MILVUS_LITE_DB_PATH` = `"./ai4rag_milvus_lite.db"`).
    `MilvusLiteConfig` validates the inverse — it rejects `http(s)://` values in `db_path`, since those belong
    in `MilvusConfig`.

!!! warning "Milvus Lite limitations"
    Milvus Lite is intended for local development, tests, and small-scale workloads, not production. It
    computes BM25 statistics segment-locally rather than corpus-wide, so hybrid-search ranking fidelity (and
    any benchmark/HPO scores measured against it) may not transfer exactly to a production Milvus server; and
    it serializes writes, so only one process should open a given `.db` file at a time. For production or
    large corpora, use a remote Milvus server (`MilvusConfig`), Zilliz Cloud, or pgvector.

Every config is a frozen dataclass exposing a `from_env()` classmethod, so
connection details (and secrets) can be sourced from environment variables and
never embedded in generated artefacts.

## Base Vector Store

::: ai4rag.rag.vector_store.base_vector_store
    options:
      show_root_heading: true
      show_source: true

## Configuration

::: ai4rag.rag.vector_store.config
    options:
      show_root_heading: true
      show_source: true

## Store Selection

::: ai4rag.rag.vector_store.get_vector_store
    options:
      show_root_heading: true
      show_source: true

## Milvus

::: ai4rag.rag.vector_store.milvus
    options:
      show_root_heading: true
      show_source: true

## PGVector

::: ai4rag.rag.vector_store.pgvector
    options:
      show_root_heading: true
      show_source: true

## Hybrid Search Reranking

Milvus fuses dense and sparse results server-side, while PGVector combines dense
similarity with PostgreSQL full-text search in memory using the reranker below.

::: ai4rag.rag.vector_store.reranker
    options:
      show_root_heading: true
      show_source: true

## Collection Naming & Search Utilities

Collections follow the `ai4rag_<timestamp>_<suffix>` convention and are capped at
63 characters. Pass an existing name via `collection_name` to reuse a collection.

::: ai4rag.rag.vector_store.utils
    options:
      show_root_heading: true
      show_source: true
      members:
        - generate_collection_name
        - resolve_collection_name
        - sanitize_collection_name
        - validate_search_params
