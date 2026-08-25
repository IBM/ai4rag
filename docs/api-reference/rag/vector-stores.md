# Vector Stores API

ai4rag talks to vector databases through **direct clients** selected by a typed
configuration object. A [config](#configuration) carries the connection details
for a single backend, and [`get_vector_store`](#store-selection) instantiates the
matching store — the backend is chosen entirely from `config.provider`, so no
separate type string is needed. Three backends are supported today:

| Backend | Config | Store | Hybrid search |
|---------|--------|-------|---------------|
| Chroma | `ChromaConfig` | `ChromaVectorStore` | ❌ vector only |
| Milvus | `MilvusConfig` | `MilvusVectorStore` | ✅ server-side dense + BM25 |
| PostgreSQL + pgvector | `PGVectorConfig` | `PGVectorStore` | ✅ dense + full-text |

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

## Chroma

::: ai4rag.rag.vector_store.chroma
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
