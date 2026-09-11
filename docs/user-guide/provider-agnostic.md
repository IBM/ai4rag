# Provider-Agnostic Design

`ai4rag` is built on a provider-agnostic architecture that allows you to use any LLM provider, embedding model, and vector database.
This flexibility means you can optimize RAG configurations regardless of your infrastructure choices.

---

## Core Philosophy

Rather than locking you into a specific vendor or technology stack, `ai4rag` defines **abstract interfaces** for the three key components of a RAG system:

1. **Foundation Models** (LLMs for text generation)
2. **Embedding Models** (for document and query embeddings)
3. **Vector Stores** (for storing and retrieving document chunks)

Concrete implementations for different providers — an OpenAI-compatible endpoint (OpenShift MaaS out of the box, accessed through the OpenAI SDK) for foundation and embedding models; Milvus (remote server), Milvus Lite (embedded, local), and PGVector for vector stores — all adhere to these interfaces, making them **interchangeable** within the optimization framework.

---

## Supported Providers

For **models**, `ai4rag` speaks the OpenAI API: any OpenAI-compatible endpoint works — a hosted service, a self-managed server (vLLM, TGI, Ollama, …), or OpenShift MaaS (the integration shipped out of the box, detailed below). Not OpenAI-compatible? Implement `BaseFoundationModel` / `BaseEmbeddingModel` (see [Extending with Custom Providers](#extending-with-custom-providers)). For **vector stores**, pick from the built-in Milvus (remote server, via `MilvusConfig`), Milvus Lite (embedded, local, via `MilvusLiteConfig`), or PGVector backends, or add your own via `BaseVectorStore`.

### OpenShift MaaS Integration

**What it is**: [OpenShift AI Models-as-a-Service (MaaS)](https://www.redhat.com/en/products/ai) exposes all deployed models through a single OpenAI-compatible endpoint, so `ai4rag` talks to it with the stock [`openai`](https://github.com/openai/openai-python) SDK. It is one example of an OpenAI-compatible provider; the setup below applies to any of them — point the client at your endpoint's URL.

**What `ai4rag` supports**:

- **Foundation Models**: Any chat/completion model deployed on your MaaS instance
- **Embedding Models**: Any embedding model deployed on your MaaS instance

**How it works**: a single client, pointing at `MAAS_BASE_URL`, serves everything — it lists the available models (`models.list()`) and is reused, unchanged, to serve `chat.completions` and `embeddings` for every model. `MAAS_BASE_URL` may be host-only or already `/v1`-suffixed — `create_maas_client()` normalizes it either way. Model ids are used verbatim, exactly as `models.list()` reports them (ids may contain `/`).

!!! note "No model metadata"
    Unlike some registries, MaaS `models.list()` carries no metadata (model type, embedding dimension, context length). So embedding dimension and context length are auto-detected by `OpenAIEmbeddingModel` at construction time (or supplied via `params`), and the caller declares which model ids are foundation vs. embedding.

**Usage**:

```python
import os
from ai4rag.utils.clients.maas_client import create_maas_client
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel

# A single client serves everything: it lists available models and serves
# chat/completions and embeddings for all of them at the one MaaS endpoint.
maas_client = create_maas_client(
    base_url=os.getenv("MAAS_BASE_URL"),
    api_key=os.getenv("MAAS_API_KEY"),
)

# Model ids are used verbatim — exactly as models.list() reports them.
foundation_model = OpenAIFoundationModel(
    model_id="qwen3-8b-fp8-dynamic",
    client=maas_client,
)
embedding_model = OpenAIEmbeddingModel(
    model_id="bge-m3",
    client=maas_client,
    params={"embedding_dimension": 1024, "context_length": 8192},
)

# Vector store: chosen independently of the model clients via a typed config
from ai4rag.rag.vector_store import MilvusConfig

vector_store_config = MilvusConfig.from_env()  # remote Milvus server; reads MILVUS_URI (http(s)://), MILVUS_TOKEN, MILVUS_SERVER_CERT
```

!!! tip "Discovering models automatically"
    To validate model ids and build a full search space from a MaaS deployment in one call, use [`prepare_search_space_with_maas`](search-space.md), passing the `maas_client` and the foundation/embedding model ids per type.

---

### Milvus Lite (Embedded, Local File)

**What it is**: The embedded, zero-server counterpart to the Milvus backend, served by the same `MilvusVectorStore` implementation but configured through its own dedicated **`MilvusLiteConfig`** class (provider `"milvus_lite"`), distinct from `MilvusConfig`. It is the local, zero-setup option for development, testing, and small-scale deployments.

**What ai4rag supports**:

- **Vector Store**: Milvus Lite (via `MilvusLiteConfig(db_path="./ai4rag.db")`) for document storage and retrieval, including hybrid (dense + BM25) search

**Key advantage**: No external services required — data is persisted to the local file at `db_path` (default `"./ai4rag_milvus_lite.db"`, exported as `DEFAULT_MILVUS_LITE_DB_PATH`). Great for quick experimentation.

!!! note "Why a separate config class"
    `MilvusConfig` and `MilvusLiteConfig` are kept separate on purpose. `MilvusConfig` validates that `uri` is
    an `http(s)://` URL and raises `ValueError` otherwise, so a mistyped or unreachable `MILVUS_URI` fails
    loudly instead of silently being treated as a local file path and creating a throwaway local database.
    `MilvusLiteConfig` is the explicit opt-in for the embedded engine — it validates that `db_path` is *not*
    an `http(s)://` URL, rejecting that case in favor of `MilvusConfig`.

**Limitations**:

- **Not for production**: Suitable for local development and small-scale workloads (prototyping, up to roughly 1M vectors), not large-scale or production deployments — use a remote Milvus server (`MilvusConfig`), Zilliz Cloud, or pgvector instead
- **Lower hybrid-ranking fidelity**: BM25 statistics are computed segment-locally rather than corpus-wide, so hybrid search scores (and benchmark/HPO results measured against them) may not transfer exactly to a production Milvus server
- **Single writer**: Writes are serialized — only one process should open a given `.db` file at a time

**Usage**:

```python
# Can use with any foundation/embedding models
from ai4rag.rag.vector_store import MilvusLiteConfig
from ai4rag.utils.event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_config=MilvusLiteConfig(db_path="./ai4rag.db"),  # embedded Milvus Lite, backed by a local file
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="./output"),  # required
)
```

---

## How It Works: Abstract Base Classes

`ai4rag` uses **abstract base classes** to define the interface for each component.
Concrete implementations inherit from these bases and provide provider-specific logic.

### Foundation Models: `BaseFoundationModel`

**Interface**:

```python
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

class BaseFoundationModel:
    def __init__(self, client, model_id, params, ...):
        self.client = client
        self.model_id = model_id
        self.params = params

    @abstractmethod
    def chat(self, messages: list[MessageTyped]) -> list[MessageTyped]:
        """Generate text based on conversation history."""
```

**What implementations must provide**:

- `chat()`: Take a list of messages (role + content) and return the model's response

**Current implementations**:

- `OpenAIFoundationModel`: OpenShift MaaS (and any OpenAI-compatible API) integration

---

### Embedding Models: `BaseEmbeddingModel`

**Interface**:

```python
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

class BaseEmbeddingModel:
    def __init__(self, client, model_id, params):
        self.client = client
        self.model_id = model_id
        self.params = params

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
```

**What implementations must provide**:

- `embed_documents()`: Batch embed document chunks
- `embed_query()`: Embed a single query string

**Current implementations**:

- `OpenAIEmbeddingModel`: OpenShift MaaS (and any OpenAI-compatible API) integration

---

### Vector Stores: `BaseVectorStore`

**Interface**:

```python
from ai4rag.rag.vector_store import BaseVectorStore

class BaseVectorStore(ABC):
    def __init__(self, embedding_model, config, distance_metric, collection_name=None):
        self.embedding_model = embedding_model
        self._config = config
        self.distance_metric = distance_metric
        self._collection_name = resolve_collection_name(collection_name)

    @abstractmethod
    def search(self, query: str, k: int, **kwargs) -> list[AI4RAGChunk]:
        """Search for relevant documents."""

    @abstractmethod
    def add_documents(self, documents: Sequence[AI4RAGChunk]) -> None:
        """Add documents to the vector store."""

    @property
    def collection_name(self) -> str:
        """Return the collection/index name. Concrete on the base class — reused
        when `collection_name` is supplied to `__init__`, otherwise generated."""
        return self._collection_name
```

**What implementations must provide**:

- `search()`: Retrieve top-k most relevant documents
- `add_documents()`: Index documents with embeddings

`collection_name` is implemented on the base class and should not be overridden.

**Current implementations**:

- `MilvusVectorStore`: Milvus — serves both a remote server (`MilvusConfig`) and the embedded, local-file Milvus Lite engine (`MilvusLiteConfig`) (hybrid: dense + BM25)
- `PGVectorStore`: PostgreSQL + pgvector (hybrid: dense + tsvector full-text)

---

## Using Different Providers

The beauty of the provider-agnostic design is that you can **mix and match** components from different providers.

The `foundation_model` and `embedding_model` below are the model wrappers built in the [OpenShift MaaS Integration](#openshift-maas-integration) usage snippet above — only the `vector_store_config` differs between the examples.

### Example 1: MaaS Models with Milvus

Use MaaS for foundation and embedding models, and Milvus (direct client) as the vector store:

```python
from ai4rag.rag.vector_store import MilvusConfig
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.utils.event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=AI4RAGSearchSpace(
        params=[
            Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
            Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
            # ... other params
        ]
    ),
    vector_store_config=MilvusConfig.from_env(),
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="./output"),  # required
)
```

---

### Example 2: MaaS Models with Milvus Lite

Use MaaS for models, but embedded Milvus Lite for quick local development:

```python
from ai4rag.rag.vector_store import MilvusLiteConfig
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.utils.event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=AI4RAGSearchSpace(
        params=[
            Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
            Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
            # ... other params
        ]
    ),
    vector_store_config=MilvusLiteConfig(db_path="./ai4rag.db"),  # embedded Milvus Lite, local file
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="./output"),  # required
)
```

!!! warning "Milvus Lite Is Not for Production"
    Milvus Lite fully supports `search_mode="hybrid"`, but its BM25 statistics are computed segment-locally
    rather than corpus-wide, so hybrid-search ranking fidelity (and any scores measured against it) may not
    transfer exactly to a production server. Use a remote Milvus server, Zilliz Cloud, or `PGVectorConfig` for
    production workloads.

---

## Milvus Lite for Development

Milvus Lite is the fastest way to get started with ai4rag without setting up external services — it's the embedded, local-file counterpart to a full Milvus deployment, configured via its own `MilvusLiteConfig` class rather than `MilvusConfig` (which is reserved for a remote server and validates its `uri` accordingly).

### Quick Setup

No configuration needed beyond a local file path — just pass `vector_store_config=MilvusLiteConfig(db_path="./ai4rag.db")` (or `MilvusLiteConfig()` for the default path):

```python
from pathlib import Path
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.rag.vector_store import MilvusLiteConfig
from ai4rag.utils.event_handler import LocalEventHandler

from dev_utils.file_store import FileStore
from dev_utils.utils import read_benchmark_from_json

# Load data (models built as in the MaaS Integration snippet above)
documents = FileStore(Path("./docs")).load_as_documents()
benchmark_data = read_benchmark_from_json(Path("./benchmark.json"))

# Run experiment with Milvus Lite (no external vector database setup needed!)
experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_config=MilvusLiteConfig(db_path="./ai4rag.db"),  # embedded, zero-server, local file
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="./output"),  # required
)

best_pattern = experiment.search()
```

**When to use Milvus Lite**:

- Local development and testing
- Prototyping RAG configurations, including hybrid search
- Small-to-medium document sets (up to roughly 1M vectors)
- Quick experiments without infrastructure setup

**When NOT to use Milvus Lite**:

- Production deployments — use a remote Milvus server, Zilliz Cloud, or pgvector instead
- Large-scale corpora beyond the embedded engine's target range
- Workloads needing corpus-global BM25 fidelity for hybrid search
- Concurrent writers to the same store (Milvus Lite serializes writes to its `.db` file)

---

## Extending with Custom Providers

Want to add support for a new provider? Implement the base classes:

### Adding a New Foundation Model

```python
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel, MessageTyped

class MyCustomFoundationModel(BaseFoundationModel):
    """Integration with my custom LLM provider."""

    def __init__(self, client, model_id, params):
        super().__init__(
            client=client,
            model_id=model_id,
            params=params,
            system_message_text="Your custom system prompt",  # Optional
            user_message_text="Your custom user prompt template",  # Optional
        )

    def chat(self, messages: list[MessageTyped]) -> list[MessageTyped]:
        """Call your custom LLM API."""
        # Transform messages to your API format
        response = self.client.generate(
            model=self.model_id,
            messages=messages,
            **self.params
        )

        # Transform response back to MessageTyped format
        return messages + [{"role": "assistant", "content": response.text}]
```

---

### Adding a New Embedding Model

```python
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

class MyCustomEmbeddingModel(BaseEmbeddingModel):
    """Integration with my custom embedding provider."""

    def __init__(self, client, model_id, params):
        super().__init__(client=client, model_id=model_id, params=params)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed documents."""
        response = self.client.embed(
            model=self.model_id,
            texts=texts
        )
        return response.embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        response = self.client.embed(
            model=self.model_id,
            texts=[query]
        )
        return response.embeddings[0]
```

---

### Adding a New Vector Store

```python
from ai4rag.rag.vector_store import BaseVectorStore
from ai4rag.rag.chunking.chunk import AI4RAGChunk

class MyCustomVectorStore(BaseVectorStore):
    """Integration with my custom vector database."""

    def __init__(self, embedding_model, config, distance_metric, collection_name=None):
        super().__init__(embedding_model, config, distance_metric, collection_name)
        # Initialize your vector store client, e.g. using `config` for connection
        # details and `self.collection_name` for the collection to open/create.

    def add_documents(self, documents: Sequence[AI4RAGChunk]) -> None:
        """Index documents with embeddings."""
        texts = [doc.text for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)

        # Insert into your vector database
        self.client.insert(
            collection=self.collection_name,
            vectors=embeddings,
            metadata=[doc.metadata for doc in documents]
        )

    def search(self, query: str, k: int, **kwargs) -> list[AI4RAGChunk]:
        """Retrieve top-k similar documents."""
        query_embedding = self.embedding_model.embed_query(query)

        # Query your vector database
        results = self.client.search(
            collection=self.collection_name,
            vector=query_embedding,
            top_k=k
        )

        # Transform to ai4rag format
        return [
            AI4RAGChunk(text=r.text, metadata=r.metadata)
            for r in results
        ]
```

!!! note "`collection_name` is provided by the base class"
    `BaseVectorStore.__init__` resolves `collection_name` for you (reusing it if
    supplied, otherwise generating a compliant name) and exposes it as the
    `collection_name` property. Custom subclasses should not override it.

---

## Vector Store Backends

`ai4rag` selects the vector store implementation from the *type* of the config object you pass as `vector_store_config` — there's no separate string to keep in sync.

| Config class | Provider | Key connection params | Env vars (`.from_env()`) |
|---|---|---|---|
| `MilvusConfig` | Milvus — remote server or Zilliz Cloud only (hybrid: dense + BM25) | `uri` (required, must be `http(s)://`; raises `ValueError` otherwise), `token`, `server_cert` | `MILVUS_URI` (required, must be `http(s)://`), `MILVUS_TOKEN`, `MILVUS_SERVER_CERT` |
| `MilvusLiteConfig` | Milvus Lite — embedded, local file only (hybrid: dense + BM25) | `db_path` (local file path; defaults to `"./ai4rag_milvus_lite.db"`; raises `ValueError` if given a `http(s)://` value) | `MILVUS_LITE_DB_PATH` (optional) |
| `PGVectorConfig` | PostgreSQL + pgvector (hybrid: dense + full-text) | `host`, `port`, `dbname`, `user`, `password` | `PGVECTOR_HOST`, `PGVECTOR_PORT`, `PGVECTOR_DB`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD` |

Both `MilvusConfig` and `MilvusLiteConfig` are served by the same `MilvusVectorStore` implementation; they only differ in where the data lives (remote server vs. local file) and are validated to prevent mixing the two up (see the note under [Milvus Lite (Embedded, Local File)](#milvus-lite-embedded-local-file)).

Each config class is a frozen, keyword-only dataclass with a `.from_env()` classmethod that builds an instance from the environment variables above:

```python
from ai4rag.rag.vector_store import MilvusConfig, MilvusLiteConfig, PGVectorConfig

# Embedded Milvus Lite backed by a local file — no external service, no env vars required
milvus_lite_config = MilvusLiteConfig(db_path="./ai4rag.db")

# Remote Milvus server (or Zilliz Cloud), reading MILVUS_URI / MILVUS_TOKEN / MILVUS_SERVER_CERT
milvus_config = MilvusConfig.from_env()

# PGVector, reading PGVECTOR_HOST / PGVECTOR_PORT / PGVECTOR_DB / PGVECTOR_USER / PGVECTOR_PASSWORD
pgvector_config = PGVectorConfig.from_env()
```

Pass the resulting config as `vector_store_config` to `AI4RAGExperiment`, or build a store directly with `get_vector_store(embedding_model, config, collection_name=None)`.

---

## Provider Comparison

| Feature | OpenShift MaaS | Milvus Lite (embedded) | Milvus (server) | PGVector |
|---------|------------|----------|--------|----------|
| **Foundation Models** | Yes (any deployed chat model) | N/A | N/A | N/A |
| **Embedding Models** | Yes (any deployed embedding model) | N/A | N/A | N/A |
| **Vector Store** | No (models only) | Yes (local file) | Yes | Yes |
| **Hybrid Search** | N/A | Yes (dense + BM25, segment-local IDF) | Yes (dense + BM25) | Yes (dense + full-text) |
| **Setup Complexity** | Medium (MaaS deployment required) | None | Medium (server required) | Medium (server required) |
| **Cost** | Self-hosted (infra cost) | Free | Self-hosted (infra cost) | Self-hosted (infra cost) |
| **Best For** | On-prem, self-hosted OpenAI-compatible models | Local dev, testing, prototyping | Production, hybrid search | Production, hybrid search, existing Postgres infra |

---

## Summary

`ai4rag`'s provider-agnostic design:

- **Abstract base classes**: `BaseFoundationModel`, `BaseEmbeddingModel`, `BaseVectorStore`
- **Extensible**: Add support for new providers by implementing base classes
- **OpenShift MaaS**: OpenAI-SDK access to any deployed foundation and embedding model
- **Direct-client vector stores**: `MilvusVectorStore`, backed by either `MilvusLiteConfig` (embedded Milvus Lite for zero-config local development) or `MilvusConfig` (remote server/Zilliz Cloud for production), plus `PGVectorConfig`/`PGVectorStore` for production deployments — all with hybrid search

The choice of provider doesn't affect the optimization process - ai4rag works the same regardless of which model you're using. Focus on finding the best RAG configuration for your use case, not your infrastructure.
