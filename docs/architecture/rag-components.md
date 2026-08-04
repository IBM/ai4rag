# RAG Components

This page provides detailed architecture documentation for the RAG pipeline components that ai4rag optimizes.

---

## Component Hierarchy

```mermaid
classDiagram
    class BaseFoundationModel {
        <<abstract>>
        +client: ClientT
        +model_id: str
        +params: ParamsT
        +language: Language
        +system_message_text: str
        +user_message_text: str
        +context_template_text: str
        +chat(messages, **kwargs)* list
    }

    class OGXFoundationModel {
        +client: OgxClient
        +params: OGXModelParameters
        +chat(messages, **kwargs) list
    }

    class BaseEmbeddingModel {
        <<abstract>>
        +client: ClientT
        +model_id: str
        +params: ParamsT
        +embed_documents(texts)* list
        +embed_query(query)* list
    }

    class OGXEmbeddingModel {
        +client: OgxClient
        +params: OGXEmbeddingParams
        +embed_documents(texts) list
        +embed_query(query) list
    }

    class BaseVectorStore {
        <<abstract>>
        +embedding_model: BaseEmbeddingModel
        +config: BaseVectorStoreConfig
        +distance_metric: str
        +collection_name: str
        +search(query, k, **kwargs)* AI4RAGChunk[]
        +add_documents(AI4RAGChunk[])* void
    }

    class ChromaVectorStore {
        +search(query, k, include_scores) AI4RAGChunk[]
        +window_search(query, k, window_size) AI4RAGChunk[]
        +add_documents(AI4RAGChunk[]) void
    }

    class MilvusVectorStore {
        +search(query, k, search_mode, ranker_*) AI4RAGChunk[]
        +add_documents(AI4RAGChunk[]) void
    }

    class PGVectorStore {
        +search(query, k, search_mode, ranker_*) AI4RAGChunk[]
        +add_documents(AI4RAGChunk[]) void
    }

    class BaseChunker {
        <<abstract>>
        +split_documents(DoclingDocument[])* AI4RAGChunk[]
        +to_dict()* dict
        +from_dict(d)* BaseChunker
    }

    class DoclingChunker {
        +max_tokens: int
        +contextualize: bool
        +merge_peers: bool
        +split_documents(DoclingDocument[]) AI4RAGChunk[]
    }

    class LangChainChunker {
        +method: str
        +chunk_size: int
        +chunk_overlap: int
        +split_documents(DoclingDocument[]) AI4RAGChunk[]
    }

    class Retriever {
        +vector_store: BaseVectorStore
        +method: str
        +number_of_chunks: int
        +search_mode: str
        +ranker_strategy: str
        +ranker_k: int
        +ranker_alpha: float
        +retrieve(query) list
    }

    class BaseRAGTemplate {
        <<abstract>>
        +foundation_model: BaseFoundationModel
        +retriever: Retriever
        +build_index(docs)* void
        +generate(question)* dict
        +generate_stream(question)* iterator
    }

    class SimpleRAG {
        +chunker: BaseChunker
        +embedding_model: BaseEmbeddingModel
        +vector_store: BaseVectorStore
        +build_index(DoclingDocument[]) void
        +generate(question) dict
        +generate_stream(question) iterator
    }

    BaseFoundationModel <|-- OGXFoundationModel
    BaseEmbeddingModel <|-- OGXEmbeddingModel
    BaseVectorStore <|-- ChromaVectorStore
    BaseVectorStore <|-- MilvusVectorStore
    BaseVectorStore <|-- PGVectorStore
    BaseChunker <|-- DoclingChunker
    BaseChunker <|-- LangChainChunker
    BaseRAGTemplate <|-- SimpleRAG

    BaseVectorStore --> BaseEmbeddingModel : uses
    Retriever --> BaseVectorStore : uses
    BaseRAGTemplate --> BaseFoundationModel : uses
    BaseRAGTemplate --> Retriever : uses
    SimpleRAG --> BaseChunker : uses
```

---

## Foundation Models

Foundation models generate text responses given prompts and retrieved context.

### BaseFoundationModel

Abstract base class defining the foundation model interface:

```python
class BaseFoundationModel(Generic[ClientT, ParamsT], ABC):
    def __init__(
        self,
        client: ClientT,
        model_id: str,
        params: ParamsT,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
        language: Language | None = None,
    ):
```

**Language-Aware Prompt Generation:**

The optional `language` parameter accepts a `Language` dataclass (with `code` and `name` fields) and controls language-aware prompt template generation. When set, `user_message_text` is regenerated to include language-specific instructions. Defaults to `Language(code="", name="auto")`.

**Configurable Prompt Templates:**

Foundation models support three customizable prompt templates:

**1. system_message_text**

The system prompt that defines the model's behavior:

```python
# Default:
"You are a helpful, respectful and honest assistant. "
"Always answer as helpfully as possible, while being safe."
```

**2. user_message_text**

Template for formatting the user's question with retrieved context:

```python
# Default:
"{reference_documents}\n\nQuestion: {question}\nAnswer:"
```

Placeholders:
- `{reference_documents}`: Formatted context from retrieval
- `{question}`: The user's question

**3. context_template_text**

Template for formatting each retrieved document:

```python
# Default:
"According to the document: {document}\n"
```

Placeholder:
- `{document}`: Individual chunk's text content

**Customization Example:**

```python
foundation_model = OGXFoundationModel(
    model_id="ollama/llama3.2:3b",
    client=client,
    system_message_text="You are a technical documentation assistant specialized in software APIs.",
    user_message_text="Context:\n{reference_documents}\n\nUser Question: {question}\n\nDetailed Answer:",
    context_template_text="[Document {document_id}] {document}\n\n"
)
```

**Prompt Template Validation:**

The `user_message_text` and `context_template_text` attributes are validated properties that check for required placeholders (`{question}`, `{reference_documents}` in user message; `{document}` in context template) on assignment. Invalid templates raise a `ValueError`.

**Interface Method:**

```python
@abstractmethod
def chat(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
    """Chat with the model based on the client capabilities."""
```

**MessageTyped Format:**

```python
class MessageTyped(TypedDict):
    role: str      # "system", "user", or "assistant"
    content: str   # Message text
```

### OGXFoundationModel

OGX integration for foundation models:

```python
class OGXFoundationModel(BaseFoundationModel[OgxClient, OGXModelParameters]):
    def __init__(
        self,
        client: OgxClient,
        model_id: str,
        params: dict | OGXModelParameters | None = None,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
        language: Language | None = None,
    ):
```

**Parameters:**

```python
@dataclass
class OGXModelParameters:
    max_completion_tokens: int = 1024  # Max tokens in response
    temperature: float = 0.1            # Sampling temperature (0.0-1.0)
```

**Chat Implementation:**

```python
def chat(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
    response = self.client.chat.completions.create(
        model=self.model_id,
        messages=messages,
        max_completion_tokens=self.params.max_completion_tokens,
        temperature=self.params.temperature,
    )
    return response.choices  # List of response choices
```

**Usage:**

```python
foundation_model = OGXFoundationModel(
    model_id="ollama/llama3.2:3b",
    client=ogx_client,
    params={"max_completion_tokens": 512, "temperature": 0.0}
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]

response = foundation_model.chat(messages)
answer = response[0].message.content
```

---

## Embedding Models

Embedding models convert text into dense vector representations for semantic search.

### BaseEmbeddingModel

Abstract base class for embedding models:

```python
class BaseEmbeddingModel(ABC, Generic[ClientT, ParamsT]):
    def __init__(
        self,
        client: ClientT,
        model_id: str,
        params: ParamsT | None = None
    ):
```

**Interface Methods:**

```python
@abstractmethod
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    """Embed multiple documents (used during indexing)."""

@abstractmethod
def embed_query(self, query: str) -> list[float]:
    """Embed a single query (used during retrieval)."""
```

### OGXEmbeddingModel

OGX integration with auto-detection of model capabilities:

```python
class OGXEmbeddingModel(BaseEmbeddingModel[OgxClient, OGXEmbeddingParams]):
    def __init__(
        self,
        client: OgxClient,
        model_id: str,
        params: dict | OGXEmbeddingParams | None = None
    ):
```

**Parameters:**

```python
@dataclass
class OGXEmbeddingParams:
    embedding_dimension: int | None = None    # Auto-detected if None
    context_length: int | None = None         # Auto-detected if None
    timeout: float | Timeout | None = None
    model_type: str | None = None
    provider_id: str | None = None
    provider_resource_id: str | None = None
```

**Auto-Detection:**

When `embedding_dimension` or `context_length` not provided, the model auto-detects them on first use:

**Chunk Truncation:**

When a chunk exceeds the embedding model's context length, `OGXEmbeddingModel` automatically truncates it using a progressive margin strategy (5%, then 10%) before retrying. This prevents embedding failures for oversized chunks while preserving as much content as possible.

**Embedding Dimension Detection:**

```python
def _detect_embedding_dimension(self) -> int:
    """Embed a test string and count dimensions."""
    test_embedding = self._embed_text("test")[0]
    return len(test_embedding)  # e.g., 768 for nomic-embed-text
```

**Context Length Detection:**

```python
def _detect_context_length(self) -> int:
    """Binary search to find max context length."""
    lo, hi, best = 64, 8192, None

    while hi - lo >= 64:
        mid = (lo + hi) // 2
        probe_text = "word " * mid  # Approx. 1 word = 1 token
        try:
            self._embed_text(probe_text)
            best = mid
            lo = mid + 1
        except:
            hi = mid - 1

    return best
```

**Performance:** ~5 API calls for context length detection via binary search.

**Batch Processing:**

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    """Process in batches of 1024 to respect API limits."""
    embeddings = []
    for idx in range(0, len(texts), 1024):
        batch = texts[idx : idx + 1024]
        batch_embeddings = self._embed_text(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

**Usage:**

```python
# Auto-detect parameters
embedding_model = OGXEmbeddingModel(
    model_id="ollama/nomic-embed-text:latest",
    client=ogx_client,
)
# First call triggers detection:
# - embedding_dimension = 768 (detected)
# - context_length = 8192 (detected)

# Or explicitly provide parameters
embedding_model = OGXEmbeddingModel(
    model_id="ollama/nomic-embed-text:latest",
    client=ogx_client,
    params={"embedding_dimension": 768, "context_length": 8192}
)

# Embed documents
embeddings = embedding_model.embed_documents(["text 1", "text 2", ...])
# Returns: [[0.1, -0.2, ...], [0.3, 0.1, ...], ...]

# Embed query
query_embedding = embedding_model.embed_query("What is X?")
# Returns: [0.05, -0.12, ...]
```

---

## Vector Stores

Vector stores manage document storage, embedding indexing, and similarity search.

### BaseVectorStore

Abstract base class for vector stores:

```python
class BaseVectorStore(ABC):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: BaseVectorStoreConfig,
        distance_metric: str,
        collection_name: str | None = None
    ):
```

**Configuration:**

Every concrete store is constructed from a typed, frozen `config` dataclass (`ChromaConfig`, `MilvusConfig`, or `PGVectorConfig`) that carries the backend's connection parameters and a `provider` discriminator (`"chroma"`, `"milvus"`, `"pgvector"`). Each config class exposes a `from_env()` classmethod that reads its own `*_ENV` variables, so connection details never need to be hardcoded in application code or generated artifacts (e.g. pattern notebooks).

**Collection naming (shared across all backends):**

The base class resolves `collection_name` once, in one place, via
`ai4rag.rag.vector_store.utils.resolve_collection_name`, so every backend
behaves identically:

- **Auto-generation** — when `collection_name` is `None`, a unique name of the
  form `ai4rag_<UTC timestamp>_<8 random chars>` is generated.
- **Mandatory `ai4rag` prefix** — a caller-supplied name **must** start with
  `ai4rag`. This prefix is the cross-backend isolation guard: because every
  collection (and, for pgvector, the physical table it maps to one-to-one)
  starts with it, ai4rag never creates, reuses, or drops a table/collection it
  does not own. A non-compliant name raises `ValueError` rather than being
  silently coerced.
- **Identifier safety** — the name is sanitized into a valid identifier
  (non-alphanumeric characters become underscores) and bounded to 63 characters
  (the tightest limit across PostgreSQL and Chroma), so it is usable verbatim as
  a backend collection name *and* as a physical SQL table name.

**Interface Methods:**

```python
@abstractmethod
def search(self, query: str, k: int, **kwargs) -> list[AI4RAGChunk]:
    """Search for k most relevant chunks."""

@abstractmethod
def add_documents(self, documents: Sequence[AI4RAGChunk]) -> None:
    """Add chunks to the collection."""

@property
def collection_name(self) -> str:
    """The resolved collection name (reused or auto-generated).

    Concrete on the base class — guaranteed to start with ``ai4rag`` and to be a
    valid, length-bounded identifier usable as both a collection name and a SQL
    table name.
    """
```

### Choosing a Backend

`ai4rag.rag.vector_store.get_vector_store` is the recommended entry point for constructing a vector store: it inspects `config.provider` and instantiates the matching concrete class, so callers do not need to import or branch on individual store classes.

```python
from ai4rag.rag.vector_store import get_vector_store, MilvusConfig

vector_store = get_vector_store(
    embedding_model=embedding_model,
    config=MilvusConfig.from_env(),
    collection_name=None,  # omit to auto-generate; pass an existing name to reuse it
)
```

**Signature:**

```python
def get_vector_store(
    embedding_model: BaseEmbeddingModel,
    config: ChromaConfig | MilvusConfig | PGVectorConfig,
    collection_name: str | None = None,
) -> BaseVectorStore:
    """Backend selected by ``config.provider``; raises TypeError on a
    config/provider mismatch, ValueError for an unsupported provider."""
```

**Available Configs:**

| Config | `provider` | Key Fields | Env Vars |
|--------|------------|------------|----------|
| `ChromaConfig` | `"chroma"` | `persist_directory`, `host`, `port` | `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_PERSIST_DIR` |
| `MilvusConfig` | `"milvus"` | `uri` (required), `token`, `server_cert` | `MILVUS_URI` (required), `MILVUS_TOKEN`, `MILVUS_SERVER_CERT` |
| `PGVectorConfig` | `"pgvector"` | `host`, `port`, `dbname`, `user`, `password` | `PGVECTOR_HOST`, `PGVECTOR_PORT`, `PGVECTOR_DB`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD` |

`get_vector_store_config(provider)` and `get_vector_store_env_vars(provider)` complement `get_vector_store` when only a provider string is available (e.g. when building a config from the `vector_store_type` selected on the search space):

```python
from ai4rag.rag.vector_store import get_vector_store_config, get_vector_store_env_vars

config = get_vector_store_config("milvus")           # MilvusConfig.from_env()
env_vars = get_vector_store_env_vars("milvus")        # (("MILVUS_URI", "..."), ...)
```

### ChromaVectorStore

In-memory ChromaDB implementation for development and testing. Chroma is **vector-only** — it does not support hybrid (dense + keyword) search:

```python
class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: ChromaConfig | None = None,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
        **kwargs
    ):
```

**Supported Distance Metrics:**

- `"cosine"`: Cosine similarity (default)
- `"l2"`: Euclidean distance

**Search Methods:**

**1. Standard Search:**

```python
def search(
    self,
    query: str,
    k: int = 5,
    include_scores: bool = False,
    **kwargs
) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
    """Vector similarity search."""
```

**2. Window Search:**

```python
def window_search(
    self,
    query: str,
    k: int = 5,
    window_size: int = 2,
    include_scores: bool = False,
    **kwargs
) -> list[AI4RAGChunk]:
    """Retrieve chunks + adjacent chunks (window) from same document."""
```

**Window Search Details:**

For each retrieved chunk:
1. Extract `document_id` and `sequence_number` from metadata
2. Query vector store for chunks with:
   - Same `document_id`
   - `sequence_number` in `[seq - window_size, seq + window_size]`
3. Sort by `sequence_number`
4. Merge into single chunk (concatenate text)

**Example:**

```python
# Retrieved chunk: document_id="doc1", sequence_number=5
# window_size=2
# Fetches chunks with sequence_number in [3, 4, 5, 6, 7]
# Returns merged document with all 5 chunks concatenated
```

**Batch Document Addition:**

```python
def add_documents(self, documents: list[AI4RAGChunk], max_batch_size: int = 2048) -> list[str]:
    """Add chunks in batches of max_batch_size."""
    for batch_start in range(0, len(docs), max_batch_size):
        batch = docs[batch_start : batch_start + max_batch_size]
        self._vector_store.add_documents(batch, ids=ids)
```

**Usage:**

```python
vector_store = ChromaVectorStore(
    embedding_model=embedding_model,
    distance_metric="cosine"
)

# Index documents
vector_store.add_documents(chunked_documents)

# Search
results = vector_store.search(query="What is X?", k=5)
# Returns: [AI4RAGChunk(...), AI4RAGChunk(...), ...]

# Window search
results = vector_store.window_search(query="What is X?", k=5, window_size=2)
# Returns: [merged_chunk_1, merged_chunk_2, ...]
```

### MilvusVectorStore

Vector store backed by a remote Milvus instance via `pymilvus`, supporting both pure dense vector search and hybrid search (dense + BM25 sparse) with **server-side** fusion:

```python
class MilvusVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: MilvusConfig,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
    ):
```

**Connection Configuration:**

TLS is driven entirely by the `uri` scheme: `https://` opens a secure channel, `http://` stays plaintext. For endpoints with a self-signed or private-CA certificate, pass the PEM text via `server_cert`.

```python
from ai4rag.rag.vector_store import MilvusConfig

# From environment: MILVUS_URI (required), MILVUS_TOKEN, MILVUS_SERVER_CERT
config = MilvusConfig.from_env()

# Or explicit
config = MilvusConfig(uri="https://localhost:19530", token="user:pass")
```

**Collection Schema:**

For a new collection, `MilvusVectorStore` creates a schema with a primary `chunk_id`, an analyzed `content` field, a dense `vector` field sized to the embedding model's dimension, a `chunk_content` JSON payload, and a `sparse` BM25 vector — with a FLAT/COSINE index on `vector`, a sparse inverted BM25 index on `sparse`, and a BM25 function deriving `sparse` from `content`. When `collection_name` names an existing collection, it is reused unchanged.

**Hybrid Search Support:**

```python
def search(
    self,
    query: str,
    k: int = 5,
    include_scores: bool = False,
    search_mode: str = "vector",
    ranker_strategy: str | None = None,
    ranker_k: int | None = None,
    ranker_alpha: float | None = None,
    **kwargs,
) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
```

**Search Modes:**

**1. Vector Mode (default):**

```python
results = vector_store.search(
    query="What is X?",
    k=5,
    search_mode="vector"
)
```

Pure semantic search using dense embeddings.

**2. Hybrid Mode:**

```python
results = vector_store.search(
    query="What is X?",
    k=5,
    search_mode="hybrid",
    ranker_strategy="rrf",
    ranker_k=60
)
```

Issues a dense `AnnSearchRequest` on `vector` and a sparse `AnnSearchRequest` on `sparse`, fused **on the Milvus server** with a native `RRFRanker` or `WeightedRanker`.

**Ranker Strategies:**

| Strategy | Description | Parameters |
|----------|-------------|------------|
| `"rrf"` | Reciprocal Rank Fusion (default fallback) | `ranker_k`: smoothing constant (30-100), default 60 |
| `"weighted"` | Weighted combination | `ranker_alpha`: dense weight (0.0-1.0), default 0.5; sparse weight is `1 - ranker_alpha` |
| `"normalized"` | Falls through to RRF fusion | Strategy-specific |

**RRF Example:**

```python
results = vector_store.search(
    query="What is X?",
    k=5,
    search_mode="hybrid",
    ranker_strategy="rrf",
    ranker_k=60,
)
```

**Weighted Example:**

```python
# 70% dense (semantic), 30% sparse (keyword)
results = vector_store.search(
    query="What is X?",
    k=5,
    search_mode="hybrid",
    ranker_strategy="weighted",
    ranker_alpha=0.7,
)
```

**Validation:**

`MilvusVectorStore` and `PGVectorStore` both validate their hybrid search parameters through the shared `ai4rag.rag.vector_store.utils.validate_search_params`:

```python
def validate_search_params(search_mode, ranker_strategy, ranker_k, ranker_alpha):
    # When search_mode != "hybrid":
    #   - ranker_strategy must be None or ""
    #   - ranker_k must be None or 0
    #   - ranker_alpha must be None or 1

    # When search_mode == "hybrid":
    #   - ranker_strategy must be non-empty ("rrf", "weighted", "normalized")
    #   - ranker_k > 0 only for "rrf"
    #   - ranker_alpha != 1 only for "weighted"
```

**Document Addition:**

```python
def add_documents(self, documents: list[AI4RAGChunk], **kwargs) -> None:
    """Embed, deduplicate by chunk_id, and upsert chunks into Milvus."""
    embeddings = self.embedding_model.embed_documents([doc.text for doc in documents])

    data = [
        {
            "chunk_id": doc.chunk_id,
            "content": doc.text,
            "vector": embedding,
            "chunk_content": {"content": doc.text, "metadata": doc.metadata, "chunk_id": doc.chunk_id},
        }
        for doc, embedding in iter_unique_chunks(documents, embeddings)
    ]

    batch_size = kwargs.get("batch_size", self._BATCH_SIZE)  # default 2048
    for idx in range(0, len(data), batch_size):
        self._client.upsert(self._collection_name, data=data[idx : idx + batch_size])
```

**Usage:**

```python
from ai4rag.rag.vector_store import MilvusVectorStore, MilvusConfig

# Create vector store (omit collection_name to auto-generate a new collection)
vector_store = MilvusVectorStore(
    embedding_model=ogx_embedding_model,
    config=MilvusConfig.from_env(),
)

# Index documents
vector_store.add_documents(chunked_documents)

# Vector search
results = vector_store.search(query="What is X?", k=5)

# Hybrid search with RRF
results = vector_store.search(
    query="What is X?",
    k=5,
    search_mode="hybrid",
    ranker_strategy="rrf",
    ranker_k=60
)

# Hybrid search with weighted ranker
results = vector_store.search(
    query="What is X?",
    k=5,
    search_mode="hybrid",
    ranker_strategy="weighted",
    ranker_alpha=0.7
)

# Reuse an existing collection instead of creating a new one
vector_store = MilvusVectorStore(
    embedding_model=ogx_embedding_model,
    config=MilvusConfig.from_env(),
    collection_name="ai4rag_20260701120000_ab12cd34",
)
```

### PGVectorStore

Vector store backed by PostgreSQL with the `pgvector` extension, supporting pure dense vector search and hybrid search (dense vector + `tsvector` full-text) with **in-memory** fusion:

```python
class PGVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: PGVectorConfig,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
    ):
```

**Connection Configuration:**

```python
from ai4rag.rag.vector_store import PGVectorConfig

# From environment: PGVECTOR_HOST, PGVECTOR_PORT, PGVECTOR_DB, PGVECTOR_USER, PGVECTOR_PASSWORD
config = PGVectorConfig.from_env()

# Or explicit
config = PGVectorConfig(host="localhost", port=5432, dbname="ai4rag", user="ai4rag", password="secret")
```

**Table Mapping:**

The resolved `collection_name` is used verbatim as the physical PostgreSQL table name — created with an `id` primary key, a `document` JSONB payload, an `embedding` vector column, `content_text`, and a `tokenized_content` `tsvector` column feeding full-text search. Supported `distance_metric` values are `"cosine"`, `"l2"`, `"l1"`, and `"inner_product"`.

!!! warning "Embedding dimension limit"
    pgvector caps HNSW indexes on the `vector` type at 2000 dimensions. `PGVectorStore.__init__` raises `ValueError` up front if the embedding model's dimension exceeds this limit, rather than failing later on the first indexed search — use `MilvusVectorStore` for higher-dimensional embedding models.

**Hybrid Search:**

`PGVectorStore.search` accepts the same `search_mode`, `ranker_strategy`, `ranker_k`, and `ranker_alpha` parameters as `MilvusVectorStore` (see the **Ranker Strategies** table under [MilvusVectorStore](#milvusvectorstore) above). The fusion mechanics differ, however: the dense search orders rows by the configured pgvector distance operator, the keyword search ranks rows by `ts_rank` against a `plainto_tsquery`, and the two independent score maps are combined **in Python** via `WeightedInMemoryAggregator` (see [Reranker](#reranker) below) before the top `k` results are returned.

**Usage:**

```python
from ai4rag.rag.vector_store import PGVectorStore, PGVectorConfig

vector_store = PGVectorStore(
    embedding_model=ogx_embedding_model,
    config=PGVectorConfig.from_env(),
)

vector_store.add_documents(chunked_documents)

# Hybrid search with RRF
results = vector_store.search(
    query="What is X?",
    k=5,
    search_mode="hybrid",
    ranker_strategy="rrf",
    ranker_k=60,
)
```

### Reranker

`ai4rag.rag.vector_store.reranker.WeightedInMemoryAggregator` implements the in-memory score fusion used by `PGVectorStore`'s hybrid search (Milvus fuses server-side instead, via its native rankers). It exposes three static methods:

```python
class WeightedInMemoryAggregator:
    @staticmethod
    def weighted_rerank(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        alpha: float = 0.5,
    ) -> dict[str, float]:
        """Weighted average of min-max normalized vector and keyword scores."""

    @staticmethod
    def rrf_rerank(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        k: float = 60.0,
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion of vector and keyword result rankings."""

    @staticmethod
    def combine_search_results(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        reranker_type: str = "rrf",
        reranker_params: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Dispatch to weighted_rerank or rrf_rerank based on reranker_type."""
```

`combine_search_results` is the single entry point: it dispatches to `weighted_rerank` (reading `reranker_params["alpha"]`) when `reranker_type == "weighted"`, and to `rrf_rerank` (reading `reranker_params["k"]`) otherwise — including for `"normalized"`, which currently falls through to RRF.

---

## Chunking

Chunkers split `DoclingDocument` objects into `AI4RAGChunk` instances for embedding and retrieval.

### AI4RAGChunk

Framework-agnostic chunk representation used across the pipeline:

```python
@dataclass
class AI4RAGChunk:
    text: str                                  # Chunk content
    metadata: dict[str, Any] = field(default_factory=dict)  # document_id, sequence_number, etc.
    chunk_id: str = field(init=False, repr=False)  # Deterministic SHA-256 (auto-computed)
```

### BaseChunker

Abstract base class for chunkers:

```python
class BaseChunker(ABC):
    @abstractmethod
    def split_documents(self, documents: Sequence[DoclingDocument]) -> list[AI4RAGChunk]:
        """Split documents into smaller chunks."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize chunker configuration."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseChunker":
        """Deserialize chunker configuration."""
```

### DoclingChunker

Structure-aware, token-aware chunker wrapping docling's `HybridChunker`. Preserves document hierarchy (headings, tables, figures) during chunking:

```python
class DoclingChunker(BaseChunker):
    def __init__(
        self,
        max_tokens: int = 8192,
        contextualize: bool = True,
        tokenizer: BaseTokenizer | None = None,
        merge_peers: bool = True,
    ):
```

**Key Features:**

- Operates directly on `DoclingDocument` objects
- Token-bounded chunks aligned to the embedding model
- When `contextualize=True`, enriches each chunk with its heading hierarchy
- Merges adjacent undersized chunks that share the same heading context
- Does **not** support chunk overlap (overlap must be `0`)

**Usage:**

```python
chunker = DoclingChunker(max_tokens=1024, contextualize=True)

chunks = chunker.split_documents(docling_documents)
# Returns: list[AI4RAGChunk] with document_id, sequence_number, and headings metadata
```

### LangChainChunker

Token-based chunking via LangChain's `RecursiveCharacterTextSplitter`, adapted for `DoclingDocument` input:

```python
class LangChainChunker(BaseChunker):
    def __init__(
        self,
        method: Literal["recursive"] = "recursive",
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        **kwargs
    ):
```

**Chunking Method:**

Currently supports `"recursive"`. Converts each `DoclingDocument` to markdown internally, then applies token-based splitting using a character approximation (4 chars = 1 token):

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", r"(?<=\. )", "\n", " ", ""],
    length_function=lambda text: math.ceil(len(text) / 4),  # char-based approximation
    add_start_index=True,
)
```

**Splitting Hierarchy:**

1. **Double newlines** (`\n\n`): Paragraph boundaries
2. **Sentence boundaries** (`(?<=\. )`): After periods
3. **Single newlines** (`\n`): Line breaks
4. **Spaces** (` `): Word boundaries
5. **Characters** (`""`): Character-level splitting (last resort)

**Metadata Management:**

**1. Document ID Assignment:**

```python
def _set_document_id_in_metadata_if_missing(documents):
    for doc in documents:
        if "document_id" not in doc.metadata:
            doc.metadata["document_id"] = str(hash(doc.page_content))
```

**2. Sequence Number Assignment:**

```python
def _set_sequence_number_in_metadata(chunks):
    # Sort by (document_id, start_index)
    sorted_chunks = sorted(chunks, key=lambda x: (
        x.metadata["document_id"],
        x.metadata["start_index"]
    ))

    # Assign sequential numbers per document
    document_sequence = {}
    for chunk in sorted_chunks:
        doc_id = chunk.metadata["document_id"]
        seq_num = document_sequence.get(doc_id, 0) + 1
        document_sequence[doc_id] = seq_num
        chunk.metadata["sequence_number"] = seq_num

    return sorted_chunks
```

**Output Chunk Structure:**

```python
AI4RAGChunk(
    text="Chunk text content...",
    metadata={
        "document_id": "doc1",
        "sequence_number": 3,
        "start_index": 1024,
    }
)
```

**Usage:**

```python
chunker = LangChainChunker(
    method="recursive",
    chunk_size=512,
    chunk_overlap=128
)

chunks = chunker.split_documents(docling_documents)
# Returns: list[AI4RAGChunk] with sequence_number and start_index metadata
```

---

## Retrieval

The Retriever class coordinates document retrieval from vector stores.

### Retriever

```python
class Retriever:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        number_of_chunks: int,
        method: Literal["simple", "window"] = "simple",
        search_mode: Literal["vector", "hybrid"] = "vector",
        ranker_strategy: str | None = None,
        ranker_k: int | None = None,
        ranker_alpha: float | None = None,
    ):
```

**Parameters:**

- **vector_store**: Vector store instance to query
- **number_of_chunks**: Top-k parameter (how many chunks to retrieve)
- **method**: Retrieval method
  - `"simple"`: Return top-k chunks as-is
  - `"window"`: Expand each chunk to include adjacent chunks (ChromaDB only)
- **search_mode**: Search type
  - `"vector"`: Dense semantic search only
  - `"hybrid"`: Dense + sparse (keyword) search
- **ranker_strategy**: Hybrid search ranker (`"rrf"`, `"weighted"`, `"normalized"`)
- **ranker_k**: RRF smoothing parameter
- **ranker_alpha**: Weighted ranker dense/sparse balance

**Retrieve Method:**

```python
def retrieve(self, query: str, **kwargs) -> list[AI4RAGChunk]:
    """Retrieve relevant documents from vector store."""
    _number_of_chunks = kwargs.get("number_of_chunks", self.number_of_chunks)

    return self.vector_store.search(
        query,
        k=_number_of_chunks,
        search_mode=self.search_mode,
        ranker_strategy=self.ranker_strategy,
        ranker_k=self.ranker_k,
        ranker_alpha=self.ranker_alpha,
    )
```

**Simple vs Window Retrieval:**

The `method` parameter determines retrieval strategy but actual implementation depends on vector store:

- **MilvusVectorStore** / **PGVectorStore**: Always return simple chunks (no window expansion)
- **ChromaVectorStore**:
  - `method="simple"`: Returns top-k chunks
  - `method="window"`: Returns top-k chunks expanded with adjacent chunks

**Usage:**

```python
# Simple vector retrieval
retriever = Retriever(
    vector_store=vector_store,
    number_of_chunks=5,
    method="simple",
    search_mode="vector"
)

docs = retriever.retrieve("What is X?")
# Returns: [AI4RAGChunk(...), AI4RAGChunk(...), ...]  (5 chunks)

# Hybrid retrieval with RRF (Milvus or PGVector; Chroma is vector-only)
retriever = Retriever(
    vector_store=milvus_vector_store,
    number_of_chunks=5,
    method="simple",
    search_mode="hybrid",
    ranker_strategy="rrf",
    ranker_k=60
)

docs = retriever.retrieve("What is X?")
# Returns: 5 chunks re-ranked by RRF (dense + sparse)
```

---

## RAG Templates

RAG templates combine all components into end-to-end retrieval-augmented generation pipelines.

### BaseRAGTemplate

Abstract interface for RAG templates:

```python
class BaseRAGTemplate(ABC):
    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        retriever: Retriever,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: BaseVectorStore | None = None,
    ):
```

**Interface Methods:**

```python
@abstractmethod
def build_index(self, documents: list[DoclingDocument], **kwargs) -> None:
    """Index documents into vector store."""

@abstractmethod
def generate(self, question: str, **kwargs) -> dict[str, Any]:
    """Generate answer for question using RAG pipeline."""

@abstractmethod
def generate_stream(self, question: str, **kwargs):
    """Generate streaming answer (for future streaming support)."""
```

### SimpleRAG

Complete RAG implementation using OGX and LangChain:

```python
class SimpleRAG(BaseRAGTemplate):
    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        retriever: Retriever,
        chunker: BaseChunker | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        vector_store: BaseVectorStore | None = None,
    ):
```

**build_index() Method:**

```python
def build_index(self, documents: list[DoclingDocument], **kwargs) -> None:
    """Index documents: chunk → embed → store."""
    chunks = self.chunker.split_documents(documents)
    self.vector_store.add_documents(chunks)
```

**generate() Method:**

```python
def generate(self, question: str, **kwargs) -> dict[str, Any]:
    """Generate answer using RAG pipeline."""

    # 1. Retrieve relevant chunks
    reference_documents = self.retriever.retrieve(question, **kwargs)

    # 2. Format context
    context = "\n".join([
        self.foundation_model.context_template_text.format(
            document=chunk.text
        )
        for chunk in reference_documents
    ])

    # 3. Format user message
    user_message = self.foundation_model.user_message_text.format(
        reference_documents=context,
        question=question
    )

    # 4. Create messages
    messages = [
        {"role": "system", "content": self.foundation_model.system_message_text},
        {"role": "user", "content": user_message}
    ]

    # 5. Generate answer
    chat_response = self.foundation_model.chat(messages)

    # 6. Return result
    return {
        "answer": chat_response[0].message.content,
        "reference_documents": reference_documents,
        "question": question
    }
```

**generate_stream() Method:**

```python
def generate_stream(self, question: str, **kwargs):
    """Placeholder for streaming (currently non-streaming)."""
    result = self.generate(question, **kwargs)
    yield result["answer"]
```

**Usage:**

```python
# Create RAG template
rag = SimpleRAG(
    foundation_model=ogx_foundation_model,
    retriever=retriever,
    chunker=chunker,
    embedding_model=ogx_embedding_model,
    vector_store=vector_store
)

# Index documents (if building index manually)
rag.build_index(documents)

# Generate answer
result = rag.generate("What is the capital of France?")
print(result["answer"])
# "Based on the provided documents, Paris is the capital of France."

print(result["reference_documents"])
# [AI4RAGChunk(...), AI4RAGChunk(...), ...]
```

**Within AI4RAGExperiment:**

The experiment creates SimpleRAG instances automatically during evaluation:

```python
rag_pattern = SimpleRAG(
    foundation_model=foundation_model,
    retriever=retriever
)
# Note: chunker, embedding_model, vector_store handled separately
#       by experiment during indexing phase
```

---

## Component Integration Example

Full RAG pipeline with all components:

```python
from ogx_client import OgxClient
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel
from ai4rag.rag.vector_store import get_vector_store, MilvusConfig
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.template.simple_rag_template import SimpleRAG

# 1. Initialize the OGX client (still the foundation-model + embedding provider)
client = OgxClient(base_url="http://localhost:8000", api_key="...")

# 2. Create foundation model
foundation_model = OGXFoundationModel(
    model_id="ollama/llama3.2:3b",
    client=client,
    params={"max_completion_tokens": 512, "temperature": 0.1}
)

# 3. Create embedding model
embedding_model = OGXEmbeddingModel(
    model_id="ollama/nomic-embed-text:latest",
    client=client,
    params={"embedding_dimension": 768, "context_length": 8192}
)

# 4. Create vector store — a direct-client store selected by config.provider
#    (swap MilvusConfig for ChromaConfig/PGVectorConfig to change backend)
vector_store = get_vector_store(
    embedding_model=embedding_model,
    config=MilvusConfig.from_env(),
)

# 5. Create chunker
chunker = LangChainChunker(
    method="recursive",
    chunk_size=512,
    chunk_overlap=128
)

# 6. Create retriever
retriever = Retriever(
    vector_store=vector_store,
    number_of_chunks=5,
    method="simple",
    search_mode="hybrid",
    ranker_strategy="rrf",
    ranker_k=60
)

# 7. Create RAG template
rag = SimpleRAG(
    foundation_model=foundation_model,
    retriever=retriever,
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store
)

# 8. Index documents
rag.build_index(documents)

# 9. Generate answer
result = rag.generate("What is X?")
print(result["answer"])
```

---

## Extension Points

All RAG components are designed for extensibility:

### Custom Foundation Model

```python
class CustomFoundationModel(BaseFoundationModel[MyClient, MyParams]):
    def chat(self, messages: list[MessageTyped], **kwargs) -> list[MessageTyped]:
        # Your implementation
        pass
```

### Custom Embedding Model

```python
class CustomEmbeddingModel(BaseEmbeddingModel[MyClient, MyParams]):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        pass

    def embed_query(self, query: str) -> list[float]:
        # Your implementation
        pass
```

### Custom Vector Store

```python
class CustomVectorStore(BaseVectorStore):
    def search(self, query: str, k: int, **kwargs) -> list[AI4RAGChunk]:
        # Your implementation
        pass

    def add_documents(self, documents: Sequence[AI4RAGChunk]) -> None:
        # Your implementation
        pass

    @property
    def collection_name(self) -> str:
        return self._collection_name
```

### Custom RAG Template

```python
class CustomRAG(BaseRAGTemplate):
    def build_index(self, documents: list[DoclingDocument], **kwargs) -> None:
        # Your indexing logic
        pass

    def generate(self, question: str, **kwargs) -> dict[str, Any]:
        # Your generation logic
        pass

    def generate_stream(self, question: str, **kwargs):
        # Your streaming logic
        pass
```

---

## Best Practices

**Foundation Models:**

1. **Customize prompts** for your domain (system_message_text, user_message_text)
2. **Use low temperature** (0.0-0.2) for factual Q&A
3. **Adjust max_completion_tokens** based on expected answer length

**Embedding Models:**

1. **Provide embedding_dimension and context_length** explicitly to avoid auto-detection overhead
2. **Choose models matching your language** (multilingual vs English-only)
3. **Consider embedding dimension** (higher = more expressive but slower/larger)

**Vector Stores:**

1. **Use Milvus or PGVector for production** hybrid search (server-side fusion for Milvus, in-memory fusion for PGVector); Chroma is vector-only
2. **Use ChromaVectorStore** for development/testing (in-memory, simpler setup)
3. **Enable hybrid search** for keyword-heavy domains (technical docs, legal, medical) — not supported on Chroma
4. **Tune ranker parameters** (ranker_k, ranker_alpha) via optimization

**Chunking:**

1. **Smaller chunks** (256-512) for precise Q&A
2. **Larger chunks** (1024-2048) for broader context
3. **Adjust chunk_overlap** (25-50% of chunk_size) to maintain coherence
4. **Ensure chunk_size < embedding context_length**

**Retrieval:**

1. **Start with simple retrieval** before trying window-based
2. **Use hybrid search** when semantic search misses exact matches
3. **Tune number_of_chunks** (5-10 typical) via optimization
4. **Monitor retrieval quality** via context_correctness metric

---

## Next Steps

- [Core Components](core-components.md) - Experiment engine and HPO details
- [Data Flow](data-flow.md) - Detailed workflow analysis
- [Architecture Overview](overview.md) - High-level design
