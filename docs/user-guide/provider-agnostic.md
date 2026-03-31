# Provider-Agnostic Design

`ai4rag` is built on a provider-agnostic architecture that allows you to use any LLM provider, embedding model, and vector database.
This flexibility means you can optimize RAG configurations regardless of your infrastructure choices.

---

## Core Philosophy

Rather than locking you into a specific vendor or technology stack, `ai4rag` defines **abstract interfaces** for the three key components of a RAG system:

1. **Foundation Models** (LLMs for text generation)
2. **Embedding Models** (for document and query embeddings)
3. **Vector Stores** (for storing and retrieving document chunks)

Concrete implementations for different providers (Llama Stack, OpenAI, ChromaDB) all adhere to these interfaces, making them **interchangeable** within the optimization framework.

---

## Supported Providers

### Llama Stack Integration

**What it is**: [Llama Stack](https://github.com/meta-llama/llama-stack) is a unified interface for working with Llama models and associated infrastructure.

**What `ai4rag` supports**:

- **Foundation Models**: Any model configured in your Llama Stack server (Llama 3.x, Mistral, etc.)
- **Embedding Models**: Any embedding model available through Llama Stack
- **Vector Stores**: Any vector database configured in Llama Stack (Milvus, Qdrant, Weaviate, etc.)

**Key advantage**: One client connection gives you access to multiple models and vector stores.

**Usage**:

```python
from llama_stack_client import LlamaStackClient
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel

client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))

foundation_model = LSFoundationModel(model_id="ollama/llama3.2:3b", client=client)
embedding_model = LSEmbeddingModel(
    model_id="ollama/nomic-embed-text:latest",
    client=client,
    params={"embedding_dimension": 768, "context_length": 8192}
)

# Vector store type: "ls_<provider_id>" where provider_id matches Llama Stack config
vector_store_type = "ls_milvus"  # or "ls_qdrant", "ls_weaviate", etc.
```

---

### OpenAI-Compatible APIs

**What it is**: Any API that implements the OpenAI API specification (OpenAI, Azure OpenAI, compatible local servers).

**What `ai4rag` supports**:

- **Foundation Models**: GPT-4, GPT-3.5, GPT-4o, and compatible models
- **Embedding Models**: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002

**Usage**:

```python
from openai import OpenAI
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

foundation_model = OpenAIFoundationModel(
    model_id="gpt-4o-mini",
    client=client,
    params={}
)

embedding_model = OpenAIEmbeddingModel(
    model_id="text-embedding-3-small",
    client=client,
    params={"embedding_dimension": 1536, "context_length": 8191}
)

# Use with Llama Stack vector store or ChromaDB
vector_store_type = "ls_milvus"  # or "chroma"
```

---

### ChromaDB (In-Memory)

**What it is**: An in-memory vector database perfect for development, testing, and small-scale deployments.

**What ai4rag supports**:

- **Vector Store**: ChromaDB for document storage and retrieval

**Key advantage**: No external services required. Great for quick experimentation.

**Limitations**:

- **No hybrid search**: ChromaDB doesn't support sparse embeddings or hybrid retrieval
- **In-memory only**: Data is not persisted between runs (by default)
- **Not for production**: Suitable for development, not large-scale deployments

**Usage**:

```python
# Can use with any foundation/embedding models (Llama Stack, OpenAI, etc.)
experiment = AI4RAGExperiment(
    client=client,  # Llama Stack or OpenAI client
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="chroma",  # In-memory vector store
    optimizer_settings=optimizer_settings,
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

- `LSFoundationModel`: Llama Stack integration
- `OpenAIFoundationModel`: OpenAI API integration

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

- `LSEmbeddingModel`: Llama Stack integration
- `OpenAIEmbeddingModel`: OpenAI API integration

---

### Vector Stores: `BaseVectorStore`

**Interface**:

```python
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore

class BaseVectorStore:
    def __init__(self, embedding_model, distance_metric, reuse_collection_name=None):
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.reuse_collection_name = reuse_collection_name

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the vector store."""

    @abstractmethod
    def search(self, query: str, k: int, **kwargs) -> list[dict]:
        """Search for relevant documents."""

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Return the collection/index name."""
```

**What implementations must provide**:

- `add_documents()`: Index documents with embeddings
- `search()`: Retrieve top-k most relevant documents
- `collection_name`: Unique identifier for the collection/index

**Current implementations**:

- `LSVectorStore`: Any Llama Stack vector database (Milvus, Qdrant, etc.)
- `ChromaVectorStore`: ChromaDB in-memory store

---

## Using Different Providers

The beauty of the provider-agnostic design is that you can **mix and match** components from different providers.

### Example 1: Llama Stack Everything

Use Llama Stack for models and vector store:

```python
from llama_stack_client import LlamaStackClient
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.core.experiment.experiment import AI4RAGExperiment

client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))

experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=AI4RAGSearchSpace(
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[LSFoundationModel(model_id="ollama/llama3.2:3b", client=client)]
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[
                    LSEmbeddingModel(
                        model_id="ollama/nomic-embed-text:latest",
                        client=client,
                        params={"embedding_dimension": 768, "context_length": 8192}
                    )
                ]
            ),
            # ... other params
        ]
    ),
    vector_store_type="ls_milvus",  # Llama Stack Milvus
    optimizer_settings=optimizer_settings,
)
```

---

### Example 2: OpenAI Models with Llama Stack Vector Store

Use OpenAI for generation and embeddings, but Llama Stack for vector storage:

```python
from openai import OpenAI
from llama_stack_client import LlamaStackClient
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.core.experiment.experiment import AI4RAGExperiment

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llama_stack_client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))

experiment = AI4RAGExperiment(
    client=llama_stack_client,  # For vector store access
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=AI4RAGSearchSpace(
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[
                    OpenAIFoundationModel(
                        model_id="gpt-4o-mini",
                        client=openai_client,
                        params={}
                    )
                ]
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[
                    OpenAIEmbeddingModel(
                        model_id="text-embedding-3-small",
                        client=openai_client,
                        params={"embedding_dimension": 1536, "context_length": 8191}
                    )
                ]
            ),
            # ... other params
        ]
    ),
    vector_store_type="ls_qdrant",  # Llama Stack Qdrant
    optimizer_settings=optimizer_settings,
)
```

---

### Example 3: Llama Stack Models with ChromaDB

Use Llama Stack for models, but ChromaDB for quick local development:

```python
from llama_stack_client import LlamaStackClient
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.core.experiment.experiment import AI4RAGExperiment

client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))

experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=AI4RAGSearchSpace(
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[LSFoundationModel(model_id="ollama/llama3.2:3b", client=client)]
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[
                    LSEmbeddingModel(
                        model_id="ollama/nomic-embed-text:latest",
                        client=client,
                        params={"embedding_dimension": 768, "context_length": 8192}
                    )
                ]
            ),
            # ... other params
        ]
    ),
    vector_store_type="chroma",  # In-memory ChromaDB
    optimizer_settings=optimizer_settings,
)
```

!!! warning "No Hybrid Search with ChromaDB"
    Remember that ChromaDB doesn't support hybrid search. If your search space includes `search_mode="hybrid"`, use a Llama Stack vector store instead (e.g., `"ls_milvus"`).

---

### Example 4: Comparing Models Across Providers

Optimize across different foundation models from different providers:

```python
from llama_stack_client import LlamaStackClient
from openai import OpenAI
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel

llama_client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

search_space = AI4RAGSearchSpace(
    params=[
        # Compare Llama models against OpenAI models
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[
                LSFoundationModel(model_id="ollama/llama3.2:3b", client=llama_client),
                LSFoundationModel(model_id="ollama/mistral:7b", client=llama_client),
                OpenAIFoundationModel(model_id="gpt-4o-mini", client=openai_client, params={}),
            ]
        ),
        # Fixed embedding (use Llama Stack)
        Parameter(
            name="embedding_model",
            param_type="C",
            values=[
                LSEmbeddingModel(
                    model_id="ollama/nomic-embed-text:latest",
                    client=llama_client,
                    params={"embedding_dimension": 768, "context_length": 8192}
                )
            ]
        ),
        # ... other params
    ]
)

experiment = AI4RAGExperiment(
    client=llama_client,  # Primary client for vector store
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="ls_milvus",
    optimizer_settings=optimizer_settings,
)
```

---

## ChromaDB for Development

ChromaDB is the fastest way to get started with ai4rag without setting up external services.

### Quick Setup

No configuration needed - just specify `vector_store_type="chroma"`:

```python
from pathlib import Path
from llama_stack_client import LlamaStackClient
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel

from dev_utils.file_store import FileStore
from dev_utils.utils import read_benchmark_from_json

# Load data
client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))
documents = FileStore(Path("./docs")).load_as_documents()
benchmark_data = read_benchmark_from_json(Path("./benchmark.json"))

# Run experiment with ChromaDB (no vector database setup needed!)
experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="chroma",  # In-memory, zero config
    optimizer_settings=optimizer_settings,
)

best_pattern = experiment.search()
```

**When to use ChromaDB**:

- Local development and testing
- Prototyping RAG configurations
- Small document sets (<1000 documents)
- Quick experiments without infrastructure setup

**When NOT to use ChromaDB**:

- Production deployments
- Large document collections (>10,000 documents)
- Hybrid search requirements
- Persistent storage requirements

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
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from langchain_core.documents import Document

class MyCustomVectorStore(BaseVectorStore):
    """Integration with my custom vector database."""

    def __init__(self, embedding_model, distance_metric, reuse_collection_name=None):
        super().__init__(embedding_model, distance_metric, reuse_collection_name)
        self._collection_name = reuse_collection_name or self._generate_collection_name()
        # Initialize your vector store client

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Index documents with embeddings."""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)

        # Insert into your vector database
        self.client.insert(
            collection=self._collection_name,
            vectors=embeddings,
            metadata=[doc.metadata for doc in documents]
        )

    def search(self, query: str, k: int, **kwargs) -> list[dict]:
        """Retrieve top-k similar documents."""
        query_embedding = self.embedding_model.embed_query(query)

        # Query your vector database
        results = self.client.search(
            collection=self._collection_name,
            vector=query_embedding,
            top_k=k
        )

        # Transform to ai4rag format
        return [
            {"page_content": r.text, "metadata": r.metadata}
            for r in results
        ]

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _generate_collection_name(self) -> str:
        """Generate unique collection name."""
        import uuid
        return f"ai4rag_{uuid.uuid4().hex[:8]}"
```

---

## Vector Store Type Naming

When specifying `vector_store_type` in your experiment:

| Pattern | Example | Provider |
|---------|---------|----------|
| `"chroma"` | `"chroma"` | ChromaDB (in-memory) |
| `"ls_<provider_id>"` | `"ls_milvus"` | Llama Stack Milvus |
| `"ls_<provider_id>"` | `"ls_qdrant"` | Llama Stack Qdrant |
| `"ls_<provider_id>"` | `"ls_weaviate"` | Llama Stack Weaviate |

The `<provider_id>` must match the provider configured in your Llama Stack server.

**Example Llama Stack configuration** (excerpt):

```yaml
# In your Llama Stack config
vector_dbs:
  - provider_id: milvus
    config:
      host: localhost
      port: 19530
```

Then use `vector_store_type="ls_milvus"` in `ai4rag`.

---

## Provider Comparison

| Feature | Llama Stack | OpenAI | ChromaDB |
|---------|------------|--------|----------|
| **Foundation Models** | Yes (Llama, Mistral, etc.) | Yes (GPT-4, GPT-3.5) | N/A |
| **Embedding Models** | Yes (any compatible model) | Yes (text-embedding-*) | N/A |
| **Vector Stores** | Yes (Milvus, Qdrant, etc.) | N/A | Yes (in-memory) |
| **Hybrid Search** | Yes (via vector store) | N/A | No |
| **Setup Complexity** | Medium (server required) | Low (API key only) | None |
| **Cost** | Self-hosted (infra cost) | Pay-per-use (API cost) | Free |
| **Best For** | On-prem, self-hosted, Llama models | Quick setup, GPT models | Local dev, testing |

---

## Summary

`ai4rag`'s provider-agnostic design:

- **Abstract base classes**: `BaseFoundationModel`, `BaseEmbeddingModel`, `BaseVectorStore`
- **Mix and match**: Use OpenAI for generation, Llama Stack for embeddings, ChromaDB for storage
- **Extensible**: Add support for new providers by implementing base classes
- **Llama Stack**: Unified access to multiple models and vector stores
- **OpenAI**: Standard API integration for GPT models
- **ChromaDB**: Zero-config in-memory vector store for development

The choice of provider doesn't affect the optimization process - ai4rag works the same regardless of whether you're using Llama 3.2, GPT-4, or a custom model. Focus on finding the best RAG configuration for your use case, not your infrastructure.
