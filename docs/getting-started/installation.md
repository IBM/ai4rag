# Installation

## Requirements

- **Python**: 3.12 or 3.13 (strictly required)
- **Operating System**: macOS or Linux
- **(Optional) OpenShift MaaS deployment**: With at least one foundation model and one embedding model available, accessed through the `openai` SDK
- **A vector store**: Chroma (in-memory by default, no setup required), or a running Milvus/PostgreSQL (pgvector) instance for hybrid retrieval


!!! note "External models and vector store integration"
    `ai4rag` is designed to be provider-agnostic.
    It means you can use any model from any source as long as it satisfies `BaseFoundationModel` interface.
    The same rule applies to embedding model.
    Vector stores are selected via a typed `vector_store_config` (`ChromaConfig`, `MilvusConfig`, or `PGVectorConfig`) passed directly to the experiment.
    A custom vector store can also be plugged in by delivering your own `BaseVectorStore` implementation.

---

## Basic Installation

Install ai4rag using pip:

```bash
pip install "git+https://github.com/IBM/ai4rag.git@main"
```

This installs the core package with all required dependencies.
Using `"@main"` will download and install latest version of `ai4rag`.
If you want to use specific version, please use e.g. `"@v0.1.1"`

Vector store clients — `chromadb`, `pymilvus`, `pgvector`, and `psycopg` — are core dependencies and install automatically.
No extra step is needed to use Chroma, Milvus, or PostgreSQL/pgvector as a vector store.

---

## Development Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management. Install it first if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and set up the project:

```bash
# Clone the repository
git clone https://github.com/IBM/ai4rag.git
cd ai4rag

# Install all development dependencies (creates .venv automatically)
uv sync --extra dev
```

The `dev` optional dependencies include:

- Testing tools (`pytest`, `pytest-cov`, `pytest-mock`)
- Code quality tools (`black`, `pylint`, `isort`)
- Documentation tools (`mkdocs`, `mkdocs-material`)
- Development utilities (`beautifulsoup4`, `dotenv`, `ipykernel`)

To install only a specific subset of dependencies, use the corresponding extra name (see `pyproject.toml`):

```bash
uv sync --extra test        # testing tools only
uv sync --extra code_check  # linting/formatting tools only
uv sync --extra docs        # documentation tools only
```

### Optional Extras

| Extra | Dependencies | Purpose |
|-------|-------------|---------|
| `test` | `pytest`, `pytest-cov`, `pytest-mock` | Testing tools |
| `code_check` | `black`, `pylint`, `isort` | Code quality tools |
| `docs` | `mkdocs`, `mkdocs-material` | Documentation tools |
| `dev` | All of the above | Full development environment |

---

## MaaS Setup

`ai4rag` uses [OpenShift AI Models-as-a-Service (MaaS)](https://www.redhat.com/en/products/ai)
as the foundation model and embedding model provider, accessed through the stock
[`openai`](https://github.com/openai/openai-python) SDK (installed automatically as a core dependency).
The vector store is configured independently, via direct clients (Chroma, Milvus, or PGVector) — see [Vector Store Setup](#vector-store-setup) below.

### 1. Get Access to a MaaS Deployment

Obtain access to an OpenShift AI MaaS deployment that exposes:

- At least one **foundation model** (e.g., `qwen3-8b-fp8-dynamic`)
- At least one **embedding model** (e.g., `bge-m3`)

MaaS serves **one OpenAI-compatible endpoint per model**, discovered through a shared
`{MAAS_BASE}/maas-api/v1` listing endpoint. No extra package is required — the `openai`
SDK ships with `ai4rag` as a core dependency.

### 2. Note Your Credentials

Record the MaaS base URL and API key for use in `ai4rag`:

- **`MAAS_BASE`** — the deployment base URL (the model-listing endpoint is `{MAAS_BASE}/maas-api/v1`)
- **`MAAS_API_KEY`** — a single API key, reused for the listing client and every per-model client

---

## Vector Store Setup

`ai4rag` connects to the vector store directly — no MaaS deployment is required for this part.
Pick a provider and pass its config to `AI4RAGExperiment` as `vector_store_config`:

| Provider | Config | Hybrid search (dense + keyword) | Setup |
|----------|--------|:---:|-------|
| Chroma | `ChromaConfig` | :material-close: vector-only | None — defaults to an ephemeral in-memory client |
| Milvus | `MilvusConfig` | :material-check: dense + BM25 | Requires a reachable Milvus instance |
| PGVector | `PGVectorConfig` | :material-check: dense + tsvector full-text | Requires a reachable PostgreSQL instance with the `pgvector` extension |

```python
from ai4rag.rag.vector_store import ChromaConfig, MilvusConfig, PGVectorConfig

# Zero-config, in-memory (great for local experimentation)
vector_store_config = ChromaConfig()

# Or build a config from environment variables
vector_store_config = MilvusConfig.from_env()      # reads MILVUS_URI, MILVUS_TOKEN, MILVUS_SERVER_CERT
vector_store_config = PGVectorConfig.from_env()     # reads PGVECTOR_HOST, PGVECTOR_PORT, PGVECTOR_DB, PGVECTOR_USER, PGVECTOR_PASSWORD
```

Each config class exposes the environment variables it reads via its `env_vars` attribute, and can be constructed explicitly instead of from the environment, e.g. `MilvusConfig(uri="https://localhost:19530")`.

---

## MaaS Environment Configuration

Store your MaaS credentials securely in a `.env` file:

```bash
# .env
MAAS_BASE="<maas_deployment_base_url>"
MAAS_API_KEY="<maas_api_key>"
```

!!! warning "Security"
    **Never commit your `.env` file to version control.** Add it to `.gitignore`.

Load environment variables in your code:

```python
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

base_url = os.getenv("MAAS_BASE")
api_key = os.getenv("MAAS_API_KEY")
```

---

## Verify Installation

Check that `ai4rag` is installed correctly:

```python
import ai4rag
print(ai4rag.__version__)
```

Test MaaS connectivity:

```python
import os
from ai4rag.components.utils import create_maas_client

# General client — points at the shared model-listing endpoint.
client = create_maas_client(
    base_url=f"{os.getenv('MAAS_BASE')}/maas-api/v1",
    api_key=os.getenv("MAAS_API_KEY"),
)

# List available models
models = client.models.list().data
print(f"Available models: {[m.id for m in models]}")
```

---

## Next Steps

- [Quick Start Guide](quick-start.md) - Run your first optimization
- [User Guide](../user-guide/overview.md) - Comprehensive usage documentation

---
