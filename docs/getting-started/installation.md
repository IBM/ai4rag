# Installation

## Requirements

- **Python**: 3.12 or 3.13 (strictly required)
- **Operating System**: macOS or Linux
- **(Optional) OGX Server** >= 1.0.0: With at least one foundation model, one embedding model, and vector database configured


!!! note "External models and vector database integration"
    `ai4rag` is designed to be provider-agnostic.
    It means you can use any model from any source as long as it satisfies `BaseFoundationModel` interface.
    The same rule applies to embedding model.
    Custom vector database cannot be explicitly passed to the experiment configuration at this moment, but it can be handled by working the project and delivering custom `VectorStore` implementation.

---

## Basic Installation

Install ai4rag using pip:

```bash
pip install "git+https://github.com/IBM/ai4rag.git@main"
```

This installs the core package with all required dependencies.
Using `"@main"` will download and install latest version of `ai4rag`.
If you want to use specific version, please use e.g. `"@v0.1.1"`

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
- Development utilities (`beautifulsoup4`, `pypdf`, `dotenv`)

To install only a specific subset of dependencies, use the corresponding extra name (see `pyproject.toml`):

```bash
uv sync --extra test        # testing tools only
uv sync --extra code_check  # linting/formatting tools only
uv sync --extra docs        # documentation tools only
uv sync --extra components  # pipeline component support (multiprocessing)
```

### Optional Extras

| Extra | Dependencies | Purpose |
|-------|-------------|---------|
| `components` | `multiprocess` | Full pipeline component support (multiprocessing for text extraction) |
| `test` | `pytest`, `pytest-cov`, `pytest-mock` | Testing tools |
| `code_check` | `black`, `pylint`, `isort` | Code quality tools |
| `docs` | `mkdocs`, `mkdocs-material` | Documentation tools |
| `dev` | All of the above | Full development environment |

---

## OGX Setup

`ai4rag` can be used with OGX server as the foundation models, embedding models and vector database provider.
Follow these steps:

### 1. Install OGX

```bash
pip install "ogx>=1.0.0"
```

### 2. Configure Your Stack

Create an OGX configuration with:

- At least one **foundation model** (e.g., `ollama/llama3.2:3b`)
- At least one **embedding model** (e.g., `ollama/nomic-embed-text:latest`)
- A **vector database** (e.g., Milvus lite or ChromaDB)

Refer to the [OGX documentation](https://ogx-ai.github.io/docs/) for detailed setup instructions.

### 3. Start the Server

```bash
ogx run <your-CONFIG.yaml>
```

Note the server URL and API key for use in `ai4rag`.

---

## Environment Configuration

Store your OGX credentials securely in a `.env` file:

```bash
# .env
BASE_URL="<ogx_server_url>"
APIKEY="<ogx_server_api_key>"
```

!!! warning "Security"
    **Never commit your `.env` file to version control.** Add it to `.gitignore`.

Load environment variables in your code:

```python
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

base_url = os.getenv("BASE_URL")
api_key = os.getenv("APIKEY")
```

---

## Verify Installation

Check that `ai4rag` is installed correctly:

```python
import ai4rag
print(ai4rag.__version__)
```

Test OGX connectivity:

```python
from ogx_client import OgxClient
import os

client = OgxClient(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("APIKEY")
)

# List available models
models = client.models.list()
print(f"Available models: {[m.id for m in models]}")
```

---

## Next Steps

- [Quick Start Guide](quick-start.md) - Run your first optimization
- [User Guide](../user-guide/overview.md) - Comprehensive usage documentation

---
