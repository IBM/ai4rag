# Installation

## Requirements

- **Python**: 3.12 or 3.13 (strictly required)
- **Operating System**: macOS or Linux
- **(Optional) Llama Stack Server** >= 0.6.0: With at least one foundation model, one embedding model, and vector database configured


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

For development work, including testing and code quality tools:

```bash
# Clone the repository
git clone https://github.com/IBM/ai4rag.git
cd ai4rag

# Install in editable mode with dev dependencies
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

The `dev` optional dependencies include:

- Testing tools (`pytest`, `pytest-cov`, `pytest-mock`)
- Code quality tools (`black`, `pylint`, `isort`)
- Documentation tools (`mkdocs`, `mkdocs-material`)
- Development utilities (`beautifulsoup4`, `pypdf`, `dotenv`)


To see what specific requirement groups are available, please look at the `pyproject.toml` file in the project's root folder.

---

## Llama Stack Setup

`ai4rag` can be used with Llama Stack server as the foundation models, embedding models and vector database provider.
Follow these steps:

### 1. Install Llama Stack

```bash
pip install "llama-stack>=0.6.0"
```

### 2. Configure Your Stack

Create a Llama Stack configuration with:

- At least one **foundation model** (e.g., `ollama/llama3.2:3b`)
- At least one **embedding model** (e.g., `ollama/nomic-embed-text:latest`)
- A **vector database** (e.g., Milvus lite or ChromaDB)

Refer to the [Llama Stack documentation](https://llamastack.github.io/docs/) for detailed setup instructions.

### 3. Start the Server

```bash
llama-stack run <your-CONFIG.yaml>
```

Note the server URL and API key for use in `ai4rag`.

---

## Environment Configuration

Store your Llama Stack credentials securely in a `.env` file:

```bash
# .env
BASE_URL="<llama_stack_server_url>"
API_KEY="<llama_stack_server_api_key>"
```

!!! warning "Security"
    **Never commit your `.env` file to version control.** Add it to `.gitignore`.

Load environment variables in your code:

```python
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
```

---

## Verify Installation

Check that `ai4rag` is installed correctly:

```python
import ai4rag
print(ai4rag.__version__)
```

Test Llama Stack connectivity:

```python
from llama_stack_client import LlamaStackClient
import os

client = LlamaStackClient(
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
