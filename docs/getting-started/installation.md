# Installation

## Requirements

- **Python**: 3.12 or 3.13 (strictly required)
- **Llama Stack Server**: With at least one foundation model, one embedding model, and vector database configured
- **Operating System**: macOS or Linux

---

## Basic Installation

Install ai4rag using pip:

```bash
pip install ai4rag
```

This installs the core package with all required dependencies.

---

## Development Installation

For development work, including testing and code quality tools:

```bash
# Clone the repository
git clone https://github.com/IBM/ai4rag.git
cd ai4rag

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

The `dev` optional dependencies include:

- Testing tools (`pytest`, `pytest-cov`, `pytest-mock`)
- Code quality tools (`black`, `pylint`, `isort`)
- Documentation tools (`mkdocs`, `mkdocs-material`)
- Development utilities (`beautifulsoup4`, `pypdf`, `dotenv`)

---

## Optional Dependencies

You can install specific optional dependency groups:

### Testing Only

```bash
pip install -e ".[test]"
```

Includes: `pytest`, `pytest-cov`, `pytest-mock`, `psutil`, `nbformat`

### Code Quality Only

```bash
pip install -e ".[code_check]"
```

Includes: `pylint`, `black`

### Documentation Only

```bash
pip install -e ".[docs]"
```

Includes: `mkdocs`, `mkdocs-material`, `mkdocstrings`, and related plugins

---

## Llama Stack Setup

ai4rag requires a running Llama Stack server. Follow these steps:

### 1. Install Llama Stack

```bash
pip install llama-stack
```

### 2. Configure Your Stack

Create a Llama Stack configuration with:

- At least one **foundation model** (e.g., `ollama/llama3.2:3b`)
- At least one **embedding model** (e.g., `ollama/nomic-embed-text:latest`)
- A **vector database** (e.g., Milvus or ChromaDB)

Refer to the [Llama Stack documentation](https://llamastack.github.io/docs/) for detailed setup instructions.

### 3. Start the Server

```bash
llama-stack run <your-config.yaml>
```

Note the server URL and API key for use in ai4rag.

---

## Environment Configuration

Store your Llama Stack credentials securely in a `.env` file:

```bash
# .env
BASE_URL=http://localhost:8000
APIKEY=your-api-key-here
```

!!! warning "Security"
    **Never commit your `.env` file to version control.** Add it to `.gitignore`.

Load environment variables in your code:

```python
import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("APIKEY")
```

---

## Verify Installation

Check that ai4rag is installed correctly:

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
print(f"Available models: {[m.identifier for m in models]}")
```

---

## Next Steps

- [Quick Start Guide](quick-start.md) - Run your first optimization
- [Configuration](configuration.md) - Detailed configuration options
- [User Guide](../user-guide/overview.md) - Comprehensive usage documentation

---

## Troubleshooting

### Python Version Issues

If you encounter version conflicts:

```bash
# Check your Python version
python --version

# Use a specific Python version
python3.13 -m pip install ai4rag
```

### Llama Stack Connection Errors

- Verify the server is running: `curl http://localhost:8000/health`
- Check your `BASE_URL` and `APIKEY` in `.env`
- Review Llama Stack logs for errors

### Dependency Conflicts

If you experience dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv ai4rag-env
source ai4rag-env/bin/activate  # On Windows: ai4rag-env\Scripts\activate

# Install ai4rag
pip install ai4rag
```
