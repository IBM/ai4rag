# Code Style Guide

Code formatting and style standards for `ai4rag`.

---

## Formatting Tools

- **Black**: Code formatter (120 character line length)
- **isort**: Import sorting
- **Pylint**: Code linter (Will be required soon)

## Quick Setup

```bash
# Format code
black ai4rag/

# Sort imports
isort ai4rag/

# Check with linter
pylint ai4rag/
```

## Style Requirements

- **Line Length**: 120 characters
- **Python Version**: 3.12 | 3.13
- **Docstring Style**: Numpy format
- **Type Hints**: Required for public APIs
- **License Header**: Required in all source files

## Configuration

See `pyproject.toml` for complete configuration:

- `[tool.black]`: Black settings
- `[tool.isort]`: Import sorting settings
- `[tool.pylint]`: Linting rules

---

## Detailed Style Guide

### NumPy-Style Docstrings

All public functions, classes, and methods must include NumPy-style docstrings:

```python
def my_function(param1: str, param2: int, optional_param: float = 1.0) -> bool:
    """Brief one-line description of the function.

    Longer description if needed, explaining the function's purpose,
    behavior, and any important implementation details.

    Parameters
    ----------
    param1 : str
        Description of param1, including expected format,
        constraints, and purpose.

    param2 : int
        Description of param2. Can span multiple lines
        when additional context is needed.

    optional_param : float, default=1.0
        Description of optional parameters with their default values.

    Returns
    -------
    bool
        Description of the return value, including meaning
        of True/False or structure of returned objects.

    Raises
    ------
    ValueError
        When param2 is negative or param1 is empty.

    SearchSpaceValueError
        When the combination does not pass validation rules.

    Examples
    --------
    >>> my_function("test", 42)
    True
    >>> my_function("example", 10, optional_param=2.5)
    False
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return len(param1) > param2
```

**Key Sections:**

- **Brief description**: Single line summarizing the function
- **Parameters**: Each parameter with type and description
- **Returns**: Return type and meaning
- **Raises**: Exceptions that may be raised
- **Examples**: Usage examples (optional but recommended)

**For Classes:**

```python
class MyClass:
    """Brief description of the class.

    Longer description explaining the class purpose,
    design patterns used, and key functionality.

    Parameters
    ----------
    param1 : str
        Description of constructor parameter.

    param2 : int, optional
        Description of optional constructor parameter.

    Attributes
    ----------
    attribute1 : list[str]
        Description of public attribute.

    attribute2 : dict[str, Any]
        Description of another public attribute.

    Examples
    --------
    >>> obj = MyClass("test", param2=10)
    >>> obj.attribute1
    ['test']
    """

    def __init__(self, param1: str, param2: int = 5):
        self.attribute1 = [param1]
        self.attribute2 = {"value": param2}
```

### Type Hints

Use Python 3.12+ type hints for all public APIs and most internal functions:

```python
from typing import Any, Literal, Sequence
from pathlib import Path
from langchain_core.documents import Document

# Simple types
def process_text(text: str, count: int) -> str:
    return text * count

# Union types (using | syntax)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path("config.json")
    return {}

# Generic types
def chunk_documents(
    documents: list[Document | tuple[str, str]],
    chunk_size: int = 2048
) -> list[Document]:
    return []

# Literal types for enums
def get_chunker(
    method: Literal["recursive", "markdown", "markdown_header"]
) -> BaseChunker:
    return LangChainChunker(method=method)

# Sequences for flexible inputs
def process_metrics(
    metrics: Sequence[str],
    scores: Sequence[float]
) -> dict[str, float]:
    return dict(zip(metrics, scores))

# Complex return types
def search_space_combinations() -> list[dict[str, Any]]:
    return [{"param": "value"}]

# Callable types
from typing import Callable

def optimize(
    objective_fn: Callable[[dict[str, Any]], float],
    iterations: int = 100
) -> dict[str, Any]:
    return {}
```

**Type Annotation Guidelines:**

- Use `list[T]`, `dict[K, V]`, `set[T]` instead of `List`, `Dict`, `Set`
- Use `str | None` instead of `Optional[str]`
- Use `tuple[str, int]` for fixed-length tuples
- Use `Sequence[T]` for flexible input (list or tuple)
- Use `Literal` for fixed string/value choices
- Use `Any` sparingly, prefer specific types

### Import Ordering

Organize imports in three groups separated by blank lines:

```python
# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

# 1. Standard library imports
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

# 2. Third-party imports
import numpy as np
import pandas as pd
from langchain_core.documents import Document
from ogx_client import OgxClient
from pydantic import BaseModel

# 3. Local/project imports
from ai4rag import logger
from ai4rag.core.hpo.base_optimizer import BaseOptimizer
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames
```

**Import Rules:**

- **Sort within groups**: Alphabetically by module name
- **Use isort**: Configured in `pyproject.toml` to enforce this automatically
- **Absolute imports**: Prefer `from ai4rag.core.hpo import X` over relative imports
- **Specific imports**: Import specific names, not entire modules (unless module is the interface)

```bash
# Format imports
isort ai4rag/
```

**isort Configuration (pyproject.toml):**

```toml
[tool.isort]
profile = "black"
py_version = 313
line_length = 120
```
