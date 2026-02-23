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
- **Docstring Style**: Google format
- **Type Hints**: Required for public APIs
- **License Header**: Required in all source files

## Configuration

See `pyproject.toml` for complete configuration:

- `[tool.black]`: Black settings
- `[tool.isort]`: Import sorting settings
- `[tool.pylint]`: Linting rules

!!! note "Coming Soon"
    Detailed style guide is under development.
