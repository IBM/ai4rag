# Contributing to ai4rag

Thank you for your interest in contributing to `ai4rag`! This guide will help you get started.

---

## Code of Conduct

We are committed to fostering a welcoming and inclusive community.
Please be respectful and professional in all interactions.

---

## Ways to Contribute

### Report Bugs

Found a bug? Help us fix it:

1. Check [existing issues](https://github.com/IBM/ai4rag/issues) to avoid duplicates
2. Use the bug report template (coming soon)
3. Include:
    - Detailed description
    - Steps to reproduce
    - Expected vs actual behavior
    - Environment details (Python version, OS, etc.)
    - Minimal code example

### Suggest Features

Have an idea for a new feature?
Open an issue and describe it. Proper template is coming soon...

### Improve Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation (future)

### Submit Code

Ready to contribute code? Great!

1. Read the [Development Workflow](workflow.md)
2. Follow our [Code Style](code-style.md) guidelines
3. Write tests for your changes
4. Update documentation as needed

---

## Getting Started

### Prerequisites

- Python `>=3.12,<=3.13`
- [uv](https://docs.astral.sh/uv/) — used for dependency management and running commands
- Git
- GitHub account

### Setup Development Environment

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/ai4rag.git
cd ai4rag

# Add upstream remote
git remote add upstream https://github.com/IBM/ai4rag.git

# Install all development dependencies (uv creates .venv automatically)
uv sync --extra dev

# Verify installation
uv run pytest tests/unit/
```

---

## Development Process

### 1. Find or Create an Issue

- Browse [open issues](https://github.com/IBM/ai4rag/issues)
- Comment on the issue to claim it
- Get feedback from maintainers before starting

### 2. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

Follow these practices:

- **One feature per PR**: Keep changes focused
- **Write tests**: All new code needs tests
- **Update docs**: Document new features
- **Follow style**: Use Black and Pylint
- **Commit often**: Small, logical commits

### 4. Test Your Changes

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ai4rag --cov-report=term-missing

# Check code style
uv run black ai4rag/
uv run pylint ai4rag/

# Build docs
uv run mkdocs build --strict
```

### 5. Commit with Sign-Off

```bash
git add .
git commit -s -m "feat: add your feature

Detailed description of what changed and why.

Signed-off-by: Your Name <your.email@example.com>"
```

!!! important "Developer Certificate of Origin"
    All commits must be signed off (`-s` flag) to indicate you accept the [DCO](https://developercertificate.org/).

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Open PR on GitHub targeting `main` branch
```

---

## Pull Request Guidelines

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines (Black, Pylint)
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] Commits are signed off
- [ ] PR targets `main` branch
- [ ] PR description explains changes clearly

### PR Template

Use this structure:

```markdown
## Description
Brief summary of changes

## Motivation
Why is this change needed?

## Changes
- Bullet point list of changes
- Another change

## Testing
How did you test this?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] All checks passing
```

### Review Process

- Maintainers will review within 3-5 business days
- Address feedback promptly
- Need at least 1 "LGTM" from maintainers
- Once approved, maintainers will merge

---

## Code Style Guidelines

See [Code Style Guide](code-style.md) for detailed formatting rules.

**Quick Summary:**

- **Line Length**: 120 characters
- **Formatter**: Black
- **Linter**: Pylint
- **Imports**: Sorted with isort
- **Docstrings**: Google style
- **Type Hints**: Required for public APIs

---

## Testing Guidelines

See [Testing Guide](testing.md) for comprehensive testing practices.

**Quick Summary:**

- Write unit tests for all new functions/classes
- Write integration tests for component interactions
- Aim for >80% code coverage
- Use pytest fixtures for common setups
- Mock external dependencies

---

## Documentation Guidelines

### Docstring Format

Use NumPy docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed, explaining the function's
    purpose and behavior.

    Parameters
    ----------
    param1 : str
        Description of param1

    param2 : int
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When param2 is negative

    Examples
    --------
    >>> my_function("test", 42)
    True
    """
    pass
```

### User Documentation

When adding features:

1. Update relevant user guide pages
2. Add examples to quick start if appropriate
3. Update API reference (auto-generated from docstrings)
4. Consider adding a tutorial

---

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

All source files must include the license header:

```python
# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
```

---

## Developer Certificate of Origin

We use the [Developer's Certificate of Origin 1.1 (DCO)](https://developercertificate.org/) to manage contributions.

By signing off your commits, you certify that:

- You created the contribution
- You have the right to submit it
- You submit it under the Apache 2.0 license
- You understand it will be publicly available

Sign commits with:

```bash
git commit -s
```

---

## Getting Help

- **Questions**: Open a [discussion](https://github.com/IBM/ai4rag/discussions)
- **Stuck?**: Comment on your PR or issue

---

Thank you for contributing to `ai4rag`!
