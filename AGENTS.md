# ai4rag — RAG Templates Optimization Engine

## Build & Test Commands

```bash
pip install -e ".[dev]"                          # Install with all dev dependencies

# Tests
pytest tests/unit/                               # Unit tests (fast, no external deps)
pytest tests/unit/path/to/test_file.py           # Single test file
pytest tests/unit/ --cov=ai4rag --cov-report=term --cov-fail-under=90  # With coverage (CI target: 90%)
pytest tests/functional/                         # Functional tests (requires live OGX + env vars)

# Formatting & Linting
bash scripts/format.sh                           # Run isort + black + copyright fix on ai4rag/
bash scripts/format.sh --all                     # Also format tests/ and dev_utils/
black --check --diff ai4rag/                     # Check formatting without modifying
isort --check-only --diff ai4rag/                # Check import order without modifying
pylint ai4rag/                                   # Lint
bash scripts/copyright_check.sh                  # Verify copyright headers

# Docs
mkdocs serve                                     # Local docs at http://127.0.0.1:8000
```

## Key Conventions

- Line length: 120 chars (Black, isort, Pylint all aligned)
- Type hints required for public APIs, use Python 3.12+ union syntax (`str | None`, not `Optional[str]`)
- Docstrings: NumPy format for public APIs
- Every source file needs the Apache 2.0 copyright header — run `bash scripts/copyright_check.sh --fix` to add missing ones
- Commits require DCO sign-off: `git commit -s`
- Squash merge to `main`, one commit per feature/fix

## Architecture

- `ai4rag/core/` — Experiment orchestrator + HPO optimizers (GAM, Random) + Models Pre-Selector
- `ai4rag/search_space/` — Parameter definitions (Categorical/Integer/Real/Boolean) and constraint validation rules
- `ai4rag/rag/` — RAG pipeline components: chunking, embedding, vector store, retrieval, foundation model, template orchestration
- `ai4rag/evaluator/` — Metrics computation via `unitxt` (faithfulness, answer/context correctness)
- `ai4rag/utils/` — Event handlers, validators, constants
- `tests/unit/` — Mirrors source structure, uses mocks. `tests/functional/` — End-to-end with live services

Provider-agnostic design: components use abstract bases (`BaseFoundationModel`, `BaseEmbeddingModel`, `BaseVectorStore`). OGX and OpenAI are the current providers.

## CI Checks (PR)

Static analysis (Black, isort, Pylint, copyright) must pass before unit tests run. Unit tests require 90%+ coverage. PR template is at `.github/pull_request_template.md`.
