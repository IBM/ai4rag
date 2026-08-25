# Testing

Comprehensive testing practices for `ai4rag` development.

---

## Overview

ai4rag uses a comprehensive testing strategy combining:

- **Unit Tests**: Fast, isolated tests with mocked dependencies
- **Functional Tests**: Integration tests against a live OpenShift MaaS deployment

All tests use **pytest** as the testing framework with additional plugins for mocking and coverage reporting.

### Testing Philosophy

- **Unit tests** validate individual components in isolation using mocks
- **Functional tests** validate end-to-end workflows with real services
- **Coverage** targets >80% for production code
- **Fast feedback** through parallelizable, independent test cases

---

## Test Structure

### Directory Layout

Tests mirror the `ai4rag/` source structure:

```
tests/
├── __init__.py
├── unit/                           # Unit tests with mocks
│   └── ai4rag/
│       ├── core/
│       │   ├── experiment/
│       │   │   ├── test_benchmark_data.py
│       │   │   ├── test_exception_handler.py
│       │   │   ├── test_mps.py
│       │   │   ├── test_results.py
│       │   │   └── test_utils.py
│       │   └── hpo/
│       │       ├── test_base_optimizer.py
│       │       └── test_gam_opt.py
│       ├── evaluator/
│       │   ├── test_base_evaluator.py
│       │   └── test_unitxt_evaluator.py
│       └── search_space/
│           ├── prepare/
│           │   ├── test_maas_utils.py
│           │   └── test_prepare_search_space.py
│           └── src/
│               ├── test_parameter.py
│               └── test_search_space.py
└── functional/                     # Integration tests with live services
    ├── test_experiment.py
    └── test_experiment_mocked_models.py
```

### Test File Naming

- **Prefix**: `test_<module_name>.py`
- **Match source**: `ai4rag/core/hpo/gam_opt.py` → `tests/unit/ai4rag/core/hpo/test_gam_opt.py`

---

## Running Tests

### All Tests

```bash
# Run entire test suite
pytest
```

### Unit Tests Only

```bash
# Run all unit tests (fast, no external dependencies)
pytest tests/unit/
```

### Functional Tests Only

```bash
# Run integration tests (requires MaaS setup)
pytest tests/functional/
```

### Specific Test File

```bash
# Run a single test file
pytest tests/unit/ai4rag/core/hpo/test_gam_opt.py

# Run a specific test class
pytest tests/unit/ai4rag/core/hpo/test_gam_opt.py::TestGAMOptSettings

# Run a specific test method
pytest tests/unit/ai4rag/core/hpo/test_gam_opt.py::TestGAMOptSettings::test_gam_opt_settings_creation_with_defaults
```

### Coverage Reports

```bash
# Run with coverage (HTML report)
pytest --cov=ai4rag --cov-report=html

# View report
open htmlcov/index.html

# Terminal coverage report with missing lines
pytest --cov=ai4rag --cov-report=term-missing
```

### Verbose Output

```bash
# Show detailed test output
pytest -v

# Show even more details (including print statements)
pytest -vv -s
```

---

## Unit Testing Patterns

Unit tests validate individual components in isolation. External dependencies (MaaS client, vector stores, etc.) are mocked using `unittest.mock` and `pytest-mock`.

### Fixtures

Fixtures provide reusable test setup using `@pytest.fixture`:

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_search_space():
    """Create a mock search space with predefined combinations."""
    mock_space = MagicMock(spec=SearchSpace)
    mock_space.combinations = [
        {"param1": "a", "param2": 1},
        {"param1": "b", "param2": 2},
    ]
    mock_space.max_combinations = 2
    return mock_space

@pytest.fixture
def optimizer_settings():
    """Create GAMOptSettings."""
    return GAMOptSettings(max_evals=6, n_random_nodes=3)

def test_gam_optimizer_initialization(mock_search_space, optimizer_settings):
    """Test that GAMOptimizer initializes correctly."""
    objective_func = MagicMock(return_value=0.5)

    optimizer = GAMOptimizer(
        objective_function=objective_func,
        search_space=mock_search_space,
        settings=optimizer_settings,
    )

    assert optimizer.objective_function == objective_func
    assert optimizer._search_space == mock_search_space
```

**Fixture Scopes:**

- `scope="function"` (default): New fixture instance per test function
- `scope="class"`: Shared across all tests in a class
- `scope="module"`: Shared across all tests in a file
- `scope="session"`: Shared across entire test session

### Mocking with MagicMock

Use `MagicMock` with `spec=` for type-safe mocking:

```python
from unittest.mock import MagicMock
from ai4rag.search_space.src.search_space import SearchSpace

# Mock with spec for type safety
mock_space = MagicMock(spec=SearchSpace)
mock_space.combinations = [{"a": 1}, {"a": 2}]
mock_space.max_combinations = 2

# Mock method return values
objective_func = MagicMock(return_value=0.42)
result = objective_func({"param": "value"})
assert result == 0.42
objective_func.assert_called_once_with({"param": "value"})
```

**Mocking with pytest-mock (mocker fixture):**

```python
def test_evaluate_initial_random_nodes(mock_search_space, optimizer_settings, mocker):
    """Test the evaluate_initial_random_nodes method."""
    # Mock module-level function
    mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

    # Mock class with return value
    mock_gam_instance = MagicMock()
    mock_gam_instance.predict.return_value = np.array([0.6, 0.8])
    mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam_instance)
```

### Parametrized Tests

Use `@pytest.mark.parametrize` to run the same test with different inputs:

```python
@pytest.mark.parametrize(
    "combination, expected_value",
    (
        ({"chunk_size": 2048, "chunk_overlap": 512}, True),
        ({"chunk_size": 512, "chunk_overlap": 512}, False),
        ({"chunk_size": 256, "chunk_overlap": 512}, False),
    ),
)
def test_rule_chunk_size_bigger_than_chunk_overlap_returns(combination, expected_value):
    val = _rule_chunk_size_bigger_than_chunk_overlap(combination)
    assert val == expected_value
```

**Benefits:**

- Reduces code duplication
- Clear test cases at a glance
- Each parameter set creates a separate test result

### Exception Testing

Use `pytest.raises` to validate error conditions:

```python
def test_rule_chunk_size_bigger_than_chunk_overlap_raises():
    with pytest.raises(SearchSpaceValueError):
        _ = _rule_chunk_size_bigger_than_chunk_overlap({"chunk_size": 512})

def test_search_all_iterations_failed(mock_search_space, mocker):
    """Test search when all iterations fail."""
    objective_func = MagicMock(side_effect=[FailedIterationError("Failed")] * 10)

    optimizer = GAMOptimizer(
        objective_function=objective_func,
        search_space=mock_search_space,
        settings=GAMOptSettings(max_evals=3, n_random_nodes=3),
    )

    with pytest.raises(OptimizationError) as exc_info:
        optimizer.search()

    assert "All iterations have failed" in str(exc_info.value)
```

### Test Organization

Group related tests using test classes:

```python
class TestGAMOptSettings:
    """Test the GAMOptSettings dataclass."""

    def test_gam_opt_settings_creation_with_defaults(self):
        """Test that GAMOptSettings can be instantiated with default values."""
        settings = GAMOptSettings(max_evals=20)
        assert settings.max_evals == 20
        assert settings.n_random_nodes == 4

    def test_gam_opt_settings_creation_with_custom_values(self):
        """Test that GAMOptSettings can be instantiated with custom values."""
        settings = GAMOptSettings(max_evals=50, n_random_nodes=10)
        assert settings.max_evals == 50
        assert settings.n_random_nodes == 10
```

---

## Functional Tests

Functional tests validate end-to-end workflows with a live OpenShift MaaS deployment. These tests require proper environment setup.

### Environment Requirements

Functional tests require these environment variables:

```bash
# Required for all functional tests
export MAAS_BASE_URL="http://localhost:5001"
export MAAS_API_KEY="your-api-key"

# Required for test data
export AI4RAG_TEST_DATA_PATH="/path/to/test/data"

# Optional: output directory for test artifacts
export AI4RAG_TEST_OUTPUT_PATH="/path/to/test/output"

# Optional: model configuration
# AI4RAG_TEST_FOUNDATION_MODEL / AI4RAG_TEST_EMBEDDING_MODEL must be the model id
# verbatim, exactly as returned by models.list() (may include a "/").
export AI4RAG_TEST_FOUNDATION_MODEL="vllm-inference-llama-3-1/model-id"
export AI4RAG_TEST_EMBEDDING_MODEL="vllm-embedding/embedding-model-id"
export AI4RAG_TEST_EMBEDDING_DIMENSION="768"
export AI4RAG_TEST_EMBEDDING_CONTEXT_LENGTH="512"
```

### Skip Conditions

Tests skip automatically when required environment variables are missing:

```python
pytestmark = pytest.mark.skipif(
    DATA_PATH is None,
    reason="AI4RAG_TEST_DATA_PATH environment variable not set",
)
```

**Running functional tests:**

```bash
# Skip if env vars not set
pytest tests/functional/

# Force skip
pytest tests/functional/ --skip-functional
```

### Module-Scoped Fixtures

Functional tests use module-scoped fixtures for expensive setup:

```python
@pytest.fixture(scope="module")
def client():
    """Create MaaS client (shared across all tests in module)."""
    return create_dev_maas_client()

@pytest.fixture(scope="module")
def documents():
    """Load documents once per module."""
    documents_path = Path(os.path.join(DATA_PATH, "documents"))
    file_store = FileStore(documents_path)
    return file_store.load_as_documents()

@pytest.fixture(scope="module")
def benchmark_data():
    """Load benchmark data once per module."""
    benchmark_data_path = Path(os.path.join(DATA_PATH, "benchmark_data.json"))
    return read_benchmark_from_json(benchmark_data_path)
```

### Example Functional Test

```python
from ai4rag.rag.vector_store import ChromaConfig


class TestExperimentChroma:
    """Run experiment with chroma vector store and MaaS models."""

    def test_experiment_chroma_maas_models(
        self, client, documents, benchmark_data, foundation_model, embedding_model
    ):
        search_space = AI4RAGSearchSpace(
            vector_store_type="chroma",
            params=[
                Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
                Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
            ],
        )

        optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=3)

        experiment = AI4RAGExperiment(
            documents=documents,
            benchmark_data=benchmark_data,
            search_space=search_space,
            optimizer_settings=optimizer_settings,
            event_handler=LocalEventHandler(),
            vector_store_config=ChromaConfig(),
        )

        experiment.search(skip_mps=True)

        assert len(experiment.results) > 0
        best_eval = experiment.results.get_best_evaluations(k=1)[0]
        assert 0 <= best_eval.final_score <= 1
```

---

## Writing New Tests

### Test Naming Conventions

Follow these conventions for clarity:

```python
# Format: test_<feature>_<scenario>
def test_gam_optimizer_initialization():
    """Test that GAMOptimizer initializes correctly."""
    pass

def test_search_successful():
    """Test the search method with successful optimization."""
    pass

def test_search_all_iterations_failed():
    """Test search when all iterations fail."""
    pass

def test_known_observations_validation_missing_score():
    """Error when a known observation is missing the 'score' key."""
    pass
```

### Descriptive Docstrings

Always include docstrings explaining what the test validates:

```python
def test_gam_opt_settings_post_init_keeps_n_random_nodes_if_smaller():
    """Test that __post_init__ keeps n_random_nodes if it's smaller than max_evals."""
    settings = GAMOptSettings(max_evals=20, n_random_nodes=5)
    assert settings.n_random_nodes == 5
```

### One Assertion Per Test (Preferred)

While multiple assertions are sometimes necessary, prefer focused tests:

```python
# Good: Focused test
def test_max_iterations_getter(mock_search_space, optimizer_settings):
    """Test the max_iterations property getter."""
    objective_func = MagicMock(return_value=0.5)
    optimizer = GAMOptimizer(
        objective_function=objective_func,
        search_space=mock_search_space,
        settings=optimizer_settings,
    )
    assert optimizer.max_iterations == 6

# Also Good: Related assertions for the same concept
def test_gam_opt_settings_creation_with_defaults():
    """Test that GAMOptSettings can be instantiated with default values."""
    settings = GAMOptSettings(max_evals=20)
    assert settings.max_evals == 20
    assert settings.n_random_nodes == 4
    assert settings.evals_per_trial == 1
    assert settings.random_state == 64
```

### Test Independence

Tests must be independent and not rely on execution order:

```python
# Bad: Test modifies shared state
shared_list = []

def test_append_first():
    shared_list.append(1)
    assert len(shared_list) == 1

def test_append_second():  # Fails if test_append_first doesn't run first
    assert len(shared_list) == 1

# Good: Each test has independent setup
def test_append_creates_list():
    my_list = []
    my_list.append(1)
    assert len(my_list) == 1
```

---

## Coverage

### Generating Coverage Reports

```bash
# HTML report
pytest --cov=ai4rag --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=ai4rag --cov-report=term

# Show missing lines
pytest --cov=ai4rag --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=ai4rag --cov-report=xml
```

### Coverage Configuration

Coverage settings are in `pyproject.toml`:

```toml
[tool.coverage.run]
omit = [
    "ai4rag/search_space/prepare/validation_error_decoder.py",
    "ai4rag/core/experiment/experiment.py",
]
```

### Coverage Goals

- **Target**: >80% code coverage
- **Focus**: Cover critical paths, edge cases, and error handling
- **Exclude**: Template code for notebooks, auto-generated code

---

## Best Practices Summary

1. **Write tests first** (TDD) or immediately after implementation
2. **Use descriptive names** for test functions and classes
3. **Mock external dependencies** in unit tests
4. **Use fixtures** for common setup
5. **Parametrize** tests to reduce duplication
6. **Test error conditions** with `pytest.raises`
7. **Keep tests fast** by using mocks in unit tests
8. **Make tests independent** - no shared state
9. **Add docstrings** explaining what each test validates
10. **Maintain high coverage** (>80%) with focus on critical paths

---

## Troubleshooting

### Tests Fail with "Environment variable not set"

Functional tests require environment setup. Set required variables or skip functional tests:

```bash
pytest tests/unit/  # Skip functional tests
```

### Mocking Not Working

Ensure mock patches use the correct import path (where the object is used, not defined):

```python
# Bad: Patching where defined
mocker.patch("random.shuffle")

# Good: Patching where imported and used
mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")
```

### Coverage Too Low

Check what's missing:

```bash
pytest --cov=ai4rag --cov-report=term-missing
```

Look for untested edge cases, error paths, and optional parameters.

---

## Next Steps

- [Contributing Guide](contributing.md) - Contribution workflow
- [Code Style](code-style.md) - Formatting and style rules
- [Development Workflow](workflow.md) - Branch and release process
