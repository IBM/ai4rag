# Configuration

This guide covers all configuration options for ai4rag experiments.

---

## Environment Variables

ai4rag uses environment variables for credentials and runtime configuration.

### Llama Stack Connection

```bash
# .env
BASE_URL=http://localhost:8000
APIKEY=your-api-key-here
```

Load in your code:

```python
import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("APIKEY")
```

### Logging Level

Control log verbosity:

```bash
# .env
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
```

```python
# Logging is configured automatically when importing ai4rag
import ai4rag

# Logs will use the LOG_LEVEL from environment
```

---

## Experiment Configuration

### AI4RAGExperiment Parameters

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment

experiment = AI4RAGExperiment(
    client=llama_stack_client,        # LlamaStackClient instance (required)
    documents=documents,                # List of documents (required)
    benchmark_data=benchmark_data,      # List of benchmark records (required)
    search_space=search_space,          # AI4RAGSearchSpace instance (required)
    vector_store_type="ls_milvus",      # Vector store type (required)
    optimizer_settings=optimizer_settings,  # HPO settings (required)
    event_handler=event_handler,        # Event handler instance (optional)
    output_path="./results",            # Results output directory (optional)
)
```

#### Parameters Explained

| Parameter            | Type                | Required | Description                               |
|----------------------|---------------------|---------|-------------------------------------------|
| `client`             | `LlamaStackClient`  | Yes     | Llama Stack client for API access         |
| `documents`          | `List[Document]`    | Yes     | Knowledge base documents                  |
| `benchmark_data`     | `List[dict]`        | Yes     | Evaluation questions and answers          |
| `search_space`       | `AI4RAGSearchSpace` | Yes     | Parameter search space                    |
| `vector_store_type`  | `str`               | Yes     | `"ls_milvus"` or `"chroma"`               |
| `optimizer_settings` | `OptSettings`       | Yes     | HPO algorithm configuration               |
| `event_handler`      | `BaseEventHandler`  | No      | Custom event handler                      |
| `output_path`        | `str`               | No      | Results directory (default: current dir)  |

---

## Search Space Configuration

### Parameter Types

ai4rag supports three parameter types:

#### Categorical (C)

Discrete choices from a fixed set:

```python
Parameter(
    name="retrieval_method",
    param_type="C",
    values=["simple", "window", "hybrid"]
)
```

#### Integer (I)

Integer values within a range:

```python
Parameter(
    name="chunk_size",
    param_type="I",
    values=[100, 200, 400, 800, 1000]
)
```

#### Float (F)

Continuous numerical values:

```python
Parameter(
    name="temperature",
    param_type="F",
    values=[0.0, 0.3, 0.5, 0.7, 1.0]
)
```

### Model Configuration

#### Foundation Models

```python
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel

Parameter(
    name="foundation_model",
    param_type="C",
    values=[
        LSFoundationModel(
            model_id="ollama/llama3.2:3b",
            client=client
        ),
        LSFoundationModel(
            model_id="ollama/llama3.1:8b",
            client=client
        ),
    ]
)
```

#### Embedding Models

```python
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel

Parameter(
    name="embedding_model",
    param_type="C",
    values=[
        LSEmbeddingModel(
            model_id="ollama/nomic-embed-text:latest",
            client=client,
            params={
                "embedding_dimension": 768,
                "context_length": 8192
            }
        ),
        LSEmbeddingModel(
            model_id="ollama/mxbai-embed-large:latest",
            client=client,
            params={
                "embedding_dimension": 1024,
                "context_length": 512
            }
        ),
    ]
)
```

### Validation Rules

Search spaces include built-in validation:

```python
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

search_space = AI4RAGSearchSpace(
    params=[...],
    rules=[
        # Custom validation rules
        lambda config: config["chunk_size"] > config["chunk_overlap"],
        lambda config: config["number_of_chunks"] <= 10,
    ]
)
```

**Built-in Rules:**

- `chunk_size > chunk_overlap` (always enforced)
- `window_size == 0 ⟺ retrieval_method == "simple"`
- `window_size > 0 ⟺ retrieval_method == "window"`

---

## Optimizer Configuration

### GAM Optimizer

Recommended for most use cases:

```python
from ai4rag.core.hpo.gam_opt import GAMOptSettings

settings = GAMOptSettings(
    max_evals=50,       # Total evaluations
    n_random_nodes=10   # Random exploration phase
)
```

**Parameters:**

- **`max_evals`**: Total number of configurations to evaluate
    - Start with 10-20 for quick experiments
    - Use 50-100 for thorough optimization
    - Each evaluation runs the full RAG pipeline

- **`n_random_nodes`**: Random explorations before using GAM model
    - Recommended: 20-30% of `max_evals`
    - Too low: May get stuck in local optima
    - Too high: Reduces benefit of intelligent search

### Random Optimizer

Baseline random search:

```python
from ai4rag.core.hpo.random_opt import RandomOptSettings

settings = RandomOptSettings(
    max_evals=50
)
```

Use for:

- Baseline comparisons
- Small search spaces
- Initial exploration

---

## Vector Store Configuration

### Milvus (via Llama Stack)

Production-ready persistent storage:

```python
experiment = AI4RAGExperiment(
    vector_store_type="ls_milvus",
    # ... other params
)
```

**Requirements:**

- Milvus configured in Llama Stack
- Sufficient storage for embeddings
- Network connectivity to Milvus instance

### ChromaDB

In-memory storage for testing:

```python
experiment = AI4RAGExperiment(
    vector_store_type="chroma",
    # ... other params
)
```

**Characteristics:**

- No persistence between runs
- Faster for small datasets
- Lower resource requirements
- Not recommended for production

---

## Evaluation Configuration

### Metrics

ai4rag uses unitxt-based metrics:

- **Faithfulness**: How grounded is the answer in retrieved context?
- **Answer Correctness**: How correct is the answer vs. ground truth?
- **Context Correctness**: How relevant are retrieved documents?

Metrics are automatically computed during evaluation. No additional configuration required.

### Benchmark Data Schema

```json
[
  {
    "question": "string",
    "correct_answers": ["string", "string"],
    "correct_answer_document_ids": ["string", "string"]
  }
]
```

**Requirements:**

- Questions must be answerable from your knowledge base
- Provide multiple correct answer variations
- Include source document IDs for context evaluation

---

## Event Handler Configuration

### Using Built-in Handler

```python
from dev_utils.local_event_handler import LocalEventHandler

handler = LocalEventHandler()

experiment = AI4RAGExperiment(
    event_handler=handler,
    # ... other params
)
```

### Custom Event Handler

Implement `BaseEventHandler` for custom tracking:

```python
from ai4rag.utils.event_handler.event_handler import BaseEventHandler

class MyEventHandler(BaseEventHandler):
    def on_experiment_start(self, experiment_id: str):
        print(f"Starting experiment: {experiment_id}")

    def on_iteration_complete(self, iteration: int, score: float):
        print(f"Iteration {iteration} complete. Score: {score}")

    def on_experiment_complete(self, best_config: dict):
        print(f"Best configuration: {best_config}")

    def on_error(self, error: Exception):
        print(f"Error occurred: {error}")

handler = MyEventHandler()
```

---

## Output Configuration

### Output Path

Specify where results are saved:

```python
experiment = AI4RAGExperiment(
    output_path="./experiments/run_001",
    # ... other params
)
```

### Output Files

After running, the output directory contains:

```
./experiments/run_001/
├── experiment_results.csv       # All evaluated configurations
├── best_configuration.json      # Optimal RAG pattern
├── search_history.json          # HPO search history
└── evaluation_details.json      # Detailed metrics
```

---

## Advanced Configuration

### Chunking Strategy

Customize document chunking:

```python
from ai4rag.rag.chunking.langchain import LangChainChunker

# Chunking is configured via search space parameters
Parameter(name="chunk_size", param_type="I", values=[200, 400, 800]),
Parameter(name="chunk_overlap", param_type="I", values=[0, 50, 100, 200]),
```

### Retrieval Strategy

Configure retrieval methods:

```python
Parameter(
    name="retrieval_method",
    param_type="C",
    values=["simple", "window"]
),
Parameter(
    name="window_size",
    param_type="I",
    values=[0, 1, 2, 3]  # 0 for simple, >0 for window
),
Parameter(
    name="number_of_chunks",
    param_type="I",
    values=[3, 5, 7, 10]  # Top-k retrieval
),
```

---

## Configuration Best Practices

1. **Start Small**: Begin with a constrained search space and few evaluations
2. **Iterate**: Expand based on initial results
3. **Version Control**: Save configurations alongside results
4. **Document Decisions**: Track why you chose specific parameter ranges
5. **Validate**: Ensure benchmark data quality before optimization
6. **Monitor Resources**: Watch memory and compute usage during runs

---

## Next Steps

- [User Guide: Search Space](../user-guide/search-space.md) - Deep dive into parameter definition
- [User Guide: Optimizers](../user-guide/optimizers.md) - HPO algorithm details
- [User Guide: Event Handlers](../user-guide/event-handlers.md) - Custom tracking
