# Quick Start

This guide walks you through running your first RAG optimization experiment with ai4rag.

---

## Prerequisites

Before starting, ensure you have:

- [x] Installed ai4rag ([Installation Guide](installation.md))
- [x] A running Llama Stack server with models configured
- [x] Environment variables set (`BASE_URL`, `APIKEY`)

---

## Step-by-Step Guide

### 1. Prepare Llama Stack Client

Create a client instance to connect to your Llama Stack server:

```python
import os
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient

load_dotenv()

client = LlamaStackClient(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("APIKEY")
)
```

---

### 2. Prepare Knowledge Base Documents

Load your knowledge base documents from a local directory:

```python
from pathlib import Path
from dev_utils.file_store import FileStore

# Path to your documents folder
documents_path = Path("path/to/your/documents")

# Load documents (supports PDF, HTML, TXT, MD, etc.)
documents = FileStore(documents_path).load_as_documents()

print(f"Loaded {len(documents)} documents")
```

!!! info "Document Format"
    Documents must include a `document_id` in their metadata. `FileStore` handles this automatically.

---

### 3. Prepare Benchmark Data

Create a `benchmark_data.json` file with questions and ground truth answers:

```json
[
  {
    "question": "What is the main purpose of ai4rag?",
    "correct_answers": [
      "ai4rag optimizes RAG templates using hyperparameter optimization",
      "ai4rag finds optimal RAG configurations"
    ],
    "correct_answer_document_ids": ["doc_001", "doc_002"]
  },
  {
    "question": "Which vector databases are supported?",
    "correct_answers": [
      "Milvus and ChromaDB are supported"
    ],
    "correct_answer_document_ids": ["doc_005"]
  }
]
```

Load the benchmark data:

```python
from dev_utils.utils import read_benchmark_from_json

benchmark_data_path = Path("path/to/benchmark_data.json")
benchmark_data = read_benchmark_from_json(benchmark_data_path)
```

!!! tip "Benchmark Quality"
    High-quality benchmark data is crucial for meaningful optimization. Ensure questions are based on your knowledge base and answers are accurate.

---

### 4. Define Search Space

Specify which parameters to optimize and their possible values:

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel

search_space = AI4RAGSearchSpace(
    params=[
        # Foundation model for generation
        Parameter(
            name="foundation_model",
            param_type="C",  # Categorical
            values=[
                LSFoundationModel(
                    model_id="ollama/llama3.2:3b",
                    client=client
                )
            ],
        ),
        # Embedding model
        Parameter(
            name="embedding_model",
            param_type="C",  # Categorical
            values=[
                LSEmbeddingModel(
                    model_id="ollama/nomic-embed-text:latest",
                    client=client,
                    params={
                        "embedding_dimension": 768,
                        "context_length": 8192
                    },
                )
            ],
        ),
        # Chunking parameters
        Parameter(
            name="chunk_size",
            param_type="I",  # Integer
            values=[200, 400, 800, 1000],
        ),
        Parameter(
            name="chunk_overlap",
            param_type="I",  # Integer
            values=[0, 50, 100, 200],
        ),
        # Retrieval parameters
        Parameter(
            name="retrieval_method",
            param_type="C",  # Categorical
            values=["simple", "window"],
        ),
        Parameter(
            name="number_of_chunks",
            param_type="I",  # Integer
            values=[3, 5, 7, 10],
        ),
    ]
)
```

!!! note "Parameter Types"
    - **C** (Categorical): Discrete choices (models, methods)
    - **I** (Integer): Integer values (chunk sizes, top-k)
    - **F** (Float): Continuous values (thresholds, weights)

---

### 5. Configure Optimizer

Set up the hyperparameter optimization algorithm:

```python
from ai4rag.core.hpo.gam_opt import GAMOptSettings

optimizer_settings = GAMOptSettings(
    max_evals=10,      # Total number of configurations to evaluate
    n_random_nodes=4   # Number of random explorations before using GAM
)
```

!!! tip "Optimization Strategy"
    - **Random phase** (`n_random_nodes`): Explores the search space randomly
    - **GAM phase**: Uses a model to suggest promising configurations
    - Start with 10-20 evaluations for initial experiments

---

### 6. Run the Experiment

Create and run the optimization experiment:

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from dev_utils.local_event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="ls_milvus",  # or "chroma" for in-memory
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(),  # Tracks progress
    output_path="./results",  # Where to save results
)

# Run optimization
best_pattern = experiment.search()

print(f"Best RAG Pattern: {best_pattern}")
```

---

### 7. Review Results

After completion, check the `output_path` directory for:

- **CSV files**: Detailed results for each evaluated configuration
- **JSON artifacts**: Best configuration parameters
- **Logs**: Event handler output and experiment progress

```python
import pandas as pd

# Load results
results = pd.read_csv("./results/experiment_results.csv")
print(results.head())

# View best configuration
print(f"\nBest Configuration:")
print(f"  Foundation Model: {best_pattern['foundation_model']}")
print(f"  Chunk Size: {best_pattern['chunk_size']}")
print(f"  Retrieval Method: {best_pattern['retrieval_method']}")
print(f"  Score: {best_pattern['score']}")
```

---

## Complete Example

Here's the full code in one place:

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.core.hpo.gam_opt import GAMOptSettings

from dev_utils.file_store import FileStore
from dev_utils.utils import read_benchmark_from_json
from dev_utils.local_event_handler import LocalEventHandler

# 1. Setup client
load_dotenv()
client = LlamaStackClient(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("APIKEY")
)

# 2. Load documents
documents = FileStore(Path("./knowledge_base")).load_as_documents()

# 3. Load benchmark data
benchmark_data = read_benchmark_from_json(Path("./benchmark_data.json"))

# 4. Define search space
search_space = AI4RAGSearchSpace(
    params=[
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[LSFoundationModel(model_id="ollama/llama3.2:3b", client=client)],
        ),
        Parameter(
            name="embedding_model",
            param_type="C",
            values=[
                LSEmbeddingModel(
                    model_id="ollama/nomic-embed-text:latest",
                    client=client,
                    params={"embedding_dimension": 768, "context_length": 8192},
                )
            ],
        ),
        Parameter(name="chunk_size", param_type="I", values=[200, 400, 800]),
        Parameter(name="chunk_overlap", param_type="I", values=[0, 50, 100]),
        Parameter(name="retrieval_method", param_type="C", values=["simple", "window"]),
        Parameter(name="number_of_chunks", param_type="I", values=[3, 5, 7]),
    ]
)

# 5. Configure optimizer
optimizer_settings = GAMOptSettings(max_evals=10, n_random_nodes=4)

# 6. Run experiment
experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="ls_milvus",
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(),
    output_path="./results",
)

best_pattern = experiment.search()
print(f"Optimization complete! Best pattern: {best_pattern}")
```

---

## Next Steps

- [Learn about search spaces](../user-guide/search-space.md) - Customize parameter ranges
- [Explore optimizers](../user-guide/optimizers.md) - Fine-tune optimization strategies
- [Understand evaluation](../user-guide/evaluation.md) - Metrics and scoring
- [Custom event handlers](../user-guide/event-handlers.md) - Track experiments in production

---

## Common Issues

??? question "Vector store connection errors"
    Ensure your Llama Stack server has Milvus properly configured. Alternatively, use `vector_store_type="chroma"` for in-memory testing.

??? question "Out of memory errors"
    Reduce `max_evals` or simplify your search space. Process documents in smaller batches.

??? question "Poor optimization results"
    - Verify benchmark data quality
    - Expand search space with more parameter options
    - Increase `max_evals` for more thorough exploration
