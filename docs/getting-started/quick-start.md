# Quick Start

This guide walks you through running your first RAG optimization experiment with `ai4rag`.
For the sake of quick-start OpenShift AI Models-as-a-Service (MaaS) will be used, but this can be run with any independently deployed OpenAI-compatible models as long as they are introduced to the experiment with the proper wrapper.

---

## Data loading

To run the experiment you need to provide documents as `DoclingDocument` instances (from the `docling-core` library).
Where those documents come from is up to you — **`ai4rag` does not require object storage.** A local folder is
enough; S3 is only involved if you use the pipeline components that read from a bucket.

Whatever the source, one rule decides whether your experiment scores correctly:

!!! important "A document's key is its `name`"
    Each document's `DoclingDocument.name` is its **key** — the identifier that follows it through chunking,
    indexing and evaluation, and the value your benchmark data must reference in
    `correct_answer_document_keys`. Pick keys that are unique across the corpus: a path relative to your
    documents folder (`manuals/xr-200/setup.pdf`) rather than a bare file name, so that two files sharing a
    basename in different folders stay distinct.

---

---

## Prerequisites

Before starting, ensure you have:

- [x] Installed ai4rag ([Installation Guide](installation.md))
- [x] Access to an OpenShift MaaS deployment (or any OpenAI-compatible models) that can be used for the experiment
- [x] Environment variables set (`MAAS_BASE_URL`, `MAAS_API_KEY`) to communicate with MaaS

---

## Step-by-Step Guide with MaaS

### 1. Prepare the MaaS Client

OpenShift MaaS serves every model from a single OpenAI-compatible endpoint —
`MAAS_BASE_URL`. One client discovers the available models
and is reused, unchanged, to serve every model wrapper.

The `dev_utils` helpers wrap this setup — `create_dev_maas_client()`
reads `MAAS_BASE_URL` / `MAAS_API_KEY` and builds that client, and
`build_maas_model()` checks that a model id is available and binds it
to the correct wrapper on the same client:

```python
from dotenv import load_dotenv, find_dotenv
from dev_utils.utils import create_dev_maas_client

load_dotenv(find_dotenv())

# Single client, used both to discover models and to serve every wrapper.
client = create_dev_maas_client()  # reads MAAS_BASE_URL / MAAS_API_KEY
```

!!! tip "Public API"
    `dev_utils` is only available when cloning the repository. For the equivalent
    setup using the public API (the single `OpenAI` client built with
    `create_maas_client`), see
    [Provider-Agnostic Design](../user-guide/provider-agnostic.md).

---

### 2. Prepare Knowledge Base Documents

Convert a local folder of documents with Docling, naming each one by its path relative to that folder.
No object storage is involved:

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

documents_root = Path("path/to/your/documents")
converter = DocumentConverter()

documents = []
for file_path in sorted(p for p in documents_root.rglob("*") if p.is_file()):
    document = converter.convert(file_path).document
    # The key: unique across the corpus, and what benchmark data references.
    document.name = str(file_path.relative_to(documents_root))
    documents.append(document)

print(f"Loaded {len(documents)} documents: {[d.name for d in documents]}")
```

!!! warning "Always set `name` yourself"
    Docling derives a name from the file stem, dropping both the folder and the extension — `notes.txt`
    becomes `notes`. Two files named `setup.pdf` in different folders would end up sharing a key, and the
    second would displace the first during indexing. Assigning `document.name` explicitly, as above, avoids
    this.

Conversion is the slow part, so save the results and reload them on later runs:

```python
from ai4rag.utils.docling_io import load_docling_documents

extracted_dir = Path("./extracted_text")
for document in documents:
    output_path = extracted_dir / f"{document.name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(output_path)

# On subsequent runs, skip conversion entirely -- names are preserved in the JSON.
documents = load_docling_documents(extracted_dir)
```

!!! tip "Reading from object storage instead"
    If your corpus already lives in a bucket, `discover_documents()` and `extract_text()` do the same job and
    apply the same naming rule, keying each document by its path relative to the discovery prefix. See
    [Pipeline Components](../user-guide/pipeline-components.md).

!!! note "`dev_utils.file_store.FileStore`"
    The repository also ships a `FileStore` helper used by the examples and tests. It is development-only —
    it is excluded from the installed package and it names documents by bare file name, so it is not suitable
    for a corpus with nested folders.

---

### 3. Prepare Benchmark Data

Create a `benchmark_data.json` file with questions and ground truth answers. Each
`correct_answer_document_keys` entry must be the `name` of a document you loaded in step 2:

```json
[
  {
    "question": "How do I set up the XR-200?",
    "correct_answers": [
      "Connect the power supply, then run the setup wizard."
    ],
    "correct_answer_document_keys": ["manuals/xr-200/setup.pdf"]
  },
  {
    "question": "Which vector databases are supported?",
    "correct_answers": [
      "Milvus and ChromaDB are supported."
    ],
    "correct_answer_document_keys": ["overview.md", "reference/stores.md"]
  }
]
```

Load the benchmark data:

```python
import pandas as pd

benchmark_data = pd.read_json("path/to/benchmark_data.json")
```

!!! warning "Keys that match nothing are silently skipped"
    A key that does not equal any document's `name` matches no document. The ground truth for that question
    is simply never retrieved, which lowers your retrieval scores without failing the run. Cross-check the
    two lists before starting a long experiment:

    ```python
    referenced = {key for keys in benchmark_data["correct_answer_document_keys"] for key in keys}
    missing = referenced - {document.name for document in documents}
    assert not missing, f"Benchmark references unknown documents: {sorted(missing)}"
    ```

!!! tip "Benchmark Quality"
    High-quality benchmark data is crucial for meaningful optimization.
    Ensure questions are based on your knowledge base and answers are accurate.

---

### 4. Define Search Space

Specify which parameters to optimize and their possible values:

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from dev_utils.utils import build_maas_model

search_space = AI4RAGSearchSpace(
    params=[
        # Foundation model for generation
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[
                build_maas_model(client, model_id="qwen3-8b-fp8-dynamic", model_type="llm")
            ],
        ),
        # Embedding model
        Parameter(
            name="embedding_model",
            param_type="C",
            values=[
                build_maas_model(
                    client,
                    model_id="bge-m3",
                    model_type="embedding",
                    embedding_params={
                        "embedding_dimension": 1024,
                        "context_length": 8192
                    },
                )
            ],
        ),
        # Chunking parameters
        Parameter(
            name="chunk_size",
            param_type="C",
            values=[200, 400, 800, 1000],
        ),
        Parameter(
            name="chunk_overlap",
            param_type="C",
            values=[0, 50, 100, 200],
        ),
        # Retrieval parameters
        Parameter(
            name="retrieval_method",
            param_type="C",
            values=["simple", "window"],
        ),
        Parameter(
            name="number_of_chunks",
            param_type="C",
            values=[3, 5, 7, 10],
        ),
    ]
)
```

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
    - **Random phase** (`n_random_nodes`): Explores the search space randomly to avoid falling into local minimum (greater value = better solutions space exploration)
    - **GAM phase**: Uses a model to suggest promising configurations

---

### 6. Run the Experiment

!!! note "Choosing a Vector Store"
    The vector store is selected by passing a `vector_store_config` to `AI4RAGExperiment`:

    - `ChromaConfig()` — zero-config, in-memory. Vector-only search (no hybrid/BM25).
    - `MilvusConfig.from_env()` or `MilvusConfig(uri=...)` — a running Milvus server. Supports hybrid search (dense + BM25).
    - `PGVectorConfig.from_env()` or `PGVectorConfig(host=...)` — a running PostgreSQL instance with `pgvector`. Supports hybrid search (dense + full-text).

    All three classes live in `ai4rag.rag.vector_store` and can be built explicitly or from environment variables via `.from_env()`.

Create and run the optimization experiment:

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.rag.vector_store import MilvusConfig
from ai4rag.utils.event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_config=MilvusConfig.from_env(),  # See "Choosing a Vector Store" above
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="<path_to_store_results>"),  # Tracks progress
)

# Run optimization
experiment.search()

best_pattern = experiment.results.get_best_evaluations(k=1)[0]

print(f"Best pattern: {best_pattern.pattern_name}, score: {best_pattern.final_score}")
```

---

### 7. Review Results

After completion, check the `output_path` directory for:

- **JSON files**: Detailed results for each evaluated configuration

---

## Complete Example

Here's the full code in one place:

```python
from pathlib import Path

import pandas as pd
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.rag.vector_store import MilvusConfig
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.utils.event_handler import LocalEventHandler

from dev_utils.utils import build_maas_model, create_dev_maas_client

# 1. Setup client
load_dotenv()
client = create_dev_maas_client()  # reads MAAS_BASE_URL / MAAS_API_KEY

# 2. Load documents from a local folder, keyed by their relative path
documents_root = Path("./knowledge_base")
converter = DocumentConverter()
documents = []
for file_path in sorted(p for p in documents_root.rglob("*") if p.is_file()):
    document = converter.convert(file_path).document
    document.name = str(file_path.relative_to(documents_root))
    documents.append(document)

# 3. Load benchmark data -- its keys must match the document names above
benchmark_data = pd.read_json("./benchmark_data.json")

# 4. Define search space
search_space = AI4RAGSearchSpace(
    params=[
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[build_maas_model(client, model_id="qwen3-8b-fp8-dynamic", model_type="llm")],
        ),
        Parameter(
            name="embedding_model",
            param_type="C",
            values=[
                build_maas_model(
                    client,
                    model_id="bge-m3",
                    model_type="embedding",
                    embedding_params={"embedding_dimension": 1024, "context_length": 8192},
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
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_config=MilvusConfig.from_env(),
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="./results"),
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
