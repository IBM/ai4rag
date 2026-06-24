<div align="center">

<img src="docs/icon.svg" alt="ai4rag icon" width="80" height="62"/>

# `ai4rag`
### RAG Templates Optimization Engine

![AI4RAG](https://img.shields.io/badge/AI4RAG-RAG%20Builder%20%26%20Optimizer-0F62FE?style=for-the-badge&logo=ibm&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

[![RAG Builder](https://img.shields.io/badge/🏗️-RAG%20Builder-10B981?style=flat-square)](#)
[![HPO](https://img.shields.io/badge/⚙️-Hyperparameter%20Optimization-F59E0B?style=flat-square)](#)
[![AutoML](https://img.shields.io/badge/🚀-AutoML%20for%20RAG-8B5CF6?style=flat-square)](#)

**Initializes RAG Templates with optimal parameters**

[Getting Started](https://ibm.github.io/ai4rag/latest/getting-started/quick-start/) • [User Guide](https://ibm.github.io/ai4rag/latest/user-guide/overview/) • [API Reference](https://ibm.github.io/ai4rag/latest/api-reference/core/experiment/) • [Development](https://ibm.github.io/ai4rag/latest/development/contributing/)

</div>

---

## 🎯 What is ai4RAG?

`ai4RAG` is an **optimization engine for RAG Templates** that is LLM and vector database provider-agnostic.
It accepts a variety of RAG Templates and a search space definition, then returns an initialized RAG Template with optimal parameter values (called a RAG Pattern).


> [!IMPORTANT]
> `ai4rag` is designed to be provider-agnostic: user may provide his own implementation for foundation model, embedding model or vector store and use them for the experiment.
> Out of the box `ai4rag` is designed to work with [OGX](https://github.com/ogx-ai/ogx).
> To use the full capabilities of `ai4rag`, you'll need access to an OGX server configured with at least one foundation model, one embedding model, and a vector database.

## OGX

ai4RAG can run experiments using an [OGX](https://github.com/ogx-ai/ogx) server for embeddings, vector storage, and text generation. Use the official client and API docs to connect and extend:

- **Client:** [ogx-client](https://pypi.org/project/ogx-client/) >= 1.1.0 (Python package used by ai4RAG; installs with this project).
- **Server:** [OGX](https://github.com/ogx-ai/ogx) >= 1.1.0.
- **API reference:** [OGX API docs](https://ogx-ai.github.io/docs/) — HTTP API used by the client.

**Features used by ai4rag**

When using the OGX backend, ai4rag relies on:

- **Embeddings** — Text embeddings via the client (e.g. for indexing and query encoding). See [Embeddings API](https://ogx-ai.github.io/docs/api/embeddings) in the docs.
- **Vector stores** — Create, retrieve, and delete vector store instances (e.g. Milvus) with a chosen embedding model and dimension. See [Vector stores](https://ogx-ai.github.io/docs/api/creates-a-vector-store) in the API docs.
- **Vector IO** — Insert document chunks (with embeddings) into a store and run similarity search (query) for retrieval. See [Vector IO](https://ogx-ai.github.io/docs/api/search-for-chunks-in-a-vector-store) and insert/query endpoints.
- **Chat / responses** — Foundation model integration for answer generation (e.g. chat completions or responses API) when evaluating RAG patterns.

## Document processing

ai4RAG uses [`docling-core`](https://github.com/docling-project/docling-core) for document representation and chunking. Documents are represented as `DoclingDocument` instances, and the `DoclingChunker` leverages docling's `HybridChunker` for structure-aware, token-aware chunking. Both `docling-core` and `ogx-client` are installed automatically with `ai4rag`.


## Quick start
1. [Provide an instance of `ogx-client` to integrate with OGX.](#prepare-ogx-client)
2. [Prepare your knowledge base documents for the experiment.](#prepare-knowledge-base-documents)
3. [Prepare `benchmark_data.json` with evaluation questions and answers.](#prepare-benchmark_datajson)
4. [Define and constrain your search space.](#define-and-constrain-search-space)
5. [Configure the optimizer.](#configure-optimizer)
6. [Create and run the experiment.](#run-the-experiment)


### Prepare `ogx-client`
To enable full integration with OGX, instantiate an `OgxClient`.
This allows `ai4rag` to use the models and vector stores available on your OGX server.

> [!tip]
> Store your credentials securely in a `.env` file.

```python
import os
from dotenv import load_dotenv, find_dotenv
from ogx_client import OgxClient

client = OgxClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
```

### Prepare knowledge base documents
Prepare a set of documents to serve as the knowledge base for retrieval.
Documents are represented as `DoclingDocument` instances (from the [`docling-core`](https://github.com/DS4SD/docling-core) library) and should be stored in a local directory.

> [!note]
> If you are using the project locally, you can load documents using the `FileStore` class from the `dev_utils` module.
> Supported document formats can be found in the `FileStore` implementation.

```python
from pathlib import Path
from dev_utils.file_store import FileStore

documents_path = Path("<path to the documents folder>")
documents = FileStore(documents_path).load_as_documents()
```


### Prepare `benchmark_data.json`
Create a `benchmark_data.json` file following this schema:
```json
[
	{
		"question": "<question_1>",
		"correct_answers": [
			"<answer 1 for question 1>",
			"<answer 2 for question 1>"
		],
		"correct_answer_document_ids": ["<list of documents ids based on which correct answers were generated>"]
	},
	{
		"question": "<question_2>",
		"correct_answers": [
			"<answer 1 for question 2>",
			"<answer 2 for question 2>"
		],
		"correct_answer_document_ids": ["<list of documents ids based on which correct answers were generated>"]
	}
]
```

All benchmark questions and answers must be derived from your knowledge base documents.

```python
from dev_utils.utils import read_benchmark_from_json

benchmark_data_path = Path("<path to benchmark_data.json>")
benchmark_data = read_benchmark_from_json(benchmark_data_path)
```


### Define and constrain search space
The search space defines all possible parameter combinations, where each combination creates a unique RAG Pattern.
During the experiment, the engine will optimize the RAG Pattern for the selected metric over the given search space, using an objective function to evaluate each configuration.

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel


search_space = AI4RAGSearchSpace(
    params=[
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[OGXFoundationModel(model_id="ollama/llama3.2:3b", client=client)],
        ),
        Parameter(
            name="embedding_model",
            param_type="C",
            values=[
                OGXEmbeddingModel(
                    model_id="ollama/nomic-embed-text:latest",
                    client=client,
                    params={"embedding_dimension": 768, "context_length": 8192},
                )
            ],
        ),
        Parameter(
            name="chunking_method",
            param_type="C",
            values=["recursive", "hybrid"],
        ),
        Parameter(
            name="chunk_size",
            param_type="C",
            values=[512, 1024, 2048],
        ),
        Parameter(
            name="chunk_overlap",
            param_type="C",
            values=[0, 128, 256],
        ),
    ]
)
```

> [!tip]
> `chunking_method` controls the chunking strategy: `"recursive"` uses LangChain's `RecursiveCharacterTextSplitter`, while `"hybrid"` uses docling's structure-aware `HybridChunker` (requires `chunk_overlap=0`).
> When omitted, both methods are included by default.

> [!tip]
> To run automatic models discovery with OGX you may use `prepare_search_space_with_ogx()` from `ai4rag.search_space.prepare_search_space`.


### Configure optimizer
You have full control over the optimization algorithm. Configure the `GAMOptimizer` by adjusting `GAMOptSettings`.

```python
from ai4rag.core.hpo.gam_opt import GAMOptSettings

optimizer_settings = GAMOptSettings(
    max_evals=10, n_random_nodes=4
)
```


### Run the experiment
Using the information from the previous steps, create an experiment and run the ai4rag optimization engine.

> [!note]
> For OGX vector stores, use `vector_store_type="ogx"` and specify the provider with
> `ogx_vector_io_provider_id` (e.g., `ogx_vector_io_provider_id="milvus"`, `ogx_vector_io_provider_id="qdrant"`).
> To use ChromaDB in-memory, specify `vector_store_type="chroma"`.

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.utils.event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="ogx",
    ogx_vector_io_provider_id="milvus",
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="<local-path-to-store-your-output-files>"),
)

experiment.search()
best_eval = experiment.results.get_best_evaluations(k=1)[0]
print(best_eval)

print(best_eval.rag_pattern.generate("What ai4rag can be used for?"))
```

> [!tip]
> For production use, implement your own custom `EventHandler` to handle status changes and artifacts produced during the experiment.
> See the [`BaseEventHandler` implementation](http://github.com/IBM/ai4rag/blob/main/ai4rag/utils/event_handler/event_handler.py) for reference.


## Contribution
Pull requests are very welcome! Make sure your patches are well tested. Ideally create a topic branch for every separate change you make.

### Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/IBM/ai4rag.git
cd ai4rag

# Install all development dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/unit/

# Check code style
uv run black --check ai4rag/
uv run pylint ai4rag/

# Build and serve documentation locally
uv run mkdocs serve
```

### Pull request workflow

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -s -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

See more details in [contributing section](contributing.md).
