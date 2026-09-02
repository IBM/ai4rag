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
> `ai4rag` is **provider-agnostic**. It reaches foundation and embedding models through the stock [`openai`](https://github.com/openai/openai-python) SDK, so any **OpenAI-compatible endpoint** works — a hosted API, a self-managed server (vLLM, TGI, Ollama, …), or an [OpenShift AI Models-as-a-Service (MaaS)](https://www.redhat.com/en/products/ai) deployment, the integration `ai4rag` ships helpers for out of the box. You can also plug in your **own** foundation model, embedding model, or vector store by implementing the matching `Base*` interface.
> To run an experiment you'll need one foundation model and one embedding model (from any of the above), plus a vector store (Chroma, Milvus, or PostgreSQL/pgvector) connected directly via `ai4rag.rag.vector_store`.

## Model providers

`ai4rag` reaches foundation and embedding models through the stock [`openai`](https://github.com/openai/openai-python) SDK, so it works with **any OpenAI-compatible endpoint** — a hosted API, a self-managed server (vLLM, TGI, Ollama, …), or an [OpenShift AI Models-as-a-Service (MaaS)](https://www.redhat.com/en/products/ai) deployment. Prefer something else entirely? Implement `BaseFoundationModel` / `BaseEmbeddingModel` and pass your own models straight into an experiment.

MaaS is the integration `ai4rag` ships helpers for, so the walkthrough below uses it:

- **SDK:** [openai](https://pypi.org/project/openai/) >= 2, < 3 (Python package used by ai4RAG; installs with this project).
- **Deployment:** an OpenShift AI MaaS instance exposing at least one foundation model and one embedding model.
- **Endpoints:** MaaS serves **everything from a single OpenAI-compatible endpoint** — `MAAS_BASE_URL` (host-only or `/v1`-suffixed; normalized automatically). One client lists the available models (`models.list()`) and serves chat/completions and embeddings for all of them. Model ids are used verbatim, exactly as `models.list()` reports them.

**Features used by ai4rag**

When using the MaaS backend, ai4rag relies on:

- **Embeddings** — Text embeddings via the `embeddings` endpoint (e.g. for indexing and query encoding). Because `models.list()` carries no metadata, embedding dimension and context length are auto-detected at construction (or supplied via `params`).
- **Chat / completions** — Foundation model integration for answer generation when evaluating RAG patterns.

Vector storage is independent of MaaS: `ai4rag` connects directly to Chroma, Milvus, or PostgreSQL/pgvector via the config classes in `ai4rag.rag.vector_store` (see [Vector stores](#vector-stores) below).

## Vector stores

ai4RAG talks to the vector store directly through provider-specific clients — no MaaS deployment is required for this part. Pick a provider and pass its config to `AI4RAGExperiment` as `vector_store_config`:

- **`ChromaConfig`** — Chroma. Ephemeral in-memory by default; persistent (via `persist_directory`) or client/server (via `host`/`port`) modes are also supported. Vector-only search.
- **`MilvusConfig`** — Milvus. Requires a `uri`; supports TLS (`https://` scheme) and self-signed CAs via `server_cert`. Hybrid search (dense + BM25).
- **`PGVectorConfig`** — PostgreSQL with the `pgvector` extension. Hybrid search (dense + `tsvector` full-text).

Each config is a frozen dataclass with a `.from_env()` constructor and an `env_vars` attribute listing the environment variables it reads (e.g. `MILVUS_URI`, `PGVECTOR_HOST`).

## Document processing

ai4RAG uses [`docling-core`](https://github.com/docling-project/docling-core) for document representation and chunking. Documents are represented as `DoclingDocument` instances, and the `DoclingChunker` leverages docling's `HybridChunker` for structure-aware, token-aware chunking. `docling-core`, `openai`, and the vector store clients (`chromadb`, `pymilvus`, `pgvector`, `psycopg`) are all installed automatically with `ai4rag`.


## Quick start
1. [Prepare a MaaS client to integrate with your models.](#prepare-the-maas-client)
2. [Prepare your knowledge base documents for the experiment.](#prepare-knowledge-base-documents)
3. [Prepare `benchmark_data.json` with evaluation questions and answers.](#prepare-benchmark_datajson)
4. [Define and constrain your search space.](#define-and-constrain-search-space)
5. [Configure the optimizer.](#configure-optimizer)
6. [Create and run the experiment.](#run-the-experiment)


### Prepare the MaaS client
To enable full integration with MaaS, build a single client that lists the available models and serves them all — `ai4rag` reuses it for every foundation and embedding model wrapper.
The `dev_utils` helper `create_dev_maas_client()` reads `MAAS_BASE_URL` / `MAAS_API_KEY` and builds that client for you.

> [!tip]
> Store your credentials securely in a `.env` file.

```python
from dotenv import load_dotenv, find_dotenv
from dev_utils.utils import create_dev_maas_client

load_dotenv(find_dotenv())

client = create_dev_maas_client()  # reads MAAS_BASE_URL / MAAS_API_KEY
```

> [!note]
> `dev_utils` is only available when cloning the repository. For the equivalent setup using the
> public API (the single `OpenAI` client built with `create_maas_client`),
> see the [Provider-Agnostic Design](https://ibm.github.io/ai4rag/latest/user-guide/provider-agnostic/) guide.

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
		"correct_answer_document_keys": ["<list of documents ids based on which correct answers were generated>"]
	},
	{
		"question": "<question_2>",
		"correct_answers": [
			"<answer 1 for question 2>",
			"<answer 2 for question 2>"
		],
		"correct_answer_document_keys": ["<list of documents ids based on which correct answers were generated>"]
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
from dev_utils.utils import build_maas_model


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
> To validate model IDs and build a search space from a MaaS deployment in one call, use `prepare_search_space_with_maas()` from `ai4rag.search_space.prepare`, passing the MaaS client and the foundation/embedding model IDs per type.


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
> Select the vector store by passing a `vector_store_config` to `AI4RAGExperiment`:
> `ChromaConfig()` for a zero-config in-memory store (vector-only search), or
> `MilvusConfig.from_env()` / `PGVectorConfig.from_env()` for a server-backed store with hybrid (dense + keyword) search.

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.rag.vector_store import MilvusConfig
from ai4rag.utils.event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_config=MilvusConfig.from_env(),
    optimizer_settings=optimizer_settings,
    event_handler=LocalEventHandler(output_path="<local-path-to-store-your-output-files>"),
)

experiment.search()
best_eval = experiment.results.get_best_evaluations(k=1)[0]
print(best_eval)

print(f"Best pattern: {best_eval.pattern_name} (score: {best_eval.final_score})")
```

> [!note]
> Each trial closes its vector store once it finishes, so `EvaluationResult` no longer exposes a reusable `rag_pattern`. Read the outcome from its fields (`pattern_name`, `final_score`, `scores`, `rag_params`); rebuild the pattern from those settings if you want to run inference.

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
