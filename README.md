> [!IMPORTANT]  
> To make the best use of `ai4rag` user is encouraged to use it with `llama-stack`.
> For that server with available models (at least 1 foundation model and 1 embedding model) and vector database needs to be provided.


<div align="center">

<img src="icon.svg" alt="ai4RAG icon" width="80" height="62"/>

# `ai4RAG`
### RAG Templates Optimization Engine

![AI4RAG](https://img.shields.io/badge/AI4RAG-RAG%20Builder%20%26%20Optimizer-0F62FE?style=for-the-badge&logo=ibm&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

[![RAG Builder](https://img.shields.io/badge/🏗️-RAG%20Builder-10B981?style=flat-square)](#)
[![HPO](https://img.shields.io/badge/⚙️-Hyperparameter%20Optimization-F59E0B?style=flat-square)](#)
[![AutoML](https://img.shields.io/badge/🚀-AutoML%20for%20RAG-8B5CF6?style=flat-square)](#)

**Initialises RAG Template with optimal parameters**

[Quick Start](#quick-start) • [Llama Stack](#running-with-llama-stack-server) • [Contribution](#contribution)

</div>

---

## 🎯 What is ai4RAG?

`ai4RAG` is an **optimization engine for RAG Templates** that is LLM and Vector Database provider agnostic. 
It accepts variety of RAG Templates and search space definition.
Returns initialised RAG Template with optimal parameters' values (called RAG Pattern).

## Llama Stack

ai4RAG can run experiments using a [Llama Stack](https://github.com/llamastack/llama-stack) server for embeddings, vector storage, and text generation. Use the official client and API docs to connect and extend:

- **Client:** [llama-stack-client](https://pypi.org/project/llama-stack-client/) (Python package used by ai4RAG; installs with this project).
- **API reference:** [Llama Stack API docs](https://llamastack.github.io/docs/) — HTTP API used by the client.

**Features used by ai4RAG**

When using the Llama Stack backend, ai4RAG relies on:

- **Embeddings** — Text embeddings via the client (e.g. for indexing and query encoding). See [Embeddings API](https://llamastack.github.io/docs/api/embeddings) in the docs.
- **Vector stores** — Create, retrieve, and delete vector store instances (e.g. Milvus) with a chosen embedding model and dimension. See [Vector stores](https://llamastack.github.io/docs/api/creates-a-vector-store) in the API docs.
- **Vector IO** — Insert document chunks (with embeddings) into a store and run similarity search (query) for retrieval. See [Vector IO](https://llamastack.github.io/docs/api/search-for-chunks-in-a-vector-store) and insert/query endpoints.
- **Chat / responses** — Foundation model integration for answer generation (e.g. chat completions or responses API) when evaluating RAG patterns.


## Quick start
1. [Provide instance of `llama-stack-client` to properly utilise `llama-stack`.](#prepare-llama-stack-client)
2. [Prepare your knowledge base to be used in the experiment.](#prepare-knowledge-base-documents)
3. [Prepare `benchmark_data.json`.](#prepare-benchmark_datajson)
4. [Define and constrain your search space.](#define-and-constrain-search-space)
5. [Configure optimiser.](#configure-optimiser)
6. [Prepare and run the experiment.](#run-the-experiment) 


### Prepare `llama-stack-client`
To provide full integration and use `ai4rag` with Llama Stack user needs to instantiate `LlamaStackClient`.
This will allow to base the experiment on tha models and vector stores available under llama stack server.

> [!tip]
> User is encouraged to store all the credential in `.env` file.

```python
import os
from dotenv import load_dotenv, find_dotenv
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))
```

### Prepare knowledge base documents
To run the experiment user needs to prepare set of documents that will be used as a knowledge base for retrieval.
These documents will be used to ground LLM's response.

They should be contained within a local directory.

To read the documents user can utilise `FileStore` class from `dev_utils` module.

> [!note]
> Supported documents formats can be found in the `FileStore` implementation.

```python
from pathlib import Path
from dev_utils.file_store import FileStore

documents_path = Path("<path to the documents folder>")
documents = FileStore(documents_path).load_as_documents()
```


### Prepare `benchmark_data.json`
To prepare for the experiment user needs to provide `benchmark_data.json` following schema:
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

Benchmark records need to be based on the knowledge base documents.

```python
from dev_utils.utils import read_benchmark_from_json

benchmark_data_path = Path("<path to benchmark_data.json>")
benchmark_data = read_benchmark_from_json(benchmark_data_path)
```


### Define and constrain search space
Search space defines possible combinations of parameters where each combination creates a unique RAG Pattern.
During the experiment engine will optimise RAG Pattern for selected metric over the given search space, using objective function to return metric.

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel


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
            ]
        )
    ]
)
```

> [!important]
> As models discovery is under development, user needs to specify what models to use.
> Additionally user may constrain any RAG parameter using similar convention.


### Configure optimiser
User has full control over the optimisation algorithm.
To configure `GAMOptimiser` user may tune `GAMOptSettings`.

```python
from ai4rag.core.hpo.gam_opt import GAMOptSettings

optimiser_settings = GAMOptSettings(
    max_evals=10, n_random_nodes=4
)
```


### Run the experiment
Using information from all previous steps user may now create an experiment and run the ai4rag optimisation engine.

> [!note]
> To use milvus via `llama-stack` user needs to specify `"ls_milvus"` as the `vector_store_type`.
> To use `chroma` in memory user needs to use `"chroma"`.

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from dev_utils.local_event_handler import LocalEventHandler

experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="ls_milvus",
    optimiser_settings=optimiser_settings,
    event_handler=LocalEventHandler(),
    output_path="<path where to store results files>",
)

best = experiment.search()
```

> [!tip]
> Users are encouraged to use their own implementation of `EventHandler` to handle status changes and artifacts produced during the experiment.
> To see more details, please see [`BaseEventHandler implementation`](http://github.com/IBM/ai4rag/blob/main/ai4rag/utils/event_handler/event_handler.py)


## Contribution
Pull requests are very welcome! Make sure your patches are well tested. Ideally create a topic branch for every separate change you make. For example:

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

See more details in [contributing section](contributing.md).





