# Event Handlers

Event handlers are the bridge between the optimization engine and your application. Every time ai4rag completes an iteration or changes its internal state, it notifies your handler — giving you full visibility into the experiment without coupling your code to the engine's internals.

## Core Concept

`BaseEventHandler` is an abstract class with two methods you must implement:

| Method | When it fires |
|---|---|
| `on_status_change` | At each step of the experiment (indexing, retrieval, evaluation, …) |
| `on_pattern_creation` | After every RAG pattern is fully evaluated |

Your handler is passed to `AI4RAGExperiment` at construction time and called automatically throughout the run.

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.utils.event_handler import BaseEventHandler

experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_config=vector_store_config,
    event_handler=MyEventHandler(),
)
```

---

## `on_status_change`

Called whenever the engine transitions between steps. Use it for progress tracking, logging, or alerting.

```python
def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `level` | `LogLevel` | Severity: `INFO`, `WARNING`, or `ERROR` |
| `message` | `str` | Human-readable description of what is happening |
| `step` | `str | None` | Name of the current experiment step |

---

## `on_pattern_creation`

Called once per evaluated RAG pattern that is published to the event handler. This is where you receive the full evaluation result and can persist, forward, or display it.

!!! note "Warm-start evaluations are not all published"
    When the GAM optimizer's warm-start phase is enabled (see [Optimizers](optimizers.md)), only the best successful warm-start candidate is published as `Pattern1`; the other warm-start evaluations are internal to the optimizer and never reach `on_pattern_creation`.

```python
def on_pattern_creation(
    self,
    payload: PatternPayload,
    evaluation_results: list[EvaluationRecord],
    conversations: list[dict] | None = None,
    **kwargs,
) -> None:
    ...
```

### `payload` — pattern summary

```python
{
    "name": "Pattern1",
    "template_id": "simple_rag",
    "iteration": 0,
    "max_combinations": 24,
    "duration_seconds": 134,
    "evaluation": {
        "metrics": [
            {
                "name": "faithfulness",
                "evaluator": "unitxt",
                "description": "...",
                "scores": {"mean": 0.72, "ci_low": 0.61, "ci_high": 0.83},
            },
            {
                "name": "answer_relevance",
                "evaluator": "judge",
                "description": "...",
                "scores": {"mean": 0.85, "ci_low": 0.78, "ci_high": 0.92},
            },
            {
                "name": "overall_score",
                "evaluator": "custom",
                "description": "...",
                "scores": {"mean": 0.75, "ci_low": 0.65, "ci_high": 0.85},
                "optimization_metric": True,
            },
        ],
    },
    "settings": {
        "chunking":    {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
        "embedding":   {"model_id": "...", "embedding_params": {"embedding_dimension": 768}},
        "retrieval":   {"method": "simple", "number_of_chunks": 5, "search_mode": "vector"},
        "generation":  {"model_id": "...", "temperature": 0.1, "max_completion_tokens": 1024},
        "store_binding": {"provider_type": "milvus_lite", "collection_name": "ai4rag_example"},
    },
}
```

`template_id` identifies the selected template: experiments publish `agentic_rag` for
`AgenticRAG` and `simple_rag` for `SimpleRAG`, including when retrieval uses Neo4j.
Notebook rendering also accepts `agentic_graph_rag` as an alias for `AgenticRAG`.

For Neo4j patterns, `settings.graph_extraction` contains the extraction policy, for example:

```python
{"mode": "free", "max_entities_per_chunk": 3, "max_relationships_per_chunk": 4}
```

This replaces `settings.knowledge_graph`; the published settings no longer contain a
separate extraction-model block. Retrieval controls remain in `settings.retrieval`.

### `evaluation_results` — per-question breakdown

```python
[
    {
        "question": "What is ...?",
        "answer": "According to the document ...",
        "correct_answers": ["The correct answer is ..."],
        "answer_contexts": [
            {"text": "Retrieved chunk text ...", "document_key": "manuals/doc1.pdf"},
        ],
        "metrics": [
            {"name": "faithfulness", "evaluator": "unitxt", "score": 0.71},
            {"name": "answer_relevance", "evaluator": "judge", "score": 0.85},
            {"name": "overall_score", "evaluator": "custom", "score": 0.78},
        ],
    },
    ...
]
```

### `conversations` — agent trajectories

When inference returns conversation records, the experiment passes them separately from
the pattern payload and evaluation results. Each record contains:

| Field | Content |
|---|---|
| `conversation_id` | Benchmark position, such as `case-0000` |
| `question` | Original question |
| `initial_retrieval` | Initial query and retrieved documents, each with text and metadata |
| `messages` | Agent messages, including assistant tool calls and tool responses |
| `tool_execution` | `rewrite_query` and `retriever` input/output records; retrieval records also include new documents |

The selected warm-start pattern retains its trajectories when it is published as `Pattern1`.
Non-agentic publication omits the `conversations` keyword. Custom handlers used with
agentic experiments must accept it explicitly or through `**kwargs`.

---

## Built-in Handlers

Two ready-to-use implementations are provided:

**`LocalEventHandler`** — saves `pattern.json` and `evaluation_results.json` under
`<output_path>/<pattern_name>/`. It also writes `conversations.json` when trajectories
are present. Ideal for local development.

```python
from ai4rag.utils.event_handler import LocalEventHandler

handler = LocalEventHandler(output_path="./results")
```

**`KFPEventHandler`** — accumulates all status changes and pattern results in memory
(`.status_changes` and `.patterns` lists). Each pattern entry includes `payload`,
`evaluation_results`, and optional `conversations`. Designed for Kubeflow Pipelines
components where you need to access results after the experiment finishes.

```python
from ai4rag.utils.event_handler import KFPEventHandler

handler = KFPEventHandler()
experiment.search()  # construct the experiment with event_handler=handler
print(handler.patterns)  # access after the run
```

---

## Writing a Custom Handler

Subclass `BaseEventHandler` and implement both methods. The example below forwards results to an external API:

```python
import requests
from ai4rag.utils.event_handler import BaseEventHandler, LogLevel, PatternPayload, EvaluationRecord

class MyEventHandler(BaseEventHandler):

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        if level == LogLevel.ERROR:
            requests.post("https://my-service/alerts", json={"step": step, "message": message})

    def on_pattern_creation(
        self,
        payload: PatternPayload,
        evaluation_results: list[EvaluationRecord],
        conversations: list[dict] | None = None,
        **kwargs,
    ) -> None:
        requests.post("https://my-service/patterns", json={
            "name": payload["name"],
            "template_id": payload["template_id"],
            "metrics": payload["evaluation"]["metrics"],
            "settings": payload["settings"],
            "evaluation_results": evaluation_results,
            "conversations": conversations,
        })
```

Your handler can do anything: write to a database, push metrics to a dashboard, send Slack notifications, or stream results to a frontend — the engine does not care.
