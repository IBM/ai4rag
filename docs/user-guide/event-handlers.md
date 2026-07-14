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
from ai4rag import AI4RAGExperiment
from ai4rag.utils.event_handler import BaseEventHandler

experiment = AI4RAGExperiment(
    ...
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

Called once per evaluated RAG pattern. This is where you receive the full evaluation result and can persist, forward, or display it.

```python
def on_pattern_creation(
    self, payload: PatternPayload, evaluation_results: list[EvaluationRecord], **kwargs
) -> None:
    ...
```

### `payload` — pattern summary

```python
{
    "name": "Pattern1",
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
        "generation":  {"model_id": "...", ...},
        "vector_store_binding": {"provider_id": "local_chroma", "vector_store_id": "..."},
    },
}
```

### `evaluation_results` — per-question breakdown

```python
[
    {
        "question": "What is ...?",
        "answer": "According to the document ...",
        "correct_answers": ["The correct answer is ..."],
        "answer_contexts": [
            {"text": "Retrieved chunk text ...", "document_id": "doc1.pdf"},
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

---

## Built-in Handlers

Two ready-to-use implementations are provided:

**`LocalEventHandler`** — saves each pattern's `payload` and `evaluation_results` as JSON files under a given output directory. Ideal for local development.

```python
from ai4rag.utils.event_handler import LocalEventHandler

handler = LocalEventHandler(output_path="./results")
```

**`KFPEventHandler`** — accumulates all status changes and pattern results in memory (`.status_changes` and `.patterns` lists). Designed for Kubeflow Pipelines components where you need to access results after the experiment finishes.

```python
from ai4rag.utils.event_handler import KFPEventHandler

handler = KFPEventHandler()
experiment.run(...)
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
        self, payload: PatternPayload, evaluation_results: list[EvaluationRecord], **kwargs
    ) -> None:
        requests.post("https://my-service/patterns", json={
            "name": payload["name"],
            "metrics": payload["evaluation"]["metrics"],
            "settings": payload["settings"],
        })
```

Your handler can do anything: write to a database, push metrics to a dashboard, send Slack notifications, or stream results to a frontend — the engine does not care.
