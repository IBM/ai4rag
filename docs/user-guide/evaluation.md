# Evaluation

Evaluation is the foundation of RAG optimization in ai4rag. Every parameter configuration is judged by how well it performs on your benchmark dataset, using standardized metrics that measure different aspects of RAG quality.

---

## Why Evaluation Matters

RAG systems can fail in subtle ways:

- Generate answers that sound plausible but contradict the retrieved documents (hallucination)
- Retrieve irrelevant documents that don't help answer the question
- Produce incorrect answers even when the right information is available

ai4rag uses **multiple evaluator types** to detect these failures and guide optimization toward configurations that produce accurate, grounded, and relevant responses.

---

## Available Metrics

ai4rag evaluates four complementary aspects of RAG performance using two evaluator types — **Unitxt** (reference-based) and **LLM-as-a-Judge**:

### Faithfulness

**What it measures**: Whether the generated answer is grounded in the retrieved context.

**Why it matters**: This metric detects hallucination. A high faithfulness score means the model is not inventing information beyond what was retrieved from your knowledge base.

**Score range**: 0.0 to 1.0 (higher is better)

**Example failure (low faithfulness)**:

```
Question: "What is the capital of France?"
Retrieved context: "France is a country in Western Europe."
Answer: "The capital of France is Paris."
Faithfulness: Low (Paris is not mentioned in the retrieved context)
```

**Example success (high faithfulness)**:

```
Question: "Where is France located?"
Retrieved context: "France is a country in Western Europe."
Answer: "France is located in Western Europe."
Faithfulness: High (answer is fully grounded in the context)
```

---

### Answer Correctness

**What it measures**: How correct the generated answer is compared to the ground truth answers in your benchmark data.

**Why it matters**: This is the ultimate test of whether your RAG system produces accurate responses. Even if the answer is grounded in context, it might still be incomplete or wrong.

**Score range**: 0.0 to 1.0 (higher is better)

**Example**:

```
Question: "What vector databases does ai4rag support?"
Ground truth: ["ChromaDB and Milvus", "Milvus and ChromaDB"]
Answer: "ai4rag supports ChromaDB and Milvus."
Answer Correctness: High (matches ground truth)
```

---

### Context Correctness

**What it measures**: How relevant the retrieved documents are to answering the question.

**Why it matters**: Good retrieval is essential for good answers. This metric evaluates whether your chunking, embedding, and retrieval strategy is finding the right information.

**Score range**: 0.0 to 1.0 (higher is better)

**How it works**: Compares the document IDs of retrieved chunks against the `correct_answer_document_ids` in your benchmark data.

**Example**:

```
Question: "How do I configure hybrid search?"
Correct document IDs: ["hybrid_search_guide.md", "vector_stores.md"]
Retrieved document IDs: ["hybrid_search_guide.md", "installation.md"]
Context Correctness: Medium (1 of 2 correct documents retrieved)
```

### Answer Relevance (LLM Judge)

**What it measures**: Whether the generated response directly and helpfully addresses the user's question, as judged by an LLM.

**Why it matters**: This metric provides an independent quality signal that does not require ground-truth answers. It detects off-topic, unhelpful, or incoherent responses that reference-based metrics might miss.

**Score range**: 0.0 to 1.0 (higher is better)

**How it works**: An LLM judge scores the response on a 1–5 rubric using structured JSON output. The raw score is normalized to [0.0, 1.0]. Confidence intervals are computed via bootstrapping.

---

### RAGAS Metrics

The optional `RagasEvaluator` adds four LLM-based metrics from the [RAGAS](https://github.com/explodinggradients/ragas) library as an independent cross-check on the reference-based and judge metrics. All are scored in [0.0, 1.0] (higher is better) and run through the same foundation and embedding models the rest of the pipeline uses.

- **`faithfulness`** (evaluator `ragas`) — how well the answer is grounded in the retrieved context without hallucination.
- **`answer_relevancy`** — how relevant and on-topic the answer is to the question. Requires an embedding model.
- **`context_precision`** — whether the retrieved contexts relevant to the ground truth are ranked highly.
- **`context_recall`** — how much of the ground-truth answer is covered by the retrieved contexts.

!!! note "Distinct from the Unitxt `faithfulness`"
    RAGAS `faithfulness` and Unitxt `faithfulness` share a name but are produced by different evaluators and computed differently. Select the RAGAS variant with `Metrics.RAGAS_FAITHFULNESS` rather than the bare string.

---

### Overall Score

**What it measures**: The mean of all other evaluated metrics.

**Why it matters**: A single aggregate score for optimization that balances all quality dimensions. This is the **default optimization metric** in ai4rag.

**Score range**: 0.0 to 1.0 (higher is better)

---

## How Evaluation Works

### Multi-Evaluator Architecture

ai4rag supports multiple evaluator types working together. Each evaluator handles the metrics matching its type:

- **`UnitxtEvaluator`** — wraps the [unitxt](https://github.com/IBM/unitxt) library for reference-based RAG metrics (`faithfulness`, `answer_correctness`, `context_correctness`)
- **`LLMaJEvaluator`** — uses an LLM as a judge for `answer_relevance`
- **`RagasEvaluator`** — wraps the [RAGAS](https://github.com/explodinggradients/ragas) library for LLM-based RAG metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`)
- **Custom metrics** — computed from the results of other evaluators (e.g. `overall_score` is the mean of all other metrics)

!!! note "Metric names are not unique across evaluators"
    A metric name can be produced by more than one evaluator — for example both the Unitxt and RAGAS evaluators expose a `faithfulness` metric. Each result carries an `evaluator` field (`"unitxt"`, `"judge"`, `"ragas"`, or `"custom"`) that disambiguates them. Because a bare name is therefore ambiguous, `metrics` and `optimization_metric` accept only `RAGMetric` instances from the `Metrics` registry (e.g. `Metrics.FAITHFULNESS` for the Unitxt variant, `Metrics.RAGAS_FAITHFULNESS` for the RAGAS one); passing a string raises an error.

For each RAG configuration being tested:

1. **Generate answers** for all benchmark questions using the current configuration
2. **Collect evaluation data** (question, answer, contexts, ground truths)
3. **Route metrics to evaluators** — each evaluator receives only its own metrics
4. **Merge results** into a single `EvaluationMetricsResult`
5. **Compute custom metrics** (e.g. `overall_score`) on top of the merged results
6. **Return scores** with confidence intervals

### EvaluationData Structure

Each question's data is packaged as an `EvaluationData` instance:

```python
from ai4rag.evaluator.base_evaluator import EvaluationData

evaluation_data = EvaluationData(
    question="What is ai4rag?",
    answer="ai4rag is a RAG optimization engine...",
    contexts=[
        "ai4rag optimizes RAG templates using hyperparameter optimization.",
        "The engine is provider-agnostic and works with any LLM."
    ],
    context_ids=["readme.md", "overview.md"],
    ground_truths=[
        "ai4rag is a RAG optimization engine",
        "ai4rag optimizes RAG configurations"
    ],
    ground_truths_context_ids=["readme.md", "architecture.md"],
    question_id="q0"
)
```

---

## Result Structure

Evaluation results are returned as an `EvaluationMetricsResult` TypedDict with two sections: aggregate metrics and per-question breakdowns.

### Aggregate Metrics

For each metric, you get:

- **`name`**: Metric identifier
- **`evaluator`**: Which evaluator produced it (`"unitxt"`, `"judge"`, `"ragas"`, or `"custom"`)
- **`scores.mean`**: Average score across all questions
- **`scores.ci_low`**: Lower bound of 95% confidence interval
- **`scores.ci_high`**: Upper bound of 95% confidence interval

**Example**:

```python
{
    "metrics": [
        {
            "name": "faithfulness",
            "evaluator": "unitxt",
            "description": "Measures whether the generated answer is grounded in the retrieved context.",
            "scores": {"mean": 0.72, "ci_low": 0.61, "ci_high": 0.83},
        },
        {
            "name": "answer_correctness",
            "evaluator": "unitxt",
            "description": "Measures how accurately the generated answer matches the ground-truth.",
            "scores": {"mean": 0.68, "ci_low": 0.55, "ci_high": 0.81},
        },
        {
            "name": "answer_relevance",
            "evaluator": "judge",
            "description": "LLM judge score for how directly the response addresses the question.",
            "scores": {"mean": 0.85, "ci_low": 0.78, "ci_high": 0.92},
            "model_id": "ollama/llama3.2:3b",
        },
        {
            "name": "overall_score",
            "evaluator": "custom",
            "description": "Aggregate score computed as the mean of all other evaluated metrics.",
            "scores": {"mean": 0.75, "ci_low": 0.65, "ci_high": 0.85},
        },
    ],
    "question_scores": [
        # Per-question breakdown (see below)
    ],
}
```

!!! tip "Confidence Intervals"
    Wide confidence intervals (e.g., 0.50-0.90) suggest high variance across questions. This might indicate that your benchmark data covers diverse scenarios, or that the configuration works well for some questions but poorly for others.

---

### Per-Question Scores

Detailed breakdown showing how each question performed:

```python
{
    "question_scores": [
        {
            "question_id": "q0",
            "metrics": [
                {"name": "faithfulness", "evaluator": "unitxt", "value": 0.71},
                {"name": "answer_correctness", "evaluator": "unitxt", "value": 0.65},
                {"name": "answer_relevance", "evaluator": "judge", "value": 0.90},
                {"name": "overall_score", "evaluator": "custom", "value": 0.75},
            ],
        },
        {
            "question_id": "q1",
            "metrics": [
                {"name": "faithfulness", "evaluator": "unitxt", "value": 0.73},
                {"name": "answer_correctness", "evaluator": "unitxt", "value": 0.70},
                {"name": "answer_relevance", "evaluator": "judge", "value": 0.80},
                {"name": "overall_score", "evaluator": "custom", "value": 0.74},
            ],
        },
    ]
}
```

This granular data helps you identify:

- Which questions are consistently difficult across all configurations
- Which configurations excel at specific question types
- Outliers that might indicate benchmark data quality issues

---

## Choosing the Optimization Metric

ai4rag optimizes for a **single objective metric**. By default, this is **`overall_score`** (the mean of all other metrics), but you can change it when creating your experiment.

The `optimization_metric` parameter accepts a `RAGMetric` instance from the `Metrics` registry:

### Default: Overall Score

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment

experiment = AI4RAGExperiment(
    # ... other parameters
    # optimization_metric defaults to Metrics.OVERALL_SCORE
)
```

**Why overall_score is the default**: It balances all quality dimensions — grounding, accuracy, retrieval precision, and response relevance — into a single aggregate score, preventing optimization from over-fitting to one aspect at the expense of others.

---

### Optimizing for a Specific Metric

You can target any metric from the `Metrics` registry:

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.evaluator.metric import Metrics

experiment = AI4RAGExperiment(
    # ... other parameters
    optimization_metric=Metrics.FAITHFULNESS,
)
```

---

### Trade-offs

| Metric | Optimizes For | Risk |
|--------|--------------|------|
| **Overall Score** | Balanced quality across all dimensions | May not maximize any single aspect |
| **Faithfulness** | Grounded, trustworthy answers | May retrieve more context than necessary |
| **Answer Correctness** | Accurate final answers | May prioritize accuracy over explainability |
| **Context Correctness** | Retrieval precision | May not account for generation quality |
| **Answer Relevance** | Direct, helpful responses (LLM judge) | Requires a judge model; adds inference cost |

!!! tip "Multi-Objective Optimization"
    While ai4rag optimizes a single metric, all configured metrics are computed for every evaluation. Review all metrics when analyzing results to ensure your best configuration doesn't sacrifice one quality for another.

---

## Benchmark Data Quality

The quality of your evaluation depends entirely on the quality of your benchmark data.

### Benchmark Data Format

Your `benchmark_data.json` must follow this schema:

```json
[
  {
    "question": "What is ai4rag?",
    "correct_answers": [
      "ai4rag is a RAG optimization engine",
      "ai4rag optimizes RAG templates using hyperparameter optimization"
    ],
    "correct_answer_document_ids": ["readme.md", "overview.md"]
  },
  {
    "question": "Which vector databases are supported?",
    "correct_answers": [
      "ChromaDB and Milvus"
    ],
    "correct_answer_document_ids": ["vector_stores.md", "quick_start.md"]
  }
]
```

---

### Best Practices for Benchmark Data

**1. Diverse Question Types**

Include different question patterns:

```json
[
  {
    "question": "What is X?",  // Factual
    "correct_answers": ["X is a RAG optimization engine"]
  },
  {
    "question": "How do I configure Y?",  // Procedural
    "correct_answers": ["To configure Y, set the parameter..."]
  },
  {
    "question": "When should I use Z?",  // Conceptual
    "correct_answers": ["Use Z when you need..."]
  }
]
```

**2. Multiple Correct Answers**

Provide alternative phrasings for the same correct answer:

```json
{
  "question": "What vector databases does ai4rag support?",
  "correct_answers": [
    "ChromaDB and Milvus",
    "Milvus and ChromaDB",
    "ChromaDB (in-memory) and Milvus"
  ]
}
```

This makes evaluation more robust to phrasing variations.

---

**3. Accurate Document IDs**

Ensure `correct_answer_document_ids` match the `document_id` metadata in your knowledge base:

```python
# When loading documents
from langchain_core.documents import Document

documents = [
    Document(
        page_content="...",
        metadata={"document_id": "readme.md"}  # Must match benchmark data
    )
]
```

---

**4. Representative Coverage**

Your benchmark should cover:

- Common questions users will ask
- Edge cases (ambiguous questions, multi-step reasoning)
- Questions that require different amounts of context
- Questions answerable from single vs. multiple documents

---

**5. Ground Truth Verification**

Manually verify that:

- All correct answers are actually correct
- All document IDs actually contain the information needed
- Questions are unambiguous and answerable from your knowledge base

!!! warning "Garbage In, Garbage Out"
    If your benchmark data contains errors (wrong answers, incorrect document IDs), optimization will converge to configurations that produce those wrong answers. Always validate your benchmark data before running experiments.

---

## Configuring Evaluators

By default, `AI4RAGExperiment` uses only the `UnitxtEvaluator`. To enable LLM-as-a-Judge evaluation alongside Unitxt, pass both evaluators:

```python
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from dev_utils.utils import build_maas_model

judge_model = build_maas_model(client, model_id="qwen3-8b-fp8-dynamic", model_type="llm")

experiment = AI4RAGExperiment(
    # ... other parameters
    evaluators=[UnitxtEvaluator(), LLMaJEvaluator(model=judge_model)],
)
```

When both evaluators are configured, the experiment automatically evaluates `answer_relevance` via the LLM judge alongside the Unitxt reference-based metrics. The default metrics list adjusts to include `answer_relevance` when a judge evaluator is present.

You can also explicitly control which metrics to evaluate:

```python
from ai4rag.evaluator.metric import Metrics

experiment = AI4RAGExperiment(
    # ... other parameters
    evaluators=[UnitxtEvaluator(), LLMaJEvaluator(model=judge_model)],
    metrics=[Metrics.FAITHFULNESS, Metrics.JUDGE_ANSWER_RELEVANCE, Metrics.OVERALL_SCORE],
)
```

### Adding the RAGAS Evaluator

To also compute the RAGAS metrics, add a `RagasEvaluator` configured with a foundation model (used as the evaluating LLM) and an embedding model:

```python
from ai4rag.evaluator.ragas_evaluator import RagasEvaluator
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel

# `maas_client` is the shared OpenAI-compatible client (see create_maas_client).
ragas_model = OpenAIFoundationModel(model_id="qwen3-8b-fp8-dynamic", client=maas_client)
ragas_embeddings = OpenAIEmbeddingModel(model_id="bge-m3", client=maas_client)

experiment = AI4RAGExperiment(
    # ... other parameters
    evaluators=[
        UnitxtEvaluator(),
        LLMaJEvaluator(model=judge_model),
        RagasEvaluator(model=ragas_model, embedding_model=ragas_embeddings),
    ],
)
```

When a `RagasEvaluator` is present, the default metrics list is extended with the four RAGAS metrics. RAGAS is a regular dependency of ai4rag, so no optional extra is required.

In the high-level `run_rag_optimization` pipeline, the LLM-as-a-judge evaluators are selected via the `llm_judge_mode` parameter (`"base"`, `"ragas"`, `"all"`, or `"none"`; default `"base"`). The reference-based `UnitxtEvaluator` always runs; `llm_judge_mode` controls whether the in-house LLM judge, RAGAS, both, or neither are added on top. Any mode other than `"none"` requires at least one foundation model and one embedding model, and RAGAS runs on the first configured foundation and embedding models.

---

## Code Example

Here's a complete example showing how evaluation is used in the experiment loop:

```python
from pathlib import Path
from dotenv import load_dotenv

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.rag.vector_store import MilvusConfig
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.evaluator.metric import Metrics
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from ai4rag.utils.event_handler import LocalEventHandler

from dev_utils.file_store import FileStore
from dev_utils.utils import build_maas_model, create_dev_maas_client, read_benchmark_from_json

# Setup
load_dotenv()
client = create_dev_maas_client()  # reads MAAS_BASE_URL / MAAS_API_KEY

# Load data
documents = FileStore(Path("./knowledge_base")).load_as_documents()
benchmark_data = read_benchmark_from_json(Path("./benchmark_data.json"))

# Define search space
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
        Parameter(name="chunk_size", param_type="C", values=[512, 1024]),
        Parameter(name="number_of_chunks", param_type="C", values=[3, 5, 7]),
    ]
)

# Configure evaluators — Unitxt for reference-based metrics, LLMaJ for judge-based
judge_model = build_maas_model(client, model_id="qwen3-8b-fp8-dynamic", model_type="llm")

# Run optimization (optimizes for overall_score by default)
experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_config=MilvusConfig.from_env(),
    optimizer_settings=GAMOptSettings(max_evals=8, n_random_nodes=3),
    evaluators=[UnitxtEvaluator(), LLMaJEvaluator(model=judge_model)],
    optimization_metric=Metrics.OVERALL_SCORE,
    event_handler=LocalEventHandler(output_path="./results"),
)

experiment.search()

# Access results
best = experiment.results.get_best_evaluations(k=1)[0]
for m in best.scores["metrics"]:
    print(f"{m['name']}: {m['scores']['mean']:.2f}")
```

---

## Troubleshooting

### All Scores Are Low

**Possible causes**:

1. **Poor benchmark quality**: Questions don't match knowledge base
2. **Model mismatch**: Foundation model isn't suitable for the task
3. **Insufficient context**: `number_of_chunks` is too low
4. **Bad retrieval**: Chunking or embedding strategy isn't working

**Actions**:

- Manually test a few benchmark questions against your knowledge base
- Verify that document IDs in benchmark data match your actual documents
- Try increasing `number_of_chunks` in your search space
- Inspect retrieved contexts in the evaluation results JSON files

---

### Faithfulness Is High but Answer Correctness Is Low

**Cause**: The model is generating grounded answers, but they're not matching the ground truth.

**Actions**:

- Review your ground truth answers - are they too specific?
- Provide multiple acceptable phrasings in `correct_answers`
- Check if the retrieved context actually contains the information needed
- Consider optimizing for `ANSWER_CORRECTNESS` instead

---

### Context Correctness Is High but Other Metrics Are Low

**Cause**: Retrieval is finding the right documents, but generation is failing.

**Actions**:

- Try a different foundation model
- Increase `max_new_tokens` if answers are being cut off
- Check prompt templates (system message, user message)
- Verify that `include_chunk_metadata` isn't confusing the model

---

### Evaluation Fails with UnitxtEvaluator Error

**Cause**: Missing required fields in evaluation data.

**Actions**:

- Ensure all benchmark questions have non-empty `correct_answers`
- Verify `correct_answer_document_ids` are provided
- Check that generated answers aren't empty (model timeout issue)

---

## Related Topics

- [Optimizers](optimizers.md): How optimization uses evaluation scores
- [Search Space](search-space.md): Parameters that affect evaluation results
- [Event Handlers](event-handlers.md): Accessing detailed evaluation results
- [Quick Start](../getting-started/quick-start.md): Creating benchmark data

---

## Summary

Evaluation in ai4rag:

- **Core metrics**: Faithfulness (grounding), Answer Correctness (accuracy), Context Correctness (retrieval quality), Answer Relevance (LLM judge), plus optional RAGAS metrics (faithfulness, answer relevancy, context precision/recall)
- **Multi-evaluator architecture**: Unitxt for reference-based metrics, LLM-as-a-Judge for response quality, RAGAS for an independent LLM-based cross-check
- **Overall score**: Cross-metric mean used as the default optimization target
- **Single objective**: Optimizes for one metric, but computes all configured metrics
- **Benchmark-driven**: Quality depends on your benchmark data
- **Confidence intervals**: Statistical rigor built-in
- **Per-question breakdown**: Detailed diagnostics for analysis

High-quality evaluation starts with high-quality benchmark data. Invest time in creating diverse, accurate, and representative questions for the best optimization results.
