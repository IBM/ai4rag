# Evaluation

Evaluation is the foundation of RAG optimization in ai4rag. Every parameter configuration is judged by how well it performs on your benchmark dataset, using standardized metrics that measure different aspects of RAG quality.

---

## Why Evaluation Matters

RAG systems can fail in subtle ways:

- Generate answers that sound plausible but contradict the retrieved documents (hallucination)
- Retrieve irrelevant documents that don't help answer the question
- Produce incorrect answers even when the right information is available

ai4rag uses **unitxt-based metrics** to detect these failures and guide optimization toward configurations that produce accurate, grounded, and relevant responses.

---

## Available Metrics

ai4rag evaluates three complementary aspects of RAG performance:

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
Ground truth: ["ChromaDB and Milvus via Llama Stack", "Milvus and ChromaDB"]
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

---

## How Evaluation Works

### The UnitxtEvaluator

ai4rag uses the `UnitxtEvaluator` class, which wraps the [unitxt](https://github.com/IBM/unitxt) library for RAG evaluation.

For each RAG configuration being tested:

1. **Generate answers** for all benchmark questions using the current configuration
2. **Collect evaluation data**:
   - Question
   - Generated answer
   - Retrieved contexts (chunks)
   - Context IDs (document IDs)
   - Ground truth answers
   - Ground truth document IDs
3. **Compute metrics** using unitxt's RAG evaluation algorithms
4. **Return scores** with confidence intervals

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

Evaluation results include both aggregate scores and per-question breakdowns.

### Aggregate Scores

For each metric, you get:

- **`mean`**: Average score across all questions
- **`ci_low`**: Lower bound of 95% confidence interval
- **`ci_high`**: Upper bound of 95% confidence interval

**Example**:

```python
{
    "scores": {
        "faithfulness": {
            "mean": 0.72,
            "ci_low": 0.61,
            "ci_high": 0.83
        },
        "answer_correctness": {
            "mean": 0.68,
            "ci_low": 0.55,
            "ci_high": 0.81
        },
        "context_correctness": {
            "mean": 0.80,
            "ci_low": 0.70,
            "ci_high": 0.90
        }
    },
    "question_scores": {
        # Per-question breakdown (see below)
    }
}
```

!!! tip "Confidence Intervals"
    Wide confidence intervals (e.g., 0.50-0.90) suggest high variance across questions. This might indicate that your benchmark data covers diverse scenarios, or that the configuration works well for some questions but poorly for others.

---

### Per-Question Scores

Detailed breakdown showing how each question performed:

```python
{
    "question_scores": {
        "faithfulness": {
            "q0": 0.71,
            "q1": 0.73,
            "q2": 0.68
        },
        "answer_correctness": {
            "q0": 0.65,
            "q1": 0.70,
            "q2": 0.69
        },
        "context_correctness": {
            "q0": 0.80,
            "q1": 0.85,
            "q2": 0.75
        }
    }
}
```

This granular data helps you identify:

- Which questions are consistently difficult across all configurations
- Which configurations excel at specific question types
- Outliers that might indicate benchmark data quality issues

---

## Choosing the Optimization Metric

ai4rag optimizes for a **single objective metric**. By default, this is **`FAITHFULNESS`**, but you can change it when creating your experiment.

### Default: Faithfulness

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment

experiment = AI4RAGExperiment(
    # ... other parameters
    # objective_metric defaults to MetricType.FAITHFULNESS
)
```

**Why faithfulness is the default**: Hallucination is the most critical failure mode. A system that invents information is worse than one that gives incomplete but accurate answers.

---

### Optimizing for Answer Correctness

If your priority is maximizing accuracy:

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.evaluator.base_evaluator import MetricType

experiment = AI4RAGExperiment(
    # ... other parameters
    objective_metric=MetricType.ANSWER_CORRECTNESS
)
```

**When to use this**: When you have high-quality ground truth answers and want to maximize end-to-end accuracy, even if it means occasionally including less relevant context.

---

### Optimizing for Context Correctness

If your priority is retrieval quality:

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.evaluator.base_evaluator import MetricType

experiment = AI4RAGExperiment(
    # ... other parameters
    objective_metric=MetricType.CONTEXT_CORRECTNESS
)
```

**When to use this**: When you're primarily optimizing retrieval (chunking, embedding, retrieval method) and your generation model is already well-tuned.

---

### Trade-offs

| Metric | Optimizes For | Risk |
|--------|--------------|------|
| **Faithfulness** | Grounded, trustworthy answers | May retrieve more context than necessary |
| **Answer Correctness** | Accurate final answers | May prioritize accuracy over explainability |
| **Context Correctness** | Retrieval precision | May not account for generation quality |

!!! tip "Multi-Objective Optimization"
    While ai4rag optimizes a single metric, all three are computed for every evaluation. Review all metrics when analyzing results to ensure your best configuration doesn't sacrifice one quality for another.

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
      "ChromaDB and Milvus via Llama Stack"
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
    "ChromaDB (in-memory) and Milvus via Llama Stack"
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

## Code Example

Here's a complete example showing how evaluation is used in the experiment loop:

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
from ai4rag.evaluator.base_evaluator import MetricType
from ai4rag.utils.event_handler import LocalEventHandler

from dev_utils.file_store import FileStore
from dev_utils.utils import read_benchmark_from_json

# Setup
load_dotenv()
client = LlamaStackClient(base_url=os.getenv("BASE_URL"), api_key=os.getenv("APIKEY"))

# Load data
documents = FileStore(Path("./knowledge_base")).load_as_documents()
benchmark_data = read_benchmark_from_json(Path("./benchmark_data.json"))

# Define search space
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
        Parameter(name="chunk_size", param_type="C", values=[512, 1024]),
        Parameter(name="number_of_chunks", param_type="C", values=[3, 5, 7]),
    ]
)

# Run optimization (optimizes for faithfulness by default)
experiment = AI4RAGExperiment(
    client=client,
    documents=documents,
    benchmark_data=benchmark_data,
    search_space=search_space,
    vector_store_type="ls_milvus",
    optimizer_settings=GAMOptSettings(max_evals=8, n_random_nodes=3),
    objective_metric=MetricType.FAITHFULNESS,  # Can change to ANSWER_CORRECTNESS or CONTEXT_CORRECTNESS
    event_handler=LocalEventHandler(output_path="./results"),
)

best_pattern = experiment.search()

# Results are automatically saved to ./results with all three metrics
print(f"Best pattern achieved faithfulness: {best_pattern.scores['scores']['faithfulness']['mean']:.2f}")
print(f"Answer correctness: {best_pattern.scores['scores']['answer_correctness']['mean']:.2f}")
print(f"Context correctness: {best_pattern.scores['scores']['context_correctness']['mean']:.2f}")
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

- **Three metrics**: Faithfulness (grounding), Answer Correctness (accuracy), Context Correctness (retrieval quality)
- **Powered by unitxt**: Industry-standard RAG evaluation library
- **Single objective**: Optimizes for one metric, but computes all three
- **Benchmark-driven**: Quality depends on your benchmark data
- **Confidence intervals**: Statistical rigor built-in
- **Per-question breakdown**: Detailed diagnostics for analysis

High-quality evaluation starts with high-quality benchmark data. Invest time in creating diverse, accurate, and representative questions for the best optimization results.
