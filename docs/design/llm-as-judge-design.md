# Design: LLM-as-a-Judge Optimization Metric

## 1. Motivation

The current ai4rag evaluation stack uses **unitxt** with algorithmic (non-LLM) metrics:
`answer_correctness`, `faithfulness`, and `context_correctness`. These are fast and deterministic
but limited in capturing nuanced quality dimensions like helpfulness, coherence, or domain-specific
correctness that only an LLM judge can assess.

**Goal:** Allow users to use LLM-as-a-Judge as optimization metrics in ai4rag's
HPO loop, by swapping the evaluator. The same metric names (`answer_correctness`, `faithfulness`,
etc.) are used — the evaluator determines the evaluation method.

---

## 2. Architecture

```
BaseEvaluator (ABC)
  └── evaluate_metrics(evaluation_data, metrics) -> dict

UnitxtEvaluator(BaseEvaluator)         # algorithmic metrics
MlflowLLMJudgeEvaluator(BaseEvaluator) # LLM-as-a-Judge metrics

MetricType (ConstantMeta)
  └── ANSWER_CORRECTNESS, FAITHFULNESS, CONTEXT_CORRECTNESS, ANSWER_RELEVANCE
```

The evaluator is a constructor parameter of `AI4RAGExperiment`. Users swap between unitxt and
LLM-as-a-Judge by passing a different evaluator — no changes to the optimizer or experiment
orchestrator are needed. Both evaluators use the same metric names and return the same dict format.

All scores are normalized to **[0.0, 1.0]**.

---

## 3. New Metric Type

One new metric added to `MetricType`:

```python
class MetricType(metaclass=ConstantMeta):
    ANSWER_CORRECTNESS = "answer_correctness"
    FAITHFULNESS = "faithfulness"
    CONTEXT_CORRECTNESS = "context_correctness"
    ANSWER_RELEVANCE = "answer_relevance"          # NEW
```

No `LLM_` prefix — the metric name describes *what* is measured, not *how*.

---

## 4. MlflowLLMJudgeEvaluator

New file: `ai4rag/evaluator/mlflow_llm_judge_evaluator.py`

### 4.1 Configuration

```python
@dataclass
class LLMJudgeConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    custom_metrics: list[CustomMetricDefinition] = field(default_factory=list)
```

The evaluator uses **MLflow's evaluation framework** (`mlflow.genai.evaluate()`) with custom
`@scorer` functions. The scorers call the judge LLM via the **OpenAI Python client** with a
configurable `base_url`, enabling any OpenAI-compatible endpoint (vLLM, TGI, etc.). This gives
full MLflow tracking/logging integration while working around MLflow's internal gateway routing
which does not respect custom base URLs for `openai:/` URIs.

### 4.2 Built-in Metric Prompts

The evaluator ships with grading prompts for all four `MetricType` values:
- `answer_correctness` — factual correctness vs ground truth
- `faithfulness` — grounding in retrieved context
- `context_correctness` — relevance of retrieved documents
- `answer_relevance` — relevance and helpfulness to the question

Each uses a 1-5 scale that is normalized to [0.0, 1.0] via `(score - 1) / 4`.

### 4.3 Evaluation Flow

```
1. Convert list[EvaluationData] → MLflow eval data format
   (inputs: {question, context}, outputs: answer, expectations: {expected_response})
2. Build MLflow @scorer functions for each requested metric
   - Each scorer calls the judge LLM via OpenAI client
   - Parses JSON {"score": 1-5, "rationale": "..."} response
   - Normalizes to [0.0, 1.0] and returns mlflow.entities.Feedback
3. Call mlflow.genai.evaluate(data=eval_data, scorers=scorers)
4. Extract per-row scores from eval_results table
5. Compute mean + bootstrap confidence intervals (seed=42, n=1000)
6. Return {"scores": {metric: {mean, ci_low, ci_high}},
           "question_scores": {metric: {q_id: score}}}
```

### 4.4 Custom LLM Judge Metrics

```python
config = LLMJudgeConfig(
    base_url="https://my-llm-endpoint.example.com/v1",
    api_key="my-token",
    model="llama-31-8b-instruct",
    custom_metrics=[
        CustomMetricDefinition(
            name="medical_accuracy",
            guidelines="Evaluate whether the answer contains medically accurate information.",
        )
    ]
)
```

Custom metric names are accepted as `optimization_metric` when the evaluator supports them.

---

## 5. Integration Points

| File | Change |
|------|--------|
| `evaluator/base_evaluator.py` | Add `ANSWER_RELEVANCE` to `MetricType` |
| `evaluator/mlflow_llm_judge_evaluator.py` | **New file** |
| `evaluator/__init__.py` | Conditional export of new classes |
| `core/experiment/experiment.py` | Accept custom evaluator metric names in validation |
| `pyproject.toml` | Add `mlflow` + `openai` as optional dependencies (`ai4rag[llm-judge]`) |

---

## 6. Dependency Management

```toml
[project.optional-dependencies]
llm-judge = ["mlflow>=3.0.0", "openai>=1.0.0"]
```

The evaluator raises a clear `ImportError` if the packages are not installed.

---

## 7. Usage Example

```python
from ai4rag.evaluator.mlflow_llm_judge_evaluator import MlflowLLMJudgeEvaluator, LLMJudgeConfig
from ai4rag.evaluator.base_evaluator import MetricType

config = LLMJudgeConfig(
    base_url="https://llama-31-8b-instruct.apps.example.com/v1",
    api_key="my-token",
    model="llama-31-8b-instruct",
)

experiment = AI4RAGExperiment(
    documents=documents,
    benchmark_data=benchmark_df,
    search_space=search_space,
    vector_store_type="chroma",
    optimizer_settings=optimizer_settings,
    event_handler=event_handler,
    evaluator=MlflowLLMJudgeEvaluator(config),
    optimization_metric=MetricType.FAITHFULNESS,
    metrics=(MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS),
)

experiment.search()
```

---

## References

- [MLflow LLM Evaluation docs](https://mlflow.org/docs/latest/genai/eval-monitor/llm-evaluation/)
- [MLflow Custom Scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/)
- [OpenAI Python Client](https://github.com/openai/openai-python)
- [vLLM OpenAI Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
