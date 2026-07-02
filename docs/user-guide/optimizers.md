# Optimizers

Optimizers are the algorithms that explore your search space to find the best RAG configuration. Instead of testing every possible combination (which could be millions), optimizers use intelligent strategies to converge on high-performing configurations with far fewer evaluations.

---

## What Optimizers Do

Given a search space with parameters like:

- `chunk_size`: [200, 400, 800, 1000]
- `chunk_overlap`: [0, 50, 100, 200]
- `retrieval_method`: ["simple", "window"]
- `number_of_chunks`: [3, 5, 7, 10]

There are **4 × 4 × 2 × 4 = 128 possible combinations** (before validation rules).

Testing all 128 would be expensive and time-consuming. Optimizers find near-optimal configurations by testing only a fraction of the space (e.g., 10-20 evaluations).

---

## Available Optimizers

ai4rag provides two optimizer implementations:

| Optimizer | Strategy | Best For |
|-----------|----------|----------|
| **GAM Optimizer** | Model-based (Generalized Additive Models) | Most use cases; recommended default |
| **Random Optimizer** | Random sampling | Quick baseline, small search spaces |

---

## GAM Optimizer (Recommended)

The GAM Optimizer uses **Generalized Additive Models** to predict which configurations are likely to perform well, then evaluates those promising candidates.

### How It Works

The optimization process has two phases:

**Phase 1: Random Exploration**

- Randomly sample `n_random_nodes` configurations from the search space
- Evaluate each using the objective function (your chosen metric)
- Build an initial dataset of (configuration, score) pairs

**Phase 2: Model-Guided Search**

- Train a LinearGAM model on the evaluated configurations
- Use the model to predict scores for all remaining (unevaluated) configurations
- Select the top `evals_per_trial` configurations with highest predicted scores
- Evaluate those configurations
- Repeat until `max_evals` total evaluations are reached

This approach balances **exploration** (random phase) with **exploitation** (model-guided phase).

---

### Configuration

```python
from ai4rag.core.hpo.gam_opt import GAMOptSettings

optimizer_settings = GAMOptSettings(
    max_evals=20,          # Total number of configurations to evaluate
    n_random_nodes=4,      # Number of random evaluations before using the model
    evals_per_trial=1,     # Number of configurations to evaluate per iteration (default: 1)
    random_state=64        # Random seed for reproducibility (default: 64)
)
```

---

### Parameters Explained

#### `max_evals`

**What it controls**: Total number of RAG configurations to evaluate.

**Typical values**:

- **10-15**: Quick exploration (small search spaces)
- **20-30**: Balanced optimization (medium search spaces)
- **50+**: Thorough search (large, complex search spaces)

**Trade-offs**:

- Higher = better results, but slower and more expensive
- Lower = faster, but may miss optimal configurations

!!! tip "Setting max_evals"
    A good rule of thumb: set `max_evals` to 10-20% of your search space size (after validation rules). For a search space with 100 combinations, try `max_evals=15`.

---

#### `n_random_nodes`

**What it controls**: Number of random configurations to evaluate before training the GAM model.

**Default**: 4

**Why it matters**: Random exploration ensures the model is trained on diverse configurations, avoiding local optima.

**Typical values**:

- **3-4**: Small search spaces, quick convergence
- **5-8**: Medium search spaces, balanced exploration
- **10+**: Large search spaces, high diversity needed

**Trade-offs**:

- Higher = more exploration, better model training data, but fewer model-guided evaluations
- Lower = more model-guided iterations, but risk of poor model predictions if initial samples aren't diverse

!!! warning "Balance with max_evals"
    Ensure `n_random_nodes < max_evals`. If `n_random_nodes` equals or exceeds `max_evals`, the optimizer will only perform random search.

---

#### `evals_per_trial`

**What it controls**: How many configurations to evaluate in each model-guided iteration.

**Default**: 1

**Why it matters**: Evaluating multiple candidates per iteration speeds up optimization but reduces the number of times the model is retrained.

**Typical values**:

- **1**: Maximum model refinement (model is retrained after every evaluation)
- **2-3**: Faster convergence with slight trade-off in model accuracy

**Trade-offs**:

- Higher = fewer iterations, faster wall-clock time, but less adaptive model
- Lower = more iterations, slower, but model adapts more frequently

---

#### `random_state`

**What it controls**: Random seed for reproducibility.

**Default**: 64

**Why it matters**: Using the same random state across runs ensures identical random exploration phases, making results reproducible.

---

### Usage Example

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.search_space.src.parameter import Parameter

# Define search space
search_space = AI4RAGSearchSpace(
    params=[
        # ... model parameters
        Parameter(name="chunk_size", param_type="C", values=[200, 400, 800, 1000]),
        Parameter(name="chunk_overlap", param_type="C", values=[0, 50, 100, 200]),
        Parameter(name="number_of_chunks", param_type="C", values=[3, 5, 7, 10]),
    ]
)

# Conservative exploration: more random sampling
optimizer_settings = GAMOptSettings(
    max_evals=20,
    n_random_nodes=8  # 40% of evaluations are random
)

experiment = AI4RAGExperiment(
    # ... other parameters
    search_space=search_space,
    optimizer_settings=optimizer_settings,
)

best_pattern = experiment.search()
```

---

### Warm-Starting with Known Observations

If you've already run experiments and want to continue optimization with new settings or a refined search space:

```python
from ai4rag.core.hpo.gam_opt import GAMOptimizer, GAMOptSettings

# Previous results (from earlier experiments)
known_observations = [
    {
        "chunk_size": 512,
        "chunk_overlap": 64,
        "number_of_chunks": 5,
        "score": 0.72
    },
    {
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "number_of_chunks": 7,
        "score": 0.68
    },
    # ... more observations
]

optimizer = GAMOptimizer(
    objective_function=my_objective_function,
    search_space=search_space,
    settings=GAMOptSettings(max_evals=20, n_random_nodes=4),
    known_observations=known_observations  # Warm-start from previous runs
)

best_result = optimizer.search()
```

**Benefits of warm-starting**:

- Skip re-evaluating configurations you've already tested
- Build on previous optimization runs
- Useful when expanding search space or changing objective metric

!!! note "Known Observations Format"
    Each observation must include:
    - All search space parameter keys with their values
    - A `"score"` key with the evaluation result (float)

---

### When to Use GAM Optimizer

**Use GAM when**:

- You have medium to large search spaces (50+ combinations)
- You want efficient optimization (minimize evaluations)
- You're willing to invest in 20+ evaluations for quality results
- You need a balance between exploration and exploitation

**Avoid GAM when**:

- Search space is tiny (<10 combinations) - use Random Optimizer instead
- You want absolute certainty of finding the global optimum - exhaustive search is better

---

## Random Optimizer

The Random Optimizer is a simple baseline that randomly shuffles all possible combinations and evaluates them up to `max_evals`.

### How It Works

1. Generate all valid combinations from the search space (after applying validation rules)
2. Shuffle the list randomly
3. Evaluate the first `max_evals` configurations
4. Return the configuration with the highest score

No model training, no prediction - just random sampling.

---

### Configuration

```python
from ai4rag.core.hpo.random_opt import RandomOptSettings

optimizer_settings = RandomOptSettings(
    max_evals=10,       # Number of configurations to evaluate (required)
    random_state=64     # Random seed for reproducibility (default: 64)
)
```

---

### Usage Example

```python
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.random_opt import RandomOptSettings
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

search_space = AI4RAGSearchSpace(
    params=[
        # ... parameters
    ]
)

optimizer_settings = RandomOptSettings(max_evals=10)

experiment = AI4RAGExperiment(
    # ... other parameters
    search_space=search_space,
    optimizer_settings=optimizer_settings,
)

best_pattern = experiment.search()
```

---

### When to Use Random Optimizer

**Use Random when**:

- You have a very small search space (<20 combinations)
- You want a quick baseline to compare against GAM
- You're debugging and want deterministic behavior
- You're running a sensitivity analysis across the full search space

**Avoid Random when**:

- Search space is large (>50 combinations) - you'll waste evaluations
- You need efficient optimization - GAM will find better results faster

---

## Optimizer Comparison

| Feature | GAM Optimizer | Random Optimizer |
|---------|--------------|------------------|
| **Strategy** | Model-based prediction | Random sampling |
| **Best for** | Medium-large search spaces (50+) | Small search spaces (<20) |
| **Efficiency** | High (finds good configs quickly) | Low (may waste evaluations) |
| **Convergence** | Typically 10-30 evaluations | Depends on luck |
| **Overhead** | Model training (minimal) | None |
| **Reproducibility** | Controlled by `random_state` | Controlled by `random_state` |
| **Warm-start support** | Yes (`known_observations`) | No |
| **Exploration** | Balanced (random + guided) | Pure random |

---

## Configuration Examples

### Example 1: Conservative Exploration (More Random)

When you have a large, complex search space and want thorough exploration:

```python
GAMOptSettings(
    max_evals=30,
    n_random_nodes=10  # 33% random exploration
)
```

**Use case**: First-time optimization on a new dataset, unfamiliar search space.

---

### Example 2: Aggressive Exploitation (More Model-Guided)

When you have a well-understood search space and want fast convergence:

```python
GAMOptSettings(
    max_evals=20,
    n_random_nodes=3  # 15% random exploration
)
```

**Use case**: Refinement runs, similar datasets, known parameter relationships.

---

### Example 3: Quick Baseline (Random)

When you just want a rough estimate:

```python
RandomOptSettings(
    max_evals=10
)
```

**Use case**: Initial exploration, proof-of-concept, small search spaces.

---

### Example 4: Warm-Start from Previous Run

When you're expanding your search space or changing the objective metric:

```python
import json

# Load previous results
with open("./results/experiment_results.json") as f:
    previous_results = json.load(f)

known_observations = [
    {
        "chunk_size": result["settings"]["chunking"]["chunk_size"],
        "chunk_overlap": result["settings"]["chunking"]["chunk_overlap"],
        "number_of_chunks": result["settings"]["retrieval"]["number_of_chunks"],
        "score": result["final_score"]
    }
    for result in previous_results
]

optimizer = GAMOptimizer(
    objective_function=objective_fn,
    search_space=expanded_search_space,
    settings=GAMOptSettings(max_evals=15, n_random_nodes=4),
    known_observations=known_observations
)
```

**Use case**: Iterative optimization, budget-constrained experiments.

---

## Tips for Choosing Optimizer Settings

### 1. Start with Defaults

For most use cases, these defaults work well:

```python
GAMOptSettings(
    max_evals=20,
    n_random_nodes=4
)
```

---

### 2. Scale with Search Space Size

| Search Space Size | Recommended `max_evals` | Recommended `n_random_nodes` |
|-------------------|------------------------|------------------------------|
| 10-50 combinations | 10-15 | 3-4 |
| 50-200 combinations | 15-25 | 4-6 |
| 200+ combinations | 25-50 | 6-10 |

---

### 3. Balance Exploration and Exploitation

**Rule of thumb**: `n_random_nodes` should be 15-30% of `max_evals`.

```python
# Good balance
GAMOptSettings(max_evals=20, n_random_nodes=5)  # 25% random

# Too much exploration
GAMOptSettings(max_evals=20, n_random_nodes=15)  # 75% random - might as well use Random Optimizer

# Too little exploration
GAMOptSettings(max_evals=20, n_random_nodes=1)  # 5% random - risk of poor model training
```

---

### 4. Consider Evaluation Cost

If each evaluation is expensive (large documents, slow models):

- Lower `max_evals` to reduce total runtime
- Use GAM Optimizer to make every evaluation count
- Increase `evals_per_trial` to 2-3 to speed up wall-clock time

If evaluations are cheap:

- Higher `max_evals` for thorough search
- Use Random Optimizer for simple baselines

---

### 5. Use Reproducible Seeds

Always set `random_state` for reproducible experiments:

```python
GAMOptSettings(
    max_evals=20,
    n_random_nodes=4,
    random_state=42  # Any integer; ensures reproducibility
)
```

---

## Troubleshooting

### Optimization Converges Too Quickly

**Symptom**: Best configuration found in first few evaluations, no improvement afterward.

**Cause**: Search space is too small or too constrained.

**Actions**:

- Expand your search space with more parameter values
- Check validation rules - are they too restrictive?
- Verify `max_evals` is appropriate for your search space size

---

### All Evaluations Fail

**Symptom**: `OptimizationError: Number of evaluations has reached limit. All iterations have failed.`

**Cause**: Objective function is raising exceptions for every configuration.

**Actions**:

- Check logs for error messages during evaluation
- Verify models are accessible (Llama Stack connection, API keys)
- Test a single configuration manually to isolate the issue
- Ensure benchmark data and documents are properly formatted

---

### Random Phase Takes Forever

**Symptom**: Optimizer gets stuck during random exploration.

**Cause**: `n_random_nodes` is set too high relative to evaluation speed.

**Actions**:

- Reduce `n_random_nodes` to speed up convergence
- Optimize evaluation pipeline (smaller models, fewer chunks, smaller benchmark)

---

### Model-Guided Phase Doesn't Improve Results

**Symptom**: Scores during GAM phase are no better than random phase.

**Cause**: GAM model can't find meaningful patterns in the search space.

**Actions**:

- Increase `n_random_nodes` to give the model more training data
- Check if your search space has too few parameters or values
- Verify that your objective metric is sensitive to parameter changes

---

## Related Topics

- [Search Space](search-space.md): Define parameters for optimization
- [Evaluation](evaluation.md): Understanding objective metrics
- [Event Handlers](event-handlers.md): Monitor optimization progress
- [Quick Start](../getting-started/quick-start.md): Complete optimization example

---

## Summary

Optimizers in ai4rag:

- **GAM Optimizer**: Recommended for most use cases, balances exploration and exploitation
- **Random Optimizer**: Simple baseline, best for small search spaces
- **Key settings**: `max_evals` (total evaluations), `n_random_nodes` (random exploration)
- **Warm-starting**: Reuse previous results with `known_observations`
- **Trade-offs**: More evaluations = better results but higher cost

Start with the default GAM settings (`max_evals=20`, `n_random_nodes=4`) and adjust based on your search space size, evaluation cost, and optimization goals.
