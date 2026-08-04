# Hybrid Search

Hybrid search combines dense vector search with sparse keyword-based search to improve retrieval quality in RAG systems.
This feature is available in `ai4rag` when using Milvus or PGVector as your vector store.

## What is Hybrid Search?

Traditional RAG systems typically rely on either:

- **Dense Vector Search**: Uses semantic embeddings to find documents similar in meaning to the query
- **Sparse/Keyword Search**: Uses lexical matching (e.g., BM25) to find documents containing specific keywords

Hybrid search **combines both approaches**, leveraging the strengths of each:

- Vector search excels at understanding semantic relationships and context
- Keyword search excels at finding exact term matches and rare/specific terminology

The results from both methods are merged using a ranking strategy to produce a final ranked list of documents.

---

## When to Use Hybrid Search

Consider enabling hybrid search when:

- **Keyword-specific queries**: Your users search for specific terms, product names, or technical jargon that might be missed by pure semantic search
- **Domain-specific terminology**: Your knowledge base contains specialized vocabulary where exact matches matter
- **Improved recall**: You want to ensure important documents aren't missed by either approach alone
- **Balanced relevance**: You need both semantic understanding and lexical precision

**Example scenarios**:

- Legal document retrieval (specific statute numbers, case names)
- Technical documentation (API names, error codes)
- Product catalogs (SKUs, model numbers)
- Medical information (drug names, procedure codes)

---

## Prerequisites

!!! warning "Vector Store Requirement"
    Hybrid search is **supported with Milvus** (`MilvusConfig`) **and PGVector** (`PGVectorConfig`). It is **NOT available with Chroma**, which is vector-only.

Ensure your experiment is configured with:

```python
from ai4rag.rag.vector_store import MilvusConfig

experiment = AI4RAGExperiment(
    vector_store_config=MilvusConfig.from_env(),  # Required for hybrid search; PGVectorConfig also works
    # ... other parameters
)
```

---

## Configuration

To enable hybrid search in your optimization experiments, add the following parameters to your search space:

### Required Parameters

#### `search_mode`

Controls the search method used for retrieval.

**Parameter Details**:

- **Name**: `AI4RAGParamNames.SEARCH_MODE`
- **Type**: Categorical (`"C"`)
- **Values**:
    - `"vector"` (default): Traditional dense vector search only
    - `"hybrid"`: Combines both vector and keyword search

**Example**:

```python
from ai4rag.utils.constants import AI4RAGParamNames
from ai4rag.search_space.src.parameter import Parameter

Parameter(
    name=AI4RAGParamNames.SEARCH_MODE,
    param_type="C",
    values=["vector", "hybrid"]
)
```

#### `ranker_strategy`

Defines how to merge results from dense and sparse search in hybrid mode.

**Parameter Details**:

- **Name**: `AI4RAGParamNames.RANKER_STRATEGY`
- **Type**: Categorical (`"C"`)
- **Values**:
    - `""` (empty string): Sentinel value when `search_mode` is not `"hybrid"`
    - `"rrf"`: Reciprocal Rank Fusion
    - `"weighted"`: Weighted combination of dense and sparse scores
    - `"normalized"`: Score normalization before combination

**Example**:

```python
Parameter(
    name=AI4RAGParamNames.RANKER_STRATEGY,
    param_type="C",
    values=["", "rrf", "weighted"]
)
```

#### `ranker_k`

Parameter for the ranking function (strategy-specific).

**Parameter Details**:

- **Name**: `AI4RAGParamNames.RANKER_K`
- **Type**: Categorical (`"C"`) or Integer (`"I"`)
- **Values**:
    - `0`: Sentinel value when not applicable
    - For `"rrf"`: Smoothing constant (typical range: 30-100)
    - For other strategies: Consult specific strategy documentation

**Example (Categorical)**:

```python
Parameter(
    name=AI4RAGParamNames.RANKER_K,
    param_type="C",
    values=[0, 30, 60, 100]
)
```

!!! note "When to use 0"
    Set `ranker_k=0` when `search_mode` is `"vector"` or when the ranker strategy doesn't use this parameter.

#### `ranker_alpha`

Weight parameter for the `"weighted"` ranking strategy.

**Parameter Details**:

- **Name**: `AI4RAGParamNames.RANKER_ALPHA`
- **Type**: Categorical (`"C"`) or Real/Float (`"R"`)
- **Values**:
    - `1`: Sentinel value when not applicable (means vector-only / 100% dense)
    - For `"weighted"`: Weight between 0 and 1 (e.g., `0.7` = 70% dense, 30% sparse; `0` = 100% sparse)

**Example (Categorical)**:

```python
Parameter(
    name=AI4RAGParamNames.RANKER_ALPHA,
    param_type="C",
    values=[1, 0.3, 0.5, 0.7]
)
```

!!! note "When to use 1"
    Set `ranker_alpha=1` when `ranker_strategy` is not `"weighted"`. The value `1` acts as a sentinel meaning "100% dense / vector-only".

---

## Ranking Strategies Explained

### Reciprocal Rank Fusion (RRF)

**When to use**: General-purpose hybrid search; works well without tuning.

**How it works**: Merges ranked lists using the RRF formula:

$$
\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}
$$

Where:

- `d` is a document
- `k` is the smoothing constant (`ranker_k`)
- `rank_r(d)` is the rank of document `d` in ranker `r`

**Parameters**:

- `ranker_k`: Smoothing constant (typical values: 30-100)
    - Lower values give more weight to top-ranked documents
    - Higher values distribute weight more evenly

**Example**:

```python
Parameter(name=AI4RAGParamNames.SEARCH_MODE, param_type="C", values=["hybrid"]),
Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, param_type="C", values=["rrf"]),
Parameter(name=AI4RAGParamNames.RANKER_K, param_type="C", values=[30, 60, 100]),
Parameter(name=AI4RAGParamNames.RANKER_ALPHA, param_type="C", values=[1]),  # Not used for RRF
```

---

### Weighted Strategy

**When to use**: You want explicit control over the balance between semantic and keyword search.

**How it works**: Combines normalized scores from both retrievers using a weighted average:

$$
\text{score}(d) = \alpha \cdot \text{score}_{\text{dense}}(d) + (1 - \alpha) \cdot \text{score}_{\text{sparse}}(d)
$$

Where:

- `α` (alpha) = `ranker_alpha`
- Higher alpha (e.g., 0.7) favors dense/semantic search
- Lower alpha (e.g., 0.3) favors sparse/keyword search

**Parameters**:

- `ranker_alpha`: Weight for dense vs sparse (range: 0.0 to 1.0)
    - `0.5` = equal weight to both
    - `0.7` = 70% dense, 30% sparse
    - `0.3` = 30% dense, 70% sparse

**Example**:

```python
Parameter(name=AI4RAGParamNames.SEARCH_MODE, param_type="C", values=["hybrid"]),
Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, param_type="C", values=["weighted"]),
Parameter(name=AI4RAGParamNames.RANKER_K, param_type="C", values=[0]),  # Not used
Parameter(name=AI4RAGParamNames.RANKER_ALPHA, param_type="R", v_min=0.0, v_max=1.0),
```

---

### Normalized Strategy

**When to use**: You need score normalization before combining results.

**How it works**: Normalizes scores from both retrievers to a common scale before merging.

**Parameters**: Strategy-specific; consult documentation.

**Example**:

```python
Parameter(name=AI4RAGParamNames.SEARCH_MODE, param_type="C", values=["hybrid"]),
Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, param_type="C", values=["normalized"]),
Parameter(name=AI4RAGParamNames.RANKER_K, param_type="C", values=[0]),  # Depends on implementation
Parameter(name=AI4RAGParamNames.RANKER_ALPHA, param_type="C", values=[1]),  # Not used for normalized
```

---

## Validation Rules

`ai4rag` automatically validates hybrid search configurations with built-in rules:

### Rule 1: Non-Hybrid Modes Must Use Sentinel Values

When `search_mode` is `"vector"`, all ranker parameters must be sentinels:

- `ranker_strategy` must be `""`
- `ranker_k` must be `0`
- `ranker_alpha` must be `1` (meaning 100% dense / vector-only)

!!! example "Valid Configuration"
    ```python
    {
        "search_mode": "vector",
        "ranker_strategy": "",
        "ranker_k": 0,
        "ranker_alpha": 1
    }
    ```

!!! failure "Invalid Configuration"
    ```python
    {
        "search_mode": "vector",
        "ranker_strategy": "rrf",  # ERROR: ranker_strategy must be "" for vector mode
        "ranker_k": 60,
        "ranker_alpha": 1
    }
    ```

---

### Rule 2: Hybrid Mode Requires Ranker Strategy

When `search_mode` is `"hybrid"`, `ranker_strategy` must be non-empty.

!!! example "Valid Configuration"
    ```python
    {
        "search_mode": "hybrid",
        "ranker_strategy": "rrf",
        "ranker_k": 60,
        "ranker_alpha": 1
    }
    ```

!!! failure "Invalid Configuration"
    ```python
    {
        "search_mode": "hybrid",
        "ranker_strategy": "",  # ERROR: must specify a ranker strategy
        "ranker_k": 0,
        "ranker_alpha": 1
    }
    ```

---

### Rule 3: ranker_alpha Only for Weighted Strategy

`ranker_alpha` must be:

- `1` (sentinel) when `ranker_strategy` is NOT `"weighted"`
- Not `1` when `ranker_strategy` is `"weighted"` (valid range: 0 to <1, where 0 = 100% sparse)

!!! example "Valid Configuration"
    ```python
    {
        "search_mode": "hybrid",
        "ranker_strategy": "weighted",
        "ranker_k": 0,
        "ranker_alpha": 0.7  # Valid for weighted
    }
    ```

!!! failure "Invalid Configuration"
    ```python
    {
        "search_mode": "hybrid",
        "ranker_strategy": "rrf",
        "ranker_k": 60,
        "ranker_alpha": 0.5  # ERROR: ranker_alpha must be 1 for non-weighted strategies
    }
    ```

---

## Code Examples

### Example 1: Basic Hybrid Search with RRF

Explore hybrid search using Reciprocal Rank Fusion:

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel

search_space = AI4RAGSearchSpace(
    params=[
        # Models (required)
        Parameter(
            name=AI4RAGParamNames.FOUNDATION_MODEL,
            param_type="C",
            values=[OGXFoundationModel(model_id="ollama/llama3.2:3b", client=client)],
        ),
        Parameter(
            name=AI4RAGParamNames.EMBEDDING_MODEL,
            param_type="C",
            values=[
                OGXEmbeddingModel(
                    model_id="ollama/nomic-embed-text:latest",
                    client=client,
                    params={"embedding_dimension": 768, "context_length": 8192},
                )
            ],
        ),
        # Chunking
        Parameter(name=AI4RAGParamNames.CHUNK_SIZE, param_type="C", values=[512, 1024]),
        Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, param_type="C", values=[128, 256]),
        # Retrieval
        Parameter(name=AI4RAGParamNames.RETRIEVAL_METHOD, param_type="C", values=["simple"]),
        Parameter(name=AI4RAGParamNames.NUMBER_OF_CHUNKS, param_type="C", values=[5, 10]),
        # Hybrid search with RRF
        Parameter(name=AI4RAGParamNames.SEARCH_MODE, param_type="C", values=["hybrid"]),
        Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, param_type="C", values=["rrf"]),
        Parameter(name=AI4RAGParamNames.RANKER_K, param_type="C", values=[30, 60, 100]),
        Parameter(name=AI4RAGParamNames.RANKER_ALPHA, param_type="C", values=[1]),
    ]
)
```

---

### Example 2: Hybrid Search with Weighted Ranker

Optimize the balance between semantic and keyword search:

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

search_space = AI4RAGSearchSpace(
    params=[
        # ... (models, chunking, retrieval as above)

        # Hybrid search with weighted ranker
        Parameter(name=AI4RAGParamNames.SEARCH_MODE, param_type="C", values=["hybrid"]),
        Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, param_type="C", values=["weighted"]),
        Parameter(name=AI4RAGParamNames.RANKER_K, param_type="C", values=[0]),  # Not used
        # Explore different weightings (0 = 100% sparse, 0.5 = balanced, 0.9 = mostly dense)
        Parameter(name=AI4RAGParamNames.RANKER_ALPHA, param_type="C", values=[0, 0.3, 0.5, 0.7, 0.9]),
    ]
)
```

!!! tip "Interpretation"
    - `0.3`: 30% semantic, 70% keyword (favors exact matches)
    - `0.5`: Equal weight to both approaches
    - `0.7`: 70% semantic, 30% keyword (favors meaning)
    - `0.9`: 90% semantic, 10% keyword (mostly semantic)

---

### Example 3: Mixed Search Space (Vector vs Hybrid)

Let the optimizer decide whether hybrid search improves results:

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

search_space = AI4RAGSearchSpace(
    params=[
        # ... (models, chunking, retrieval as above)

        # Explore both vector-only and hybrid search
        Parameter(
            name=AI4RAGParamNames.SEARCH_MODE,
            param_type="C",
            values=["vector", "hybrid"]
        ),
        # When hybrid, try both RRF and weighted
        Parameter(
            name=AI4RAGParamNames.RANKER_STRATEGY,
            param_type="C",
            values=["", "rrf", "weighted"]  # "" for vector mode
        ),
        # RRF smoothing parameter (0 when not applicable)
        Parameter(
            name=AI4RAGParamNames.RANKER_K,
            param_type="C",
            values=[0, 30, 60]
        ),
        # Weighted alpha (1 = sentinel for non-weighted / vector-only)
        Parameter(
            name=AI4RAGParamNames.RANKER_ALPHA,
            param_type="C",
            values=[1, 0.5, 0.7]
        ),
    ]
)
```

!!! note "Validation"
    `ai4rag`'s built-in validation rules automatically filter invalid combinations:

    - When `search_mode="vector"`, only combinations with sentinel ranker params are kept
    - When `search_mode="hybrid"` and `ranker_strategy="rrf"`, only `ranker_alpha=1` (sentinel) is valid
    - When `search_mode="hybrid"` and `ranker_strategy="weighted"`, `ranker_alpha` must not be `1`

---

## Best Practices

### 1. Start with RRF

Reciprocal Rank Fusion is a good default choice:

- Works well without extensive tuning
- Robust across different domains
- Only one parameter to optimize (`ranker_k`)

```python
Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, param_type="C", values=["rrf"]),
Parameter(name=AI4RAGParamNames.RANKER_K, param_type="C", values=[30, 60, 100]),
```

### 2. Use Weighted for Fine Control

If you have domain knowledge about the importance of semantic vs keyword matching:

```python
Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, param_type="C", values=["weighted"]),
# Favor semantic for conceptual queries
Parameter(name=AI4RAGParamNames.RANKER_ALPHA, param_type="C", values=[0.6, 0.7, 0.8]),
```

### 3. Let the Optimizer Explore

Include both vector-only and hybrid in your search space to see if hybrid search actually improves your specific use case:

```python
Parameter(name=AI4RAGParamNames.SEARCH_MODE, param_type="C", values=["vector", "hybrid"]),
```

### 4. Combine with Other Parameters

Hybrid search interacts with other retrieval parameters:

```python
# Hybrid search might work better with more retrieved chunks
Parameter(name=AI4RAGParamNames.NUMBER_OF_CHUNKS, param_type="C", values=[5, 7, 10]),
Parameter(name=AI4RAGParamNames.SEARCH_MODE, param_type="C", values=["vector", "hybrid"]),
```

### 5. Monitor Results

Check experiment outputs to see performance differences:

```python
# After experiment completes
results_df = pd.read_csv(f"{output_path}/experiment_results.csv")

# Compare vector vs hybrid performance
vector_results = results_df[results_df["search_mode"] == "vector"]
hybrid_results = results_df[results_df["search_mode"] == "hybrid"]

print("Vector avg score:", vector_results["objective_value"].mean())
print("Hybrid avg score:", hybrid_results["objective_value"].mean())
```

---

## Troubleshooting

### Error: "Search mode ... is not supported with chroma vector store"

**Cause**: Your `vector_store_config` is a `ChromaConfig` (Chroma is vector-only).

**Solution**: Switch to Milvus or PGVector:

```python
from ai4rag.rag.vector_store import MilvusConfig

experiment = AI4RAGExperiment(
    vector_store_config=MilvusConfig.from_env(),  # or PGVectorConfig.from_env()
    # ...
)
```

---

### Error: "Invalid parameter combination"

**Cause**: Validation rules are rejecting your configuration.

**Solution**: Check that:

- When `search_mode="vector"`: all ranker params are sentinels (`""`, `0`, `1` for alpha)
- When `search_mode="hybrid"`: `ranker_strategy` is non-empty
- When `ranker_strategy="weighted"`: `ranker_alpha` is not `1`
- When `ranker_strategy` is NOT `"weighted"`: `ranker_alpha = 1`

---

### No Performance Improvement with Hybrid Search

**Possible causes**:

1. **Your use case favors pure semantic search**: Some domains (e.g., conversational queries) benefit more from semantic understanding
2. **Benchmark questions lack keyword-specific elements**: Hybrid search shines when exact term matches matter
3. **Incorrect ranker configuration**: Try different strategies (RRF vs weighted) and parameters

**Actions**:

- Review your benchmark questions - do they include specific terms/names?
- Examine retrieved documents manually to see if keyword search adds value
- Experiment with different `ranker_k` (for RRF) or `ranker_alpha` (for weighted) values

---

## Related Topics

- [Search Space Configuration](search-space.md): Learn about defining search spaces
- [Retrieval Strategies](../api-reference/rag/retrieval.md): Understand retrieval methods
- [Vector Stores](../api-reference/rag/vector-stores.md): Vector store configuration and options
- [Evaluation Metrics](evaluation.md): How hybrid search impacts evaluation scores

---

## Summary

Hybrid search in `ai4rag` combines the best of semantic and keyword-based retrieval:

- **Use `search_mode="hybrid"`** to enable hybrid search (requires Milvus or PGVector, i.e., `vector_store_config=MilvusConfig(...)` or `PGVectorConfig(...)`)
- **Choose a ranker strategy**: `"rrf"` (general-purpose), `"weighted"` (fine control), or `"normalized"`
- **Configure strategy parameters**: `ranker_k` for RRF, `ranker_alpha` for weighted
- **Let the optimizer explore**: Include both vector and hybrid modes to find the best approach
- **Validation is automatic**: `ai4rag` enforces rules to prevent invalid configurations

Hybrid search is particularly valuable when your knowledge base contains specialized terminology or when users search for specific terms that pure semantic search might miss.
