# Search Space

The search space defines all possible RAG configurations that the optimizer can explore.
It specifies which parameters to optimize, what values they can take, and the rules that govern valid combinations.

---

## What Is a Search Space?

A search space is a collection of **parameters** that define a RAG configuration:

```python
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.search_space.src.parameter import Parameter

search_space = AI4RAGSearchSpace(
    params=[
        Parameter(name="chunk_size", param_type="C", values=[512, 1024, 2048]),
        Parameter(name="chunk_overlap", param_type="C", values=[64, 128, 256]),
        Parameter(name="retrieval_method", param_type="C", values=["simple", "window"]),
    ]
)
```

This search space has **3 × 3 × 2 = 18 possible combinations** (before validation rules).

The optimizer explores this space to find the combination that maximizes your chosen evaluation metric.

---

## Parameter Types

`ai4rag` supports four parameter types:

| Type | Code | Description | Example |
|------|------|-------------|---------|
| **Categorical** | `"C"` | Discrete set of values (strings, objects) | `["recursive", "markdown"]` |
| **Integer** | `"I"` | Integer range with min/max | `v_min=100, v_max=1000` |
| **Real** | `"R"` | Continuous range (float) | `v_min=0.0, v_max=1.0` |
| **Boolean** | `"B"` | True or False | `values=[True, False]` |

---

### Categorical Parameters (`"C"`)

Define a discrete set of possible values.

**Common use cases**:

- Model selection
- Method choices (chunking method, retrieval method, ranker strategy)
- Discrete numeric values (chunk sizes, number of chunks)

**Example 1: String values**

```python
Parameter(
    name="chunking_method",
    param_type="C",
    values=["recursive", "markdown", "markdown_header"]
)
```

**Example 2: Numeric values**

```python
Parameter(
    name="chunk_size",
    param_type="C",
    values=[200, 400, 800, 1000, 2048]
)
```

**Example 3: Model objects**

```python
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel

Parameter(
    name="foundation_model",
    param_type="C",
    values=[
        OGXFoundationModel(model_id="ollama/llama3.2:3b", client=client),
        OGXFoundationModel(model_id="ollama/llama3.1:8b", client=client),
    ]
)
```

!!! tip "Categorical for Discrete Numerics"
    Even for numeric parameters like `chunk_size`, use Categorical (`"C"`) when you want to test specific values rather than a continuous range. This gives you more control over which values are tested.

---

### Integer Parameters (`"I"`)

Define an integer range with minimum and maximum bounds.

**Syntax**:

```python
Parameter(
    name="parameter_name",
    param_type="I",
    v_min=100,    # Minimum value (inclusive)
    v_max=1000    # Maximum value (inclusive)
)
```

**Example**:

```python
Parameter(
    name="chunk_size",
    param_type="I",
    v_min=200,
    v_max=2048
)
# Generates: [200, 201, 202, ..., 2048]
```

!!! warning "Large Integer Ranges"
    Be cautious with large ranges. `v_min=100, v_max=5000` creates 4,901 possible values, which exponentially increases search space size when combined with other parameters. Consider using Categorical with specific values instead.

---

### Real Parameters (`"R"`)

Define a continuous floating-point range.

**Syntax**:

```python
Parameter(
    name="parameter_name",
    param_type="R",
    v_min=0.0,
    v_max=1.0
)
```

**Example**:

```python
Parameter(
    name="ranker_alpha",
    param_type="R",
    v_min=0.0,
    v_max=1.0
)
```

!!! note "Real Type Not Fully Supported"
    Currently, Real parameters cannot be enumerated (no `.all_values()` method). For practical optimization, use Categorical with discrete float values instead:
    ```python
    Parameter(name="ranker_alpha", param_type="C", values=[0.0, 0.3, 0.5, 0.7, 1.0])
    ```

---

### Boolean Parameters (`"B"`)

True/False parameters.

**Syntax**:

```python
Parameter(
    name="parameter_name",
    param_type="B",
    values=[True, False]
)
```

**Example**:

```python
Parameter(
    name="include_chunk_metadata",
    param_type="B",
    values=[True, False]
)
```

---

## Required Parameters

Two parameters are **always required** in an `AI4RAGSearchSpace`:

### 1. `foundation_model`

The LLM used for text generation.

**Example (OGX)**:

```python
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel

Parameter(
    name="foundation_model",
    param_type="C",
    values=[
        OGXFoundationModel(model_id="ollama/llama3.2:3b", client=client)
    ]
)
```

---

### 2. `embedding_model`

The model used for generating document and query embeddings.

**Example (OGX)**:

```python
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel

Parameter(
    name="embedding_model",
    param_type="C",
    values=[
        OGXEmbeddingModel(
            model_id="ollama/nomic-embed-text:latest",
            client=client,
            params={"embedding_dimension": 768, "context_length": 8192}
        )
    ]
)
```

!!! note "Embedding Model params"
    The `params` dict should include:
    - `embedding_dimension`: Vector size (e.g., 768, 1536)
    - `context_length`: Maximum tokens the model can process (used for validation)

---

## Default Parameters

If you don't specify certain parameters, `AI4RAGSearchSpace` uses sensible defaults. These defaults differ slightly between ChromaDB and OGX vector stores.

### Default Values

| Parameter | Default (OGX) | Default (ChromaDB) | Type |
|-----------|----------------------|-------------------|------|
| `chunking_method` | `("recursive", "hybrid")` | `("recursive", "hybrid")` | Categorical |
| `chunk_size` | `(512, 1024, 2048)` | `(512, 1024, 2048)` | Categorical |
| `chunk_overlap` | `(0, 128, 256)` | `(0, 128, 256)` | Categorical |
| `retrieval_method` | `("simple",)` | `("simple", "window")` | Categorical |
| `window_size` | `(0,)` | `(0, 1, 3, 5)` | Categorical |
| `number_of_chunks` | `(3, 5, 10)` | `(3, 5, 10)` | Categorical |
| `search_mode` | `("vector", "hybrid")` | `("vector",)` | Categorical |
| `ranker_strategy` | `("", "rrf", "weighted")` | N/A | Categorical |
| `ranker_k` | `(0, 60)` | N/A | Categorical |
| `ranker_alpha` | `(1, 0.5)` | N/A | Categorical |

!!! note "Why Different Defaults?"
    - **ChromaDB** doesn't support hybrid search, so `search_mode` is fixed to `"vector"` and ranker parameters are excluded
    - **ChromaDB** defaults include window retrieval options since it's an in-memory store (faster experimentation)
    - **OGX** defaults focus on simple retrieval but include hybrid search exploration

---

### Overriding Defaults

User-provided parameters always override defaults:

```python
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.search_space.src.parameter import Parameter

search_space = AI4RAGSearchSpace(
    params=[
        # Required parameters
        Parameter(name="foundation_model", param_type="C", values=[model]),
        Parameter(name="embedding_model", param_type="C", values=[embedding]),

        # Override default chunk_size
        Parameter(name="chunk_size", param_type="C", values=[512, 1024, 1536]),

        # chunk_overlap, retrieval_method, etc. will use defaults
    ]
)
```

---

## Validation Rules

`AI4RAGSearchSpace` enforces **built-in validation rules** to filter out invalid parameter combinations.

### Rule 1: Chunk Size Greater Than Chunk Overlap

**Rule**: `chunk_size > 2 * chunk_overlap`

**Why**: Text chunkers need enough non-overlapping content to create meaningful chunks.

**Example**:

```python
# Valid
{"chunk_size": 1024, "chunk_overlap": 256}  # 1024 > 2*256 = 512 ✓

# Invalid (filtered out)
{"chunk_size": 512, "chunk_overlap": 300}   # 512 > 2*300 = 600 ✗
```

!!! note "Exception: chunk_size = 0"
    When `chunk_size` is 0 (used for `markdown_header` structural-only splitting), this rule is skipped.

---

### Rule 2: Retrieval Method and Window Size Consistency

**Rule**:

- When `retrieval_method == "simple"`, `window_size` must be `0`
- When `retrieval_method == "window"`, `window_size` must be `> 0`

**Why**: Window retrieval requires a non-zero window; simple retrieval doesn't use windows.

**Example**:

```python
# Valid
{"retrieval_method": "simple", "window_size": 0}    ✓
{"retrieval_method": "window", "window_size": 3}    ✓

# Invalid (filtered out)
{"retrieval_method": "simple", "window_size": 2}    ✗
{"retrieval_method": "window", "window_size": 0}    ✗
```

---

### Rule 3: Chunk Overlap Consistent with Chunking Method

**Rule**:

- When `chunking_method == "hybrid"` (DoclingChunker), `chunk_overlap` must be `0` — the chunker does not support overlap
- When `chunking_method == "recursive"` (LangChainChunker), `chunk_overlap` must be `> 0` — splitting without overlap loses context between chunks

**Example**:

```python
# Valid
{"chunking_method": "hybrid", "chunk_overlap": 0}      ✓
{"chunking_method": "recursive", "chunk_overlap": 128}  ✓

# Invalid (filtered out)
{"chunking_method": "hybrid", "chunk_overlap": 128}     ✗
{"chunking_method": "recursive", "chunk_overlap": 0}    ✗
```

---

### Rule 4: Chunk Size Within Embedding Context Length

**Rule**: `chunk_size` (in tokens) must fit within the embedding model's `context_length`, with a 10% safety margin to account for tokenizer divergence.

**How it works**: Both chunkers express `chunk_size` in tokens, so the comparison is direct:

```python
chunk_size <= context_length * 0.9

# Example
chunk_size = 1024
context_length = 8192
# Check: 1024 <= 8192 * 0.9 = 7372.8 ✓
```

**Why**: If chunks exceed the embedding model's context window, embedding generation will fail or be truncated. The 10% margin accounts for the approximation in the chunker's character-based token estimator (4 chars = 1 token).

**Example**:

```python
# Embedding model with context_length = 512
embedding = OGXEmbeddingModel(
    model_id="small-embedder",
    params={"context_length": 512, "embedding_dimension": 384}
)

# Valid
{"chunk_size": 256, "embedding_model": embedding}   # 256 <= 460.8 ✓

# Invalid (filtered out)
{"chunk_size": 512, "embedding_model": embedding}    # 512 <= 460.8 ✗
```

---

### Rule 5: Hybrid Search Ranker Consistency

**Rule**: Ranker parameters must only be set when `search_mode == "hybrid"`.

**Sub-rules**:

- When `search_mode == "vector"`:
  - `ranker_strategy` must be `""` (empty string)
  - `ranker_k` must be `0`
  - `ranker_alpha` must be `1` (sentinel for vector-only)

- When `search_mode == "hybrid"`:
  - `ranker_strategy` must be a non-empty string (`"rrf"`, `"weighted"`, or `"normalized"`)

**Example**:

```python
# Valid: vector mode
{"search_mode": "vector", "ranker_strategy": "", "ranker_k": 0, "ranker_alpha": 1}    ✓

# Valid: hybrid mode
{"search_mode": "hybrid", "ranker_strategy": "rrf", "ranker_k": 60, "ranker_alpha": 1}    ✓

# Invalid (filtered out)
{"search_mode": "vector", "ranker_strategy": "rrf", "ranker_k": 60, "ranker_alpha": 1}   ✗
```

---

### Rule 6: ranker_k Only for RRF

**Rule**:

- When `ranker_strategy == "rrf"`, `ranker_k` must be `> 0`
- When `ranker_strategy != "rrf"`, `ranker_k` must be `0` (sentinel)

**Example**:

```python
# Valid
{"ranker_strategy": "rrf", "ranker_k": 60}         ✓
{"ranker_strategy": "weighted", "ranker_k": 0}     ✓

# Invalid (filtered out)
{"ranker_strategy": "rrf", "ranker_k": 0}          ✗
{"ranker_strategy": "weighted", "ranker_k": 60}    ✗
```

---

### Rule 7: ranker_alpha Only for Weighted

**Rule**:

- When `ranker_strategy == "weighted"`, `ranker_alpha` must be `!= 1` (valid range: 0 to <1)
- When `ranker_strategy != "weighted"`, `ranker_alpha` must be `1` (sentinel for 100% dense / vector-only)

**Example**:

```python
# Valid
{"ranker_strategy": "weighted", "ranker_alpha": 0.7}    ✓
{"ranker_strategy": "rrf", "ranker_alpha": 1}           ✓

# Invalid (filtered out)
{"ranker_strategy": "weighted", "ranker_alpha": 1}      ✗
{"ranker_strategy": "rrf", "ranker_alpha": 0.5}         ✗
```

---

## Sentinel Values

Sentinel values are special placeholder values used to indicate "not applicable" for optional parameters:

| Parameter | Sentinel Value | Meaning |
|-----------|---------------|----------|
| `ranker_strategy` | `""` (empty string) | No ranker (vector-only search) |
| `ranker_k` | `0` | Parameter not used |
| `ranker_alpha` | `1` | 100% dense / vector-only (not applicable) |

When `search_mode == "vector"`, all ranker parameters must be sentinels to indicate they're unused.

---

## Custom Validation Rules

You can add custom rules beyond the built-in ones:

```python
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.search_space.src.parameter import Parameter

def my_custom_rule(combination: dict) -> bool:
    """Only allow large chunks with window retrieval."""
    if combination["retrieval_method"] == "window":
        return combination["chunk_size"] >= 1024
    return True

search_space = AI4RAGSearchSpace(
    params=[
        # ... parameters
    ],
    rules=[my_custom_rule]  # Add custom rules here
)
```

**Custom rule requirements**:

- Function signature: `def rule_name(combination: dict) -> bool`
- Return `True` if the combination is valid, `False` to filter it out
- `combination` is a dict with all parameter names as keys

**Example use cases**:

- Domain-specific constraints (e.g., "small models need smaller chunks")
- Cost constraints (e.g., "don't use expensive models with large retrieval")
- Performance requirements (e.g., "window retrieval only with chunk_size > 512")

---

## Search Space Size

The `max_combinations` property tells you how many valid configurations exist:

```python
search_space = AI4RAGSearchSpace(
    params=[
        Parameter(name="chunk_size", param_type="C", values=[512, 1024, 2048]),
        Parameter(name="chunk_overlap", param_type="C", values=[64, 128, 256]),
        Parameter(name="retrieval_method", param_type="C", values=["simple", "window"]),
        # ... other params
    ]
)

print(f"Search space size: {search_space.max_combinations}")
# Output: Search space size: 42 (after validation rules filter invalid combos)
```

!!! tip "Estimating Optimizer Settings"
    Use `max_combinations` to guide your optimizer settings:
    - For spaces with <20 combinations: `max_evals=10`
    - For spaces with 20-100 combinations: `max_evals=15-25`
    - For spaces with 100+ combinations: `max_evals=30-50`

---

## Code Examples

### Example 1: Minimal Search Space

Optimize chunking and retrieval with fixed models:

```python
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel

search_space = AI4RAGSearchSpace(
    params=[
        # Required: models (not optimized, just fixed)
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[OGXFoundationModel(model_id="ollama/llama3.2:3b", client=client)]
        ),
        Parameter(
            name="embedding_model",
            param_type="C",
            values=[
                OGXEmbeddingModel(
                    model_id="ollama/nomic-embed-text:latest",
                    client=client,
                    params={"embedding_dimension": 768, "context_length": 8192}
                )
            ]
        ),

        # Optimize chunking
        Parameter(name="chunk_size", param_type="C", values=[512, 1024, 2048]),
        Parameter(name="chunk_overlap", param_type="C", values=[64, 128, 256]),

        # Optimize retrieval
        Parameter(name="number_of_chunks", param_type="C", values=[3, 5, 7, 10]),
    ]
)

# Other parameters (retrieval_method, window_size, etc.) will use defaults
```

---

### Example 2: Comprehensive Search Space

Explore chunking methods, retrieval strategies, and hybrid search:

```python
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.search_space.src.parameter import Parameter

search_space = AI4RAGSearchSpace(
    params=[
        # Models
        Parameter(name="foundation_model", param_type="C", values=[model]),
        Parameter(name="embedding_model", param_type="C", values=[embedding]),

        # Chunking
        Parameter(name="chunking_method", param_type="C", values=["recursive", "hybrid"]),
        Parameter(name="chunk_size", param_type="C", values=[512, 1024, 1536]),
        Parameter(name="chunk_overlap", param_type="C", values=[64, 128, 200]),

        # Retrieval
        Parameter(name="retrieval_method", param_type="C", values=["simple", "window"]),
        Parameter(name="window_size", param_type="C", values=[0, 1, 3, 5]),
        Parameter(name="number_of_chunks", param_type="C", values=[5, 7, 10]),

        # Hybrid search
        Parameter(name="search_mode", param_type="C", values=["vector", "hybrid"]),
        Parameter(name="ranker_strategy", param_type="C", values=["", "rrf", "weighted"]),
        Parameter(name="ranker_k", param_type="C", values=[0, 30, 60, 100]),
        Parameter(name="ranker_alpha", param_type="C", values=[1, 0.3, 0.5, 0.7]),
    ],
    vector_store_type="ogx",  # Required for hybrid search
    ogx_vector_io_provider_id="milvus",
)
```

---

### Example 3: Model Comparison

Compare different foundation models with fixed parameters:

```python
search_space = AI4RAGSearchSpace(
    params=[
        # Optimize model selection
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[
                OGXFoundationModel(model_id="ollama/llama3.2:3b", client=client),
                OGXFoundationModel(model_id="ollama/llama3.1:8b", client=client),
                OGXFoundationModel(model_id="ollama/mistral:7b", client=client),
            ]
        ),

        # Fixed embedding and parameters
        Parameter(name="embedding_model", param_type="C", values=[embedding]),
        Parameter(name="chunk_size", param_type="C", values=[1024]),
        Parameter(name="chunk_overlap", param_type="C", values=[128]),
        Parameter(name="number_of_chunks", param_type="C", values=[5]),
    ]
)
```

---

### Example 4: Custom Rule for Budget Constraints

Prevent expensive model + large retrieval combinations:

```python
def budget_constraint(combination: dict) -> bool:
    """Don't use large models with high retrieval counts."""
    model_id = str(combination["foundation_model"])
    num_chunks = combination["number_of_chunks"]

    # Large models (8b+) should use fewer chunks
    if "8b" in model_id or "13b" in model_id:
        return num_chunks <= 5
    return True

search_space = AI4RAGSearchSpace(
    params=[
        Parameter(
            name="foundation_model",
            param_type="C",
            values=[
                OGXFoundationModel(model_id="ollama/llama3.2:3b", client=client),
                OGXFoundationModel(model_id="ollama/llama3.1:8b", client=client),
            ]
        ),
        Parameter(name="embedding_model", param_type="C", values=[embedding]),
        Parameter(name="number_of_chunks", param_type="C", values=[3, 5, 7, 10]),
    ],
    rules=[budget_constraint]
)
```

---

## Best Practices

### 1. Start Small, Expand Gradually

Begin with a narrow search space to quickly find a baseline:

```python
# Iteration 1: Minimal space
search_space = AI4RAGSearchSpace(
    params=[
        # ... models
        Parameter(name="chunk_size", param_type="C", values=[1024]),  # Fixed
        Parameter(name="number_of_chunks", param_type="C", values=[5, 10]),  # Optimize
    ]
)
```

Once you understand which parameters matter, expand:

```python
# Iteration 2: Expanded space
search_space = AI4RAGSearchSpace(
    params=[
        # ... models
        Parameter(name="chunk_size", param_type="C", values=[512, 1024, 2048]),  # Now optimize
        Parameter(name="chunk_overlap", param_type="C", values=[64, 128]),  # Add overlap
        Parameter(name="number_of_chunks", param_type="C", values=[5, 7, 10]),  # Expand
    ]
)
```

---

### 2. Use Categorical for Most Parameters

Even for numeric values, prefer Categorical over Integer ranges:

```python
# Preferred
Parameter(name="chunk_size", param_type="C", values=[200, 400, 800, 1600])

# Avoid (creates 1401 values!)
Parameter(name="chunk_size", param_type="I", v_min=200, v_max=1600)
```

---

### 3. Respect Validation Rules

Check your search space size to ensure rules aren't over-filtering:

```python
search_space = AI4RAGSearchSpace(params=[...])

print(f"Valid combinations: {search_space.max_combinations}")

# If this is 0 or suspiciously low, check your parameter values
```

---

### 4. Avoid Overly Large Spaces

Search space grows multiplicatively. With 5 parameters, each with 4 values:

```
4 × 4 × 4 × 4 × 4 = 1,024 combinations
```

Keep individual parameter value counts reasonable (3-5 values per parameter).

---

### 5. Document Your Search Space

Add comments explaining your choices:

```python
search_space = AI4RAGSearchSpace(
    params=[
        # Using 3b model for speed; 8b is too slow for our budget
        Parameter(name="foundation_model", param_type="C", values=[model_3b]),

        # Chunk sizes optimized for technical documentation (shorter paragraphs)
        Parameter(name="chunk_size", param_type="C", values=[256, 512, 1024]),

        # Higher retrieval counts since documents are highly fragmented
        Parameter(name="number_of_chunks", param_type="C", values=[7, 10, 15]),
    ]
)
```

---

## Troubleshooting

### Error: "Missing required parameters"

**Cause**: You didn't provide `foundation_model` or `embedding_model`.

**Solution**: Always include both required parameters:

```python
search_space = AI4RAGSearchSpace(
    params=[
        Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
        Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
        # ... other params
    ]
)
```

---

### Error: "Not supported parameters were given"

**Cause**: You provided a parameter name that's not in `AI4RAGParamNames`.

**Solution**: Check the parameter name spelling and consult `ai4rag.utils.constants.AI4RAGParamNames` for valid names.

---

### Search Space Size Is Zero

**Cause**: Validation rules are filtering out all combinations.

**Solution**:

1. Check parameter value combinations manually
2. Verify chunk_size > 2 * chunk_overlap for all combinations
3. Ensure retrieval_method and window_size are consistent
4. For hybrid search, verify ranker parameter sentinels

---

### Optimizer Says "max_evals exceeded combinations"

**Cause**: You set `max_evals` higher than your search space size.

**Solution**: Reduce `max_evals` or expand your search space:

```python
search_space = AI4RAGSearchSpace(params=[...])
print(f"Max combinations: {search_space.max_combinations}")

# Set max_evals <= max_combinations
optimizer_settings = GAMOptSettings(max_evals=15)
```

---

## Related Topics

- [Optimizers](optimizers.md): How optimizers explore the search space
- [Evaluation](evaluation.md): Metrics used to score configurations
- [Hybrid Search](hybrid-search.md): Configuring hybrid search parameters
- [Quick Start](../getting-started/quick-start.md): Complete search space examples

---

## Summary

Search spaces in ai4rag:

- **Define parameters**: Specify what to optimize and possible values
- **Four types**: Categorical (`"C"`), Integer (`"I"`), Real (`"R"`), Boolean (`"B"`)
- **Required**: `foundation_model` and `embedding_model` must always be provided
- **Defaults**: Sensible defaults for chunking, retrieval, and search parameters
- **Validation rules**: Built-in rules filter invalid combinations automatically
- **Sentinel values**: `""`, `0`, and `1` indicate unused parameters
- **Custom rules**: Add domain-specific constraints with custom validation functions

Start with a minimal search space, optimize the most impactful parameters first (chunking and retrieval), then expand to explore advanced features like hybrid search and model comparison.
