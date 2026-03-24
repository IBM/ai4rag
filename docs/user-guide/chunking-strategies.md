# Chunking Strategies

ai4rag supports multiple document chunking strategies to handle different document formats effectively.
Beyond the default recursive character splitting, markdown-aware splitters preserve document structure, and chunk metadata enrichment gives the LLM visibility into where each chunk originates.

## Available Chunking Methods

| Method | Splitter | Description |
|--------|----------|-------------|
| `"recursive"` | `RecursiveCharacterTextSplitter` | Default. Splits by character separators recursively. Best for plain text. |
| `"markdown"` | `MarkdownTextSplitter` | Like recursive, but uses markdown-aware separators (headers, code blocks, lists). |
| `"markdown_header"` | `MarkdownHeaderTextSplitter` | Splits at markdown header boundaries. Preserves header hierarchy in metadata. |

All methods are provided through LangChain text splitters and accessed via the chunker factory.

---

## When to Use Markdown Splitting

Consider markdown-aware chunking when:

- **Knowledge base is in Markdown**: Documentation sites, wikis, README files
- **Document structure matters**: Headers define logical sections that should stay together
- **Metadata enrichment is needed**: Header hierarchy (Section > Subsection) can be passed to the LLM for better grounding
- **Chunks cross logical boundaries**: Recursive splitting may break a section mid-paragraph

**Example scenarios**:

- Technical documentation with clear header hierarchy
- API documentation with structured sections
- Wiki pages or knowledge base articles
- Project READMEs and guides

---

## Configuration

### Method: `recursive` (default)

Splits text using a cascade of separators (`\n\n`, sentence boundaries, `\n`, spaces). Requires `chunk_size > 0`.

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils.constants import AI4RAGParamNames

Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, param_type="C", values=["recursive"]),
Parameter(name=AI4RAGParamNames.CHUNK_SIZE, param_type="C", values=[512, 1024, 2048]),
Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, param_type="C", values=[128, 256]),
```

### Method: `markdown`

Uses markdown-specific separators (headers, code fences, lists) instead of generic character separators. Like `recursive`, it requires `chunk_size > 0` and respects `chunk_overlap`.

```python
Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, param_type="C", values=["markdown"]),
Parameter(name=AI4RAGParamNames.CHUNK_SIZE, param_type="C", values=[512, 1024]),
Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, param_type="C", values=[128]),
```

### Method: `markdown_header`

Splits documents at header boundaries (`#`, `##`, `###`). Each resulting chunk carries header metadata (`Header 1`, `Header 2`, `Header 3`) describing its position in the document hierarchy.

This method supports two modes:

#### Pure structural splitting (chunk_size = 0)

Chunks are determined entirely by header boundaries. No size constraint is applied. Useful when sections are naturally short.

```python
Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, param_type="C", values=["markdown_header"]),
Parameter(name=AI4RAGParamNames.CHUNK_SIZE, param_type="C", values=[0]),
Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, param_type="C", values=[0]),
```

#### Structural splitting with refinement (chunk_size > 0)

After splitting by headers, any chunk exceeding `chunk_size` is further split using `RecursiveCharacterTextSplitter`. This ensures chunks fit within embedding model context limits while preserving header metadata.

```python
Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, param_type="C", values=["markdown_header"]),
Parameter(name=AI4RAGParamNames.CHUNK_SIZE, param_type="C", values=[0, 1024]),
Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, param_type="C", values=[0, 128]),
```

!!! tip "Custom Headers"
    By default, `markdown_header` splits on `#` (Header 1), `##` (Header 2), and `###` (Header 3). You can customize this when creating a chunker directly:
    ```python
    from ai4rag.rag.chunking import LangChainChunker

    chunker = LangChainChunker(
        method="markdown_header",
        chunk_size=1024,
        chunk_overlap=128,
        headers_to_split_on=[("#", "Title"), ("##", "Section"), ("###", "Subsection"), ("####", "Detail")],
    )
    ```

---

## Chunk Metadata Enrichment

When using `markdown_header`, each chunk carries header metadata describing its location in the document. The `include_chunk_metadata` parameter controls whether this metadata is included in the LLM context.

### How It Works

When `include_chunk_metadata=True`, each retrieved chunk is prefixed with source and section information before being passed to the LLM:

```
Source: installation_guide.md
Section: Getting Started > Prerequisites

To install the package, run pip install ai4rag.
```

When `include_chunk_metadata=False` (default), only the raw `page_content` is passed to the LLM — the same behavior as before this feature was introduced.

### Configuration

Add `include_chunk_metadata` to your search space to let the optimizer decide:

```python
Parameter(
    name=AI4RAGParamNames.INCLUDE_CHUNK_METADATA,
    param_type="C",
    values=[True, False],
)
```

Or force it on:

```python
Parameter(
    name=AI4RAGParamNames.INCLUDE_CHUNK_METADATA,
    param_type="C",
    values=[True],
)
```

!!! note "Works with any chunking method"
    `include_chunk_metadata` works regardless of the chunking method. For non-markdown methods, it adds a `Source: <document_id>` prefix. For `markdown_header`, it adds both source and section hierarchy.

---

## Search Space Validation Rules

ai4rag enforces consistency between chunking method and chunk size parameters:

### Rule: Chunk Parameters Consistency with Method

- **`recursive` and `markdown`**: `chunk_size` must be > 0 (these methods require a size constraint)
- **`markdown_header`**: `chunk_size` may be 0 (pure structural splitting) or > 0 (with refinement)

### Rule: Chunk Size Bigger Than Overlap

- When `chunk_size > 0`: `chunk_size > 2 * chunk_overlap` must hold
- When `chunk_size == 0`: the rule is skipped (structural-only splitting)

!!! example "Valid Configurations"
    ```python
    # Recursive — chunk_size required
    {"chunking_method": "recursive", "chunk_size": 1024, "chunk_overlap": 128}  # OK

    # Markdown header — pure structural
    {"chunking_method": "markdown_header", "chunk_size": 0, "chunk_overlap": 0}  # OK

    # Markdown header — with refinement
    {"chunking_method": "markdown_header", "chunk_size": 1024, "chunk_overlap": 128}  # OK
    ```

!!! failure "Invalid Configurations"
    ```python
    # Recursive with chunk_size=0 — not allowed
    {"chunking_method": "recursive", "chunk_size": 0, "chunk_overlap": 0}  # ERROR

    # Overlap too large relative to chunk_size
    {"chunking_method": "markdown", "chunk_size": 256, "chunk_overlap": 200}  # ERROR
    ```

---

## Code Examples

### Example 1: Compare Chunking Methods

Let the optimizer explore different chunking strategies:

```python
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

search_space = AI4RAGSearchSpace(
    params=[
        # ... (models as usual)

        # Explore all chunking methods
        Parameter(
            name=AI4RAGParamNames.CHUNKING_METHOD,
            param_type="C",
            values=["recursive", "markdown", "markdown_header"],
        ),
        # chunk_size=0 is valid only for markdown_header (pure structural)
        Parameter(name=AI4RAGParamNames.CHUNK_SIZE, param_type="C", values=[0, 512, 1024]),
        Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, param_type="C", values=[0, 128]),

        # Let optimizer decide on metadata enrichment
        Parameter(name=AI4RAGParamNames.INCLUDE_CHUNK_METADATA, param_type="C", values=[True, False]),

        # Retrieval
        Parameter(name=AI4RAGParamNames.RETRIEVAL_METHOD, param_type="C", values=["simple"]),
        Parameter(name=AI4RAGParamNames.NUMBER_OF_CHUNKS, param_type="C", values=[5, 10]),
    ]
)
```

!!! note "Validation"
    Invalid combinations (e.g., `recursive` + `chunk_size=0`) are automatically filtered out by built-in validation rules.

---

### Example 2: Markdown-Optimized Pipeline

Focus on markdown documents with header-aware splitting and metadata enrichment:

```python
search_space = AI4RAGSearchSpace(
    params=[
        # ... (models as usual)

        # Markdown header splitting only
        Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, param_type="C", values=["markdown_header"]),
        # Pure structural vs refined
        Parameter(name=AI4RAGParamNames.CHUNK_SIZE, param_type="C", values=[0, 1024, 2048]),
        Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, param_type="C", values=[0, 128]),

        # Always enrich with metadata
        Parameter(name=AI4RAGParamNames.INCLUDE_CHUNK_METADATA, param_type="C", values=[True]),

        Parameter(name=AI4RAGParamNames.RETRIEVAL_METHOD, param_type="C", values=["simple"]),
        Parameter(name=AI4RAGParamNames.NUMBER_OF_CHUNKS, param_type="C", values=[3, 5, 7]),
    ]
)
```

---

### Example 3: Using the Chunker Factory Directly

The chunker factory creates the appropriate chunker instance based on the method name:

```python
from ai4rag.rag.chunking.chunker_factory import get_chunker

# Recursive chunker
chunker = get_chunker(chunking_method="recursive", chunk_size=1024, chunk_overlap=128)

# Markdown-aware chunker
chunker = get_chunker(chunking_method="markdown", chunk_size=1024, chunk_overlap=128)

# Markdown header chunker (pure structural)
chunker = get_chunker(chunking_method="markdown_header", chunk_size=0, chunk_overlap=0)

# Markdown header chunker (with refinement)
chunker = get_chunker(chunking_method="markdown_header", chunk_size=1024, chunk_overlap=128)
```

---

## Default Search Space

The default search space includes all three chunking methods and metadata enrichment:

```python
from ai4rag.search_space.src.default_search_space import get_default_ai4rag_search_space_parameters

params = get_default_ai4rag_search_space_parameters()
# Includes:
#   chunking_method: ("recursive", "markdown", "markdown_header")
#   chunk_size: (1024, 2048, 4096)
#   chunk_overlap: (128, 256)
#   include_chunk_metadata: (False, True)
```

---

## Best Practices

### 1. Match Method to Document Format

- **Plain text / mixed formats**: Use `recursive`
- **Markdown with consistent headers**: Use `markdown_header`
- **Markdown without clear header structure**: Use `markdown`

### 2. Use Refinement for Long Sections

If your markdown documents have long sections under a single header, enable refinement with `chunk_size > 0` to avoid exceeding embedding model context limits.

### 3. Enable Metadata Enrichment with Header Splitting

`include_chunk_metadata=True` is most valuable with `markdown_header`, where header hierarchy provides meaningful context to the LLM. With `recursive`, it only adds the source document ID.

### 4. Let the Optimizer Compare

Include multiple methods and let the HPO engine determine which works best for your data:

```python
Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, param_type="C", values=["recursive", "markdown", "markdown_header"]),
```

### 5. Consider Search Space Size

Adding multiple chunking methods multiplies the search space. If evaluation is expensive, consider fixing the chunking method and only optimizing size/overlap, or increasing `max_evals`.

---

## Related Topics

- [Search Space Configuration](search-space.md): Defining and constraining parameter search spaces
- [Hybrid Search](hybrid-search.md): Combining vector and keyword search
- [Evaluation Metrics](evaluation.md): Understanding how chunking impacts evaluation scores
- [Chunking API Reference](../api-reference/rag/chunking.md): API documentation for chunker classes
