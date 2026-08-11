# Data Flow

This page provides a detailed analysis of how data flows through the ai4rag system during optimization, from documents to optimal RAG configuration.

---

## Overview

ai4rag's data flow consists of four distinct phases executed for each configuration evaluated during hyperparameter optimization:

1. **Indexing Phase**: Documents → Chunks → Embeddings → Vector Store
2. **Query Phase**: Question → Retrieval → Context → Generation → Answer
3. **Evaluation Phase**: Answers + Ground Truth → Metrics → Scores
4. **Optimization Loop**: Scores → Optimizer → Next Configuration

---

## Indexing Phase

The indexing phase transforms raw documents into searchable vector embeddings stored in a vector database.

```mermaid
sequenceDiagram
    participant Exp as AI4RAGExperiment
    participant LC as BaseChunker
    participant EM as OpenAIEmbeddingModel
    participant VS as VectorStore
    participant MaaSClient as MaaS Client
    participant DB as Vector DB (Milvus/Chroma/PGVector)

    Exp->>Exp: check if collection exists
    alt Collection exists
        Note over Exp: Reuse existing collection
    else New collection needed
        Exp->>LC: split_documents(DoclingDocuments)
        activate LC
        LC->>LC: apply chunking method
        LC->>LC: set document_id and sequence_number
        LC-->>Exp: AI4RAGChunks (with metadata)
        deactivate LC

        Exp->>VS: add_documents(chunks)
        activate VS
        VS->>EM: embed_documents([chunk.text])
        activate EM
        loop Batches of 1024 chunks
            EM->>MaaSClient: embeddings.create(batch)
            MaaSClient-->>EM: embeddings
        end
        EM-->>VS: all embeddings
        deactivate EM

        loop Batches (backend-specific size: Milvus/Chroma 2048, PGVector 1024)
            VS->>DB: upsert(chunks + embeddings)
            DB-->>VS: success
        end
        VS-->>Exp: indexing complete
        deactivate VS
    end
```

### Document Processing

**Input Documents:**

Documents are provided as `list[DoclingDocument]` (from the `docling-core` library). Each document is a structured representation of a parsed file, with the document name used as `document_id`. If the name is not set, a content-based hash is generated instead.

### Chunking

The configured chunker (`DoclingChunker` or `LangChainChunker`) splits `DoclingDocument` objects into `AI4RAGChunk` instances. Both chunkers use a character-based token approximation (4 characters = 1 token) for token counting.

**DoclingChunker** (structure-aware): Operates directly on the document structure, preserving headings and tables. Does not support overlap.

**LangChainChunker** (token-based): Converts each `DoclingDocument` to markdown internally, then applies `RecursiveCharacterTextSplitter` with token-based length measurement.

**Metadata:**

Each chunk receives:

```python
AI4RAGChunk(
    text="Chunk text content...",
    metadata={
        "document_id": "original_doc_id",    # From document name or content hash
        "sequence_number": 1,                 # Position within document
        # chunker-specific keys (e.g. "start_index" for LangChain, "headings" for Docling)
    }
)
```

**Sequence numbering:**
- Sequence numbers assigned sequentially per document
- Enables window-based retrieval (adjacent chunks)

**Output:**

```python
[
    AI4RAGChunk(
        text="First chunk text...",
        metadata={"document_id": "doc1", "sequence_number": 1}
    ),
    AI4RAGChunk(
        text="Second chunk text...",
        metadata={"document_id": "doc1", "sequence_number": 2}
    ),
    # ...
]
```

### Embedding

**OpenAIEmbeddingModel** converts chunk text to vector embeddings:

**Auto-detection (on first use):**

```python
embedding_model = OpenAIEmbeddingModel(
    model_id="ollama/nomic-embed-text:latest",
    client=maas_client,
    params={"embedding_dimension": 768, "context_length": 8192}  # Optional
)
```

If `embedding_dimension` or `context_length` not provided:
- `embedding_dimension`: Detected via test embedding call (counts dimensions)
- `context_length`: Detected via binary search (64 to 8192 tokens, ~5 API calls)

**Batch Processing:**

Embeddings created in batches of 1024 chunks to respect API limits:

```python
def embed_documents(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for idx in range(0, len(texts), 1024):
        batch = texts[idx : idx + 1024]
        batch_embeddings = self.client.embeddings.create(
            input=batch,
            model=self.model_id
        )
        embeddings.extend([data.embedding for data in batch_embeddings.data])
    return embeddings
```

**Output:**

```python
[
    [0.023, -0.145, 0.678, ...],  # 768-dimensional vector for chunk 1
    [0.091, -0.023, 0.445, ...],  # 768-dimensional vector for chunk 2
    # ...
]
```

### Vector Store Insertion

The backend is selected by `vector_store_config` (a `ChromaConfig`, `MilvusConfig`, or `PGVectorConfig`) passed to `AI4RAGExperiment`. The experiment resolves the concrete store once via `get_vector_store`, which talks **directly** to the configured backend (Milvus, Chroma, or PostgreSQL/pgvector) — there is no intermediary API server between ai4rag and the vector database.

**Vector Store Selection:**

```python
from ai4rag.rag.vector_store import get_vector_store

vector_store = get_vector_store(
    embedding_model=embedding_model,
    config=vector_store_config,        # e.g. MilvusConfig.from_env(); provider="milvus"
    collection_name=collection_name,   # None to auto-generate a new "ai4rag_..." name
)
collection_name = vector_store.collection_name  # Resolved, ai4rag-prefixed name
```

**Batch Insertion (Milvus example):**

Chunks are embedded once, then upserted directly into the backend in batches (Milvus/Chroma: 2048, PGVector: 1024):

```python
embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

data = [
    {
        "chunk_id": chunk.chunk_id,
        "content": chunk.text,
        "vector": embedding,
        "chunk_content": {"content": chunk.text, "metadata": chunk.metadata, "chunk_id": chunk.chunk_id},
    }
    for chunk, embedding in zip(chunks, embeddings)
]

for idx in range(0, len(data), 2048):
    vector_store._client.upsert(collection_name, data=data[idx : idx + 2048])
```

### Collection Reuse

**Key Optimization:** Collections are reused when `indexing_params` match:

```python
indexing_params = {
    "chunking": {
        "chunking_method": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 128,
    },
    "embedding": {
        "model_id": "ollama/nomic-embed-text:latest",
        "distance_metric": "cosine",
        "embedding_dimension": 768,
        "context_length": 8192,
    }
}
```

**Reuse Logic:**

1. Hash `indexing_params` to generate canonical representation
2. Check `results.collection_names` for matching params
3. If match found, reuse `collection_name` (skip chunking, embedding, insertion)
4. If no match, create new collection

**Performance Impact:**

- Avoids re-chunking and re-embedding when only retrieval/generation params change
- Typical speedup: 50-80% per evaluation when collection reused

---

## Query Phase

The query phase retrieves relevant chunks and generates answers for benchmark questions.

```mermaid
sequenceDiagram
    participant QR as query_rag()
    participant RAG as SimpleRAG
    participant Ret as Retriever
    participant VS as VectorStore
    participant EM as OpenAIEmbeddingModel
    participant FM as OpenAIFoundationModel
    participant MaaSClient as MaaS Client
    participant DB as Vector DB (Milvus/Chroma/PGVector)

    Note over QR: Parallel execution (ThreadPoolExecutor)
    par Question 1
        QR->>RAG: generate(question_1)
    and Question 2
        QR->>RAG: generate(question_2)
    and Question N
        QR->>RAG: generate(question_N)
    end

    activate RAG
    RAG->>Ret: retrieve(question)
    activate Ret
    Ret->>VS: search(question, k, search_mode, ranker_*)
    activate VS
    VS->>EM: embed_query(question)
    activate EM
    EM->>MaaSClient: embeddings.create(question)
    MaaSClient-->>EM: query_embedding
    EM-->>VS: query_embedding
    deactivate EM

    alt search_mode == "vector"
        VS->>DB: dense vector query
        DB-->>VS: ranked chunks
    else search_mode == "hybrid"
        VS->>DB: dense + sparse/keyword query (+ fusion)
        DB-->>VS: re-ranked chunks (dense + sparse)
    end

    VS-->>Ret: retrieved documents
    deactivate VS
    Ret-->>RAG: reference_documents
    deactivate Ret

    RAG->>RAG: format context using context_template_text
    RAG->>RAG: format user message using user_message_text
    RAG->>FM: chat(messages)
    activate FM
    FM->>MaaSClient: chat.completions.create(model, messages, params)
    MaaSClient-->>FM: response
    FM-->>RAG: answer
    deactivate FM

    RAG-->>QR: {"answer": ..., "reference_documents": ..., "question": ...}
    deactivate RAG
```

### Parallel Query Execution

**ThreadPoolExecutor** processes benchmark questions concurrently:

```python
def query_rag(
    rag: BaseRAGTemplate,
    questions: list[str],
    max_threads: int = 10
) -> list[dict]:

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        responses = list(executor.map(
            partial(_generate_response, rag=rag),
            questions
        ))
    return responses
```

**Concurrency:**
- Default: 10 concurrent threads
- Balances throughput with server load
- Each thread executes full retrieval + generation pipeline independently

### Retrieval

**Retriever** delegates directly to the vector store's `search()` method, passing the raw query text — embedding and backend querying both happen **inside** the store:

**Step 1: Delegate to the Vector Store**

```python
reference_documents = vector_store.search(
    question,
    k=number_of_chunks,
    search_mode=search_mode,        # "vector" or "hybrid"
    ranker_strategy=ranker_strategy,
    ranker_k=ranker_k,
    ranker_alpha=ranker_alpha,
)
# reference_documents: list[AI4RAGChunk] — returned directly, no separate conversion step
```

**Step 2: Query Embedding (internal to the vector store)**

Every concrete store's `search()` embeds the raw query on entry:

```python
query_embedding = self.embedding_model.embed_query(query)
# Returns: [0.123, -0.456, 0.789, ...]  (same dimension as document embeddings)
```

**Vector Mode (semantic search only):**

Each backend then runs its native similarity search against `query_embedding` — e.g. `MilvusClient.search` on the dense `vector` field, or PGVector's `<=>`-style distance query — returning the top-k chunks by the configured distance metric.

**Hybrid Mode (semantic + keyword):**

```python
reference_documents = vector_store.search(
    question,
    k=5,
    search_mode="hybrid",
    ranker_strategy="rrf",   # or "weighted", "normalized"
    ranker_k=60,             # applies to "rrf"
    # ranker_alpha=0.7,      # applies to "weighted"
)
```

Combines the dense vector search with a backend-native sparse/keyword search — Milvus's server-side BM25 field fused via `RRFRanker`/`WeightedRanker`, or PGVector's `tsvector` full-text search fused in-memory via `WeightedInMemoryAggregator` — then returns the fused top-k chunks. Chroma does not support `search_mode="hybrid"`.

### Context Formatting

**SimpleRAG** formats retrieved chunks into LLM context:

```python
# Default context_template_text: "{document}\n"
context = "\n".join([
    foundation_model.context_template_text.format(document=chunk.text)
    for chunk in reference_documents
])
```

**Example output:**

```
According to the document: "First retrieved chunk text..."

According to the document: "Second retrieved chunk text..."

...
```

**Customization:**

```python
foundation_model = OpenAIFoundationModel(
    model_id="ollama/llama3.2:3b",
    client=client,
    context_template_text="Source {document}\n---\n"  # Custom format
)
```

### User Message Construction

Combine context with question:

```python
# Default user_message_text:
# "{reference_documents}\n\nQuestion: {question}\nAnswer:"

user_message = foundation_model.user_message_text.format(
    reference_documents=context,
    question=question
)
```

**Example output:**

```
According to the document: "Chunk 1..."

According to the document: "Chunk 2..."

Question: What is the capital of France?
Answer:
```

### Generation

**OpenAIFoundationModel** generates answer via chat completion:

```python
messages = [
    {
        "role": "system",
        "content": foundation_model.system_message_text
        # Default: "You are a helpful, respectful and honest assistant..."
    },
    {
        "role": "user",
        "content": user_message
    }
]

response = client.chat.completions.create(
    model=foundation_model.model_id,
    messages=messages,
    max_completion_tokens=1024,
    temperature=0.1
)

answer = response.choices[0].message.content
```

**Output:**

```python
{
    "answer": "Based on the provided documents, Paris is the capital of France.",
    "reference_documents": [Document(...), Document(...), ...],
    "question": "What is the capital of France?"
}
```

---

## Evaluation Phase

The evaluation phase compares generated answers against ground truth using multiple evaluator types and merges results.

```mermaid
sequenceDiagram
    participant Exp as AI4RAGExperiment
    participant Builder as build_evaluation_data()
    participant UE as UnitxtEvaluator
    participant JE as LLMaJEvaluator
    participant CM as custom_metrics

    Exp->>Builder: build_evaluation_data(benchmark_data, inference_response)
    activate Builder
    loop For each question
        Builder->>Builder: extract answer, contexts, context_ids
        Builder->>Builder: match with ground_truths, ground_truth_doc_ids
        Builder->>Builder: create EvaluationData instance
    end
    Builder-->>Exp: evaluation_data: list[EvaluationData]
    deactivate Builder

    Note over Exp: Route metrics to matching evaluators

    Exp->>UE: evaluate_metrics(eval_data, unitxt_metrics)
    activate UE
    UE-->>Exp: EvaluationMetricsResult (faithfulness, answer_correctness, ...)
    deactivate UE

    Exp->>JE: evaluate_metrics(eval_data, judge_metrics)
    activate JE
    JE-->>Exp: EvaluationMetricsResult (answer_relevance)
    deactivate JE

    Exp->>Exp: merge_evaluation_results(partial_results)

    Exp->>CM: apply_custom_metrics(merged, metrics)
    activate CM
    CM-->>Exp: overall_score appended
    deactivate CM
```

### EvaluationData Assembly

**build_evaluation_data()** combines inference results with benchmark data:

**Input: inference_response**

```python
[
    {
        "question": "What is X?",
        "answer": "Based on the documents, X is...",
        "reference_documents": [Document(...), Document(...)]
    },
    # ... one per benchmark question
]
```

**Input: benchmark_data**

```python
BenchmarkData with:
- questions: ["What is X?", ...]
- correct_answers: [["X is...", "Alternative answer"], ...]
- document_ids: [["doc1", "doc3"], ...]  # Ground truth source docs
- questions_ids: ["q0", "q1", ...]
```

**Output: EvaluationData list**

```python
[
    EvaluationData(
        question="What is X?",
        answer="Based on the documents, X is...",
        contexts=[
            "Retrieved chunk 1 text...",
            "Retrieved chunk 2 text..."
        ],
        context_ids=["doc1", "doc1"],  # Source document IDs
        ground_truths=["X is...", "Alternative answer"],
        ground_truths_context_ids=["doc1", "doc3"],
        question_id="q0"
    ),
    # ...
]
```

### Unitxt Evaluation

**UnitxtEvaluator** wraps the unitxt library for metric computation:

**Metrics Evaluated:**

| ai4rag Metric | Unitxt Metric | Description |
|---------------|---------------|-------------|
| `faithfulness` | `metrics.rag.external_rag.faithfulness` | How well is the answer grounded in retrieved context? |
| `answer_correctness` | `metrics.rag.external_rag.answer_correctness` | How correct is the answer vs ground truth? |
| `context_correctness` | `metrics.rag.external_rag.context_correctness` | How relevant are retrieved docs vs ground truth docs? |

**Evaluation Process:**

1. **Convert to DataFrame**:

```python
df = pd.DataFrame([data.to_dict() for data in evaluation_data])
```

2. **Call unitxt.evaluate()**:

```python
scores_df, ci_table = evaluate(
    df,
    metric_names=[
        "metrics.rag.external_rag.faithfulness",
        "metrics.rag.external_rag.answer_correctness",
        "metrics.rag.external_rag.context_correctness"
    ],
    compute_conf_intervals=True
)
```

3. **Extract Aggregate Scores** (mean + confidence intervals):

```python
{
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
}
```

4. **Extract Per-Question Scores**:

```python
{
    "faithfulness": {
        "q0": 0.71,
        "q1": 0.73,
        "q2": 0.68,
        ...
    },
    "answer_correctness": {
        "q0": 0.65,
        "q1": 0.70,
        ...
    },
    ...
}
```

### Optimization Score Calculation

The final optimization score is extracted from the aggregate results:

```python
optimization_metric = "overall_score"  # Default; user-configurable
final_score = next(
    m["scores"]["mean"] for m in result_scores["metrics"]
    if m["name"] == optimization_metric
)
# Example: 0.75
```

This single scalar value is returned to the optimizer.

---

## Optimization Loop

The optimization loop iterates configurations until the evaluation budget is exhausted.

```mermaid
flowchart TD
    Start([Start Experiment]) --> MPS{Models Pre-Selection<br/>needed?}

    MPS -->|Yes| RunMPS[Run MPS on sample]
    RunMPS --> UpdateSS[Update search space<br/>with selected models]
    UpdateSS --> InitOpt[Initialize Optimizer]

    MPS -->|No| InitOpt

    InitOpt --> RandomPhase[Random Phase:<br/>Evaluate n_random_nodes]
    RandomPhase --> CheckBudget1{Reached<br/>max_evals?}

    CheckBudget1 -->|Yes| ReturnBest
    CheckBudget1 -->|No| GAMPhase

    GAMPhase[GAM Phase:<br/>Train model on evaluations]
    GAMPhase --> Predict[Predict scores for<br/>remaining configs]
    Predict --> SelectTop[Select top evals_per_trial<br/>configurations]
    SelectTop --> EvalSelected[Evaluate selected configs]

    EvalSelected --> CheckBudget2{Reached<br/>max_evals?}
    CheckBudget2 -->|No| GAMPhase
    CheckBudget2 -->|Yes| ReturnBest

    ReturnBest([Return Best Configuration])
```

### Iteration Details

Each optimizer iteration follows this flow:

**Random Phase (GAMOptimizer only):**

```python
def evaluate_initial_random_nodes():
    successful = count(evaluations where score is not None)

    while successful < n_random_nodes and len(evaluations) < max_evals:
        config = random.choice(unevaluated_combinations)
        score = objective_function(config)  # Calls run_single_evaluation()
        evaluations.append({"config": config, "score": score})
        if score is not None:
            successful += 1
```

**GAM Phase:**

```python
def _run_iteration():
    # 1. Train GAM
    successful_evals = [e for e in evaluations if e["score"] is not None]
    X_train = encode_categorical_params(successful_evals)
    y_train = [e["score"] for e in successful_evals]
    gam = LinearGAM().fit(X_train, y_train)

    # 2. Predict
    remaining = get_unevaluated_configs()
    X_predict = encode_categorical_params(remaining)
    predictions = gam.predict(X_predict)

    # 3. Select top N
    ranked = sorted(
        zip(remaining, predictions),
        key=lambda x: x[1],
        reverse=True  # Highest predictions first
    )
    top_n = ranked[:evals_per_trial]

    # 4. Evaluate
    for config, predicted_score in top_n:
        actual_score = objective_function(config)
        evaluations.append({"config": config, "score": actual_score})
```

### Results Caching

**Two-level caching prevents redundant work:**

**Level 1: Results Cache (entire evaluation)**

```python
def results.evaluation_explored_or_cached(indexing_params, rag_params) -> float | None:
    cache_key = hash((indexing_params, rag_params))
    if cache_key in cached_evaluations:
        return cached_evaluations[cache_key]["final_score"]
    return None
```

If exact `(indexing_params, rag_params)` evaluated before, return cached score immediately.

**Level 2: Collection Reuse (indexing only)**

```python
def results.get_existing_collection(indexing_params) -> str | None:
    cache_key = hash(indexing_params)
    if cache_key in existing_collections:
        return existing_collections[cache_key]["collection_name"]
    return None
```

If `indexing_params` match but `rag_params` differ, reuse collection (skip indexing, but re-run retrieval/generation/evaluation).

**Cache Hit Rates:**

- Full cache hit: ~1% of time (optimizer rarely suggests exact duplicate)
- Collection reuse: ~40-60% of time (common when varying only retrieval/generation params)

### Event Streaming

Throughout the loop, **event handlers** receive status updates and results:

**on_status_change() calls:**

```python
# Chunking
event_handler.on_status_change(
    level=LogLevel.INFO,
    message="Chunking documents using recursive method...",
    step=ExperimentStep.CHUNKING
)

# Embedding
event_handler.on_status_change(
    level=LogLevel.INFO,
    message="Embedding chunks using ollama/nomic-embed-text:latest...",
    step=ExperimentStep.EMBEDDING
)

# Generation
event_handler.on_status_change(
    level=LogLevel.INFO,
    message="Retrieval and generation using collection 'xyz'...",
    step=ExperimentStep.GENERATION
)

# Evaluation
event_handler.on_status_change(
    level=LogLevel.INFO,
    message="Evaluating RAG Pattern 'Pattern5'...",
    step="evaluation"
)
```

**on_pattern_creation() calls:**

After each evaluation completes:

```python
event_handler.on_pattern_creation(
    payload={
        "name": "Pattern5",
        "iteration": 4,
        "max_combinations": 24,
        "duration_seconds": 134,
        "evaluation": {
            "metrics": [
                {"name": "faithfulness", "evaluator": "unitxt", "description": "...",
                 "scores": {"mean": 0.72, "ci_low": 0.61, "ci_high": 0.83}},
                {"name": "overall_score", "evaluator": "custom", "description": "...",
                 "scores": {"mean": 0.75, ...}, "optimization_metric": True},
            ],
        },
        "settings": {
            "chunking": {...},
            "embedding": {...},
            "retrieval": {...},
            "generation": {...},
            "vector_store_binding": {
                "provider_type": "milvus",  # From vector_store_config.provider
                "collection_name": "ai4rag_20260701120000_ab12cd34",
            },
        },
    },
    evaluation_results=[
        {"question": ..., "answer": ..., "metrics": [
            {"name": "faithfulness", "evaluator": "unitxt", "score": 0.71}, ...
        ]},
        ...
    ],
)
```

---

## Performance Optimizations

### 1. Parallel Query Execution

**Impact:** 5-10x speedup for query phase

**Implementation:**

```python
with ThreadPoolExecutor(max_workers=10) as executor:
    responses = list(executor.map(generate_fn, questions))
```

**Trade-offs:**
- More threads = higher throughput but more server load
- 10 threads balances throughput with API rate limits

### 2. Batch Embedding

**Impact:** 2-3x speedup for indexing phase

**Implementation:**

```python
for idx in range(0, len(texts), 1024):
    batch = texts[idx : idx + 1024]
    embeddings.extend(embed_batch(batch))
```

**Constraints:**
- MaaS max batch size: 1024 chunks
- Smaller batches = more API calls = slower

### 3. Collection Reuse

**Impact:** 50-80% speedup when indexing params unchanged

**Cache Key:**

```python
cache_key = hash((
    chunking_method,
    chunk_size,
    chunk_overlap,
    embedding_model_id,
    embedding_dimension
))
```

**Reuse Rate:**
- Typical search spaces: 40-60% reuse
- Search spaces varying only retrieval/generation: 80-95% reuse

### 4. Results Caching

**Impact:** Instant return for duplicate evaluations

**Frequency:**
- Random optimizer: ~0-1% hit rate (no duplicates by design)
- GAM optimizer: ~1-5% hit rate (rare duplicate suggestions)

### 5. Early Stopping (Models Pre-Selection)

**Impact:** 30-70% reduction in total evaluations

**Example:**
- Without MPS: 10 models × 5 embeddings × 4 configs = 200 evaluations
- With MPS: 3 models × 2 embeddings × 4 configs = 24 evaluations (88% reduction)

---

## Data Transformations Summary

**Documents → Chunks:**

```python
DoclingDocument(name="doc1", ...)
↓ (DoclingChunker or LangChainChunker)
[
    AI4RAGChunk(text="Chunk 1", metadata={"document_id": "doc1", "sequence_number": 1}),
    AI4RAGChunk(text="Chunk 2", metadata={"document_id": "doc1", "sequence_number": 2}),
    ...
]
```

**Chunks → Embeddings:**

```python
["Chunk 1 text", "Chunk 2 text", ...]
↓ (OpenAIEmbeddingModel)
[[0.1, -0.2, ...], [0.3, 0.1, ...], ...]  # 768-dim vectors
```

**Embeddings → Vector Store:**

```python
[{"content": "Chunk 1", "embedding": [0.1, ...], "metadata": {...}}, ...]
↓ (MilvusVectorStore.add_documents / ChromaVectorStore.add_documents / PGVectorStore.add_documents)
Collection "xyz" in the configured backend (Milvus, Chroma, or PGVector)
```

**Question → Retrieved Chunks:**

```python
"What is the capital of France?"
↓ (MilvusVectorStore.search — embeds via OpenAIEmbeddingModel.embed_query internally,
   then queries Milvus directly; ChromaVectorStore/PGVectorStore follow the same
   embed-then-query pattern against their own backend)
[
    AI4RAGChunk(text="Paris is the capital...", metadata={...}),
    AI4RAGChunk(text="France's capital city...", metadata={...}),
    ...
]
```

**Chunks + Question → Answer:**

```python
contexts = ["Paris is the capital...", "France's capital city..."]
question = "What is the capital of France?"
↓ (SimpleRAG.generate via OpenAIFoundationModel.chat)
"Based on the provided documents, Paris is the capital of France."
```

**Answer + Ground Truth → Scores:**

```python
{
    "question": "What is the capital of France?",
    "answer": "Paris is the capital of France.",
    "ground_truths": ["Paris"],
    "contexts": [...],
    "ground_truths_context_ids": [...]
}
↓ (Multi-Evaluator Dispatch → merge → custom metrics)
{
    "metrics": [
        {"name": "faithfulness", "evaluator": "unitxt", ...,
         "scores": {"mean": 0.95, "ci_low": 0.90, "ci_high": 1.0}},
        {"name": "answer_relevance", "evaluator": "judge", ...,
         "scores": {"mean": 0.92, "ci_low": 0.85, "ci_high": 0.98}},
        {"name": "overall_score", "evaluator": "custom", ...,
         "scores": {"mean": 0.91, ...}},
    ],
    "question_scores": [...]
}
```

---

## Error Handling

**Failed Iterations:**

When an evaluation fails, the optimizer continues:

```python
try:
    score = run_single_evaluation(config)
except AI4RAGError as err:
    exception_handler.handle_exception(err)
    raise FailedIterationError(msg) from err
```

**In optimizer:**

```python
try:
    score = objective_function(config)
except FailedIterationError:
    score = None  # Don't penalize, just skip

evaluations.append({"config": config, "score": score})
```

**Result:** Failed evaluations don't stop optimization but don't contribute to GAM training.

**Common Failures:**

- **IndexingError**: Embedding API timeout, vector store connection issue
- **GenerationError**: LLM API error, rate limit exceeded
- **EvaluationError**: Malformed unitxt data, metric computation failure

---

## Next Steps

- [Core Components](core-components.md) - Detailed component documentation
- [RAG Components](rag-components.md) - RAG pipeline component details
- [Architecture Overview](overview.md) - High-level design
