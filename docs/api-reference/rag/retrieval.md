# Retrieval API

## Retriever

::: ai4rag.rag.retrieval.retriever.Retriever
    options:
      show_root_heading: true
      show_source: true

## Document Compressor

`LLMDocumentCompressor` uses a LangChain chat model to extract query-relevant,
verbatim excerpts from `AI4RAGChunk` objects while preserving their metadata.
`AgenticRAG` uses it for its
[context-overflow fallback](../../architecture/rag-components.md#context-overflow-fallback).
Both `Retriever` and `LLMDocumentCompressor` are exported from `ai4rag.rag.retrieval`.

::: ai4rag.rag.retrieval.document_compressor.LLMDocumentCompressor
    options:
      show_root_heading: true
      show_source: true
