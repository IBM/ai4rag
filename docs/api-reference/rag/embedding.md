# Embedding Models API

::: ai4rag.rag.embedding.openai_model.OpenAIEmbeddingModel
    options:
      show_root_heading: true
      show_source: true

## Request Policy

`OpenAIEmbeddingModel.embed_documents()` accepts a `policy` that controls recovery
for each request batch. It defaults to `INDEXING_POLICY`. Import `QUERY_POLICY` from
`ai4rag.rag.embedding.request_executor` to use shorter waits for transient errors.
[Embedding recovery](../../architecture/rag-components.md#openaiembeddingmodel)

::: ai4rag.rag.embedding.request_executor.RequestPolicy
    options:
      show_root_heading: true
      show_source: true
