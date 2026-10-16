# Notebooks & Templates

Programmatic notebook generation from RAG pattern definitions.

## Notebook Builder

::: ai4rag.assets_generator.notebook
    options:
      members:
        - Notebook
        - NotebookCell

## Template Rendering

The MaaS indexing template installs `ai4rag[text-extraction]` for the generated
notebook's ai4rag version. That extra supplies `rapidocr~=3.9.2` and
`onnxruntime~=1.25.0`, replacing separate OCR package installation commands.

`create_placeholder_mapping()` reads the pattern's `template_id` and resolves it
through the template registry. Standard and graph inference notebooks instantiate
that template. Patterns without an ID emit a `UserWarning` and use `SequentialRAG`;
unsupported IDs raise `ValueError`. Regenerate older patterns to persist the selected
template before generating new notebooks.

For graph indexing, `KG_EXTRACTION_CONFIG` comes from `settings.graph_extraction`,
then `indexing.pipeline_spec.parameters.graph_extraction_config`, with
`{"mode": "constrained"}` as the fallback. The previous `settings.knowledge_graph`
block is not read: migrate its `extraction_config` into `settings.graph_extraction`
or regenerate the pattern.

The knowledge-graph templates are named `mass_creating_knowledge_graph` and
`mass_inference_knowledge_graph` (the `mass_` prefix is part of their names).
Indexing includes S3 discovery, extraction, chunking, and graph construction; inference
includes query examples and optional benchmark evaluation snippets.

::: ai4rag.assets_generator.templates
    options:
      members:
        - generate_notebook_from_template
        - create_placeholder_mapping

## RAG Template Registry

The registry maps stable IDs to allowed template classes and import paths:

| ID | Class | Use |
|---|---|---|
| `sequential_rag` | `SequentialRAG` | Single-pass retrieval and generation |
| `agentic_rag` | `AgenticRAG` | Agent-driven retrieval and query rewriting |
| `agentic_graph_rag` | `AgenticRAG` | Neo4j graph retrieval |

Experiments publish the candidate’s selected ID, including `agentic_graph_rag` for Neo4j.

::: ai4rag.rag.template.registry
    options:
      members:
        - RAGTemplateSpec
        - get_template_spec
