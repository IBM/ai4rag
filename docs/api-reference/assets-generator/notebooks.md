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

Both MaaS and knowledge-graph indexing templates use this extra and default to
connected mode. Set `RHOAI_NOTEBOOK_CLUSTER_MODE=disconnected` in a workbench;
`NOTEBOOK_CLUSTER_MODE` takes precedence when set, including in AutoX Jobs. Both
variables accept `connected` or `disconnected`. Run All uses the selected mode
without skipping cells; disconnected mode sets Hugging Face and Transformers
offline flags, while connected mode clears them.

After document discovery, the notebooks call `prepare_extraction()` to select only
the models needed by the corpus. Connected runs download pinned PP-OCRv4 or Whisper
Tiny models when local paths are unset. Disconnected runs validate the mounted
`DOCLING_ARTIFACTS_PATH` and `HF_MODEL_DIR` bundles for PDF/image and audio inputs,
respectively; text-only inputs require neither. The pattern language selects Chinese
OCR for Chinese codes and English otherwise. Extraction uses one worker to reduce
memory use. See [Text Extraction](../../user-guide/pipeline-components.md#text-extraction).

`create_placeholder_mapping()` reads the pattern's `template_id` and resolves it
through the template registry. Standard and graph inference notebooks instantiate
that template. Patterns without an ID emit a `UserWarning` and use `SequentialRAG`;
unsupported IDs raise `ValueError`. Regenerate older patterns to persist the selected
template before generating new notebooks.

For agentic templates, `settings.agentic.enable_context_compression` controls the
generated constructor's boolean `enable_context_compression` argument and defaults
to `True` when absent. Set it to `False` in the pattern dictionary before rendering
to disable the [context-overflow fallback](../../architecture/rag-components.md#context-overflow-fallback).
Sequential templates receive no compression argument.

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
