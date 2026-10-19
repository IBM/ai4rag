# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import warnings
from pathlib import Path
from typing import Any

from ai4rag import __version__
from ai4rag.assets_generator.notebook import Notebook
from ai4rag.rag.template.registry import get_template_spec
from ai4rag.rag.vector_store import get_vector_store_env_vars
from ai4rag.rag.vector_store.neo4j import Neo4jGraphRetrievalConfig


def _format_required_env_vars(provider: str) -> str:
    """Render a provider's environment variables as a Markdown bullet list.

    Parameters
    ----------
    provider : str
        Vector store backend discriminator (e.g. ``"milvus"``). An empty or
        unsupported value yields an empty string so notebook generation never
        fails on partial pattern data.

    Returns
    -------
    str
        One ``- `NAME` — description`` bullet per variable, joined by newlines;
        empty string when *provider* is unknown or missing.
    """
    if not provider:
        return ""
    try:
        env_vars = get_vector_store_env_vars(provider)
    except ValueError:
        return ""
    return "\n".join(f"- `{name}` — {description}" for name, description in env_vars)


def _template_placeholder_mapping(output_data: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Return notebook placeholders determined by the selected RAG template."""
    template_id = output_data.get("template_id")
    if template_id is None:
        # Patterns written before template_id was added generated SequentialRAG
        # notebooks. Retain that behaviour during the schema transition.
        warnings.warn(
            "Pattern has no template_id; generating a legacy SequentialRAG notebook. "
            "Regenerate the pattern to persist its selected template.",
            UserWarning,
            stacklevel=3,
        )
        template_id = "sequential_rag"
    template_spec = get_template_spec(template_id)
    enable_context_compression = settings.get("agentic", {}).get("enable_context_compression", True)
    return {
        "TEMPLATE_ID": template_spec.template_id,
        "RAG_TEMPLATE_IMPORT": f"from {template_spec.import_path} import {template_spec.template_class.__name__}",
        "RAG_TEMPLATE_CLASS": template_spec.template_class.__name__,
        "RAG_TEMPLATE_ARGUMENTS": (
            f", enable_context_compression={enable_context_compression!r}"
            if template_spec.template_id in {"agentic_rag", "agentic_graph_rag"}
            else ""
        ),
    }


def _generation_placeholder_mapping(settings: dict[str, Any]) -> dict[str, Any]:
    """Return notebook placeholders for the foundation model configuration."""
    foundation_model = settings.get("generation", {})
    return {
        "FM_MODEL_ID": foundation_model.get("model_id", ""),
        "SYSTEM_MESSAGE": foundation_model.get("system_message_text", ""),
        "USER_MESSAGE": foundation_model.get("user_message_text", ""),
        "CONTEXT_TEXT": foundation_model.get("context_template_text", ""),
        "FM_PARAMS": {
            key: foundation_model[key] for key in ("temperature", "max_completion_tokens") if key in foundation_model
        },
        "LANGUAGE": foundation_model.get("language", {"code": "", "name": "auto"}),
    }


def _store_placeholder_mapping(settings: dict[str, Any]) -> dict[str, Any]:
    """Return notebook placeholders for the embedding and vector store configuration."""
    embedding_model = settings.get("embedding", {})
    store_binding = settings.get("store_binding", {})
    provider_type = store_binding.get("provider_type", "")
    return {
        "EMBEDDING_MODEL_ID": embedding_model.get("model_id", ""),
        "EMBEDDING_PARAMS": embedding_model.get("embedding_params", {"embedding_dimension": 768}),
        "PROVIDER_TYPE": provider_type,
        "COLLECTION_NAME": store_binding.get("collection_name", ""),
        "REQUIRED_ENV_VARS": _format_required_env_vars(provider_type),
    }


def _retrieval_placeholder_mapping(settings: dict[str, Any]) -> dict[str, Any]:
    """Return notebook placeholders for retrieval configuration."""
    retrieval = settings.get("retrieval", {})
    return {
        "RETRIEVAL_METHOD": retrieval.get("method", ""),
        "NUMBER_OF_CHUNKS": retrieval.get("number_of_chunks", 5),
        "SEARCH_MODE": retrieval.get("search_mode"),
        "RANKER_STRATEGY": retrieval.get("ranker_strategy"),
        "RANKER_K": retrieval.get("ranker_k"),
        "RANKER_ALPHA": retrieval.get("ranker_alpha"),
        "GRAPH_RETRIEVAL_CONFIG": {key: retrieval[key] for key in Neo4jGraphRetrievalConfig.keys() if key in retrieval},
    }


def _chunking_placeholder_mapping(output_data: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Return notebook placeholders for chunking and graph extraction configuration."""
    chunking = settings.get("chunking", {})
    graph_extraction = settings.get("graph_extraction") or {}
    indexing_parameters = output_data.get("indexing", {}).get("pipeline_spec", {}).get("parameters", {})
    return {
        "CHUNKING_METHOD": chunking.get("method", ""),
        "CHUNK_SIZE": chunking.get("chunk_size", 512),
        "CHUNK_OVERLAP": chunking.get("chunk_overlap", 50),
        "KG_EXTRACTION_CONFIG": graph_extraction
        or indexing_parameters.get("graph_extraction_config")
        or {"mode": "constrained"},
    }


def create_placeholder_mapping(
    output_data: dict[str, Any],
    test_data_key: str = "",
    input_data_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Create a mapping from placeholder names to their values from a pattern definition.

    Extracts values from the ``pattern.json`` structure produced by the
    optimisation pipeline and returns a flat dictionary suitable for
    ``NotebookCell.format_source()``.

    Parameters
    ----------
    output_data : dict[str, Any]
        The parsed ``pattern.json`` data.
    test_data_key : str, default=""
        S3 key of the test data file used as input to AI4RAG.
    input_data_keys : list[str] | None, default=None
        S3 key prefixes of the document locations used as input to AI4RAG.
        Rendered into the notebook as a Python list literal.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping placeholder names to their values.
    """
    settings = output_data.get("settings", {})
    mapping = {
        "AI4RAG_VERSION": __version__,
        "PATTERN_NAME": output_data.get("name", ""),
        **_template_placeholder_mapping(output_data, settings),
        **_generation_placeholder_mapping(settings),
        **_store_placeholder_mapping(settings),
        **_retrieval_placeholder_mapping(settings),
        **_chunking_placeholder_mapping(output_data, settings),
        "TEST_DATA_KEY": test_data_key,
    }
    # Rendered by ``str.format`` into a bare expression, so it has to carry its
    # own quoting and brackets to become a valid Python list literal.
    mapping["INPUT_DATA_KEYS"] = json.dumps(input_data_keys or [])

    return mapping


def generate_notebook_from_template(
    notebook_template: str,
    output_data: dict[str, Any],
    output_notebook_path: str | Path,
    test_data_key: str = "",
    input_data_keys: list[str] | None = None,
) -> None:
    """Generate a filled notebook from a template and pattern configuration.

    Loads the named template, substitutes all placeholders with values
    extracted from *output_data*, and writes the result to disk.

    Parameters
    ----------
    notebook_template : str
        Template base name without the ``_template.ipynb`` suffix
        (e.g. ``"maas_inference"`` or ``"maas_indexing"``).
    output_data : dict[str, Any]
        The parsed ``pattern.json`` data.
    output_notebook_path : str | Path
        Path where the generated notebook is saved.
    test_data_key : str, default=""
        S3 key of the test data file used as input to AI4RAG.
    input_data_keys : list[str] | None, default=None
        S3 key prefixes of the document locations used as input to AI4RAG.
    """
    placeholder_mapping = create_placeholder_mapping(
        output_data,
        test_data_key=test_data_key,
        input_data_keys=input_data_keys,
    )
    notebook = Notebook.load(
        notebook_name=f"{notebook_template}_template.ipynb",
    )
    filled_cells = [cell.format_source(placeholder_mapping) for cell in notebook.cells]
    if notebook_template.startswith("mass_"):
        for cell in filled_cells:
            if isinstance(cell.source, list):
                cell.source = "".join(cell.source)

    notebook = Notebook(cells=filled_cells)
    notebook.save(Path(output_notebook_path))
