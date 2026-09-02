# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from pathlib import Path
from typing import Any

from ai4rag import __version__
from ai4rag.rag.vector_store import get_vector_store_env_vars
from ai4rag.utils.assets_generator.notebook import Notebook


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


def create_placeholder_mapping(
    output_data: dict[str, Any],
    test_data_key: str = "",
    input_data_key: str = "",
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
    input_data_key : str, default=""
        S3 key of the documents directory used as input to AI4RAG.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping placeholder names to their values.
    """
    mapping: dict[str, Any] = {}

    mapping["AI4RAG_VERSION"] = __version__
    mapping["PATTERN_NAME"] = output_data.get("name", "")
    settings = output_data.get("settings", {})
    fm = settings.get("generation", {})
    mapping["FM_MODEL_ID"] = fm.get("model_id", "")
    mapping["SYSTEM_MESSAGE"] = fm.get("system_message_text", "")
    mapping["USER_MESSAGE"] = fm.get("user_message_text", "")
    mapping["CONTEXT_TEXT"] = fm.get("context_template_text", "")
    # Detected generation language ({"code", "name"}); defaults mirror
    # BaseFoundationModel's "auto" so the notebook restores the same behaviour.
    mapping["LANGUAGE"] = fm.get("language", {"code": "", "name": "auto"})

    em = settings.get("embedding", {})
    mapping["EMBEDDING_MODEL_ID"] = em.get("model_id", "")
    mapping["EMBEDDING_PARAMS"] = em.get("embedding_params", {"embedding_dimension": 768})
    vs = settings.get("vector_store_binding", {})
    provider_type = vs.get("provider_type", "")
    mapping["PROVIDER_TYPE"] = provider_type
    mapping["COLLECTION_NAME"] = vs.get("collection_name", "")
    mapping["REQUIRED_ENV_VARS"] = _format_required_env_vars(provider_type)

    ret = settings.get("retrieval", {})
    mapping["RETRIEVAL_METHOD"] = ret.get("method", "")
    mapping["NUMBER_OF_CHUNKS"] = ret.get("number_of_chunks", 5)
    mapping["SEARCH_MODE"] = ret.get("search_mode")
    mapping["RANKER_STRATEGY"] = ret.get("ranker_strategy")
    mapping["RANKER_K"] = ret.get("ranker_k")
    mapping["RANKER_ALPHA"] = ret.get("ranker_alpha")

    ch = settings.get("chunking", {})
    mapping["CHUNKING_METHOD"] = ch.get("method", "")
    mapping["CHUNK_SIZE"] = ch.get("chunk_size", 512)
    mapping["CHUNK_OVERLAP"] = ch.get("chunk_overlap", 50)

    mapping["TEST_DATA_KEY"] = test_data_key
    mapping["INPUT_DATA_KEY"] = input_data_key

    return mapping


def generate_notebook_from_template(
    notebook_template: str,
    output_data: dict[str, Any],
    output_notebook_path: str | Path,
    test_data_key: str = "",
    input_data_key: str = "",
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
    input_data_key : str, default=""
        S3 key of the documents directory used as input to AI4RAG.
    """
    placeholder_mapping = create_placeholder_mapping(
        output_data,
        test_data_key=test_data_key,
        input_data_key=input_data_key,
    )
    notebook = Notebook.load(
        notebook_name=f"{notebook_template}_template.ipynb",
    )
    filled_cells = [cell.format_source(placeholder_mapping) for cell in notebook.cells]

    notebook = Notebook(cells=filled_cells)
    notebook.save(Path(output_notebook_path))
