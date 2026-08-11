# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from pathlib import Path

import pytest

from ai4rag.components.assets_generator import create_placeholder_mapping, generate_notebook_from_template

# ---------------------------------------------------------------------------
# create_placeholder_mapping
# ---------------------------------------------------------------------------

_SAMPLE_PATTERN_DATA: dict = {
    "name": "pattern_001",
    "settings": {
        "generation": {
            "model_id": "ibm/granite-3.1-8b-instruct",
            "base_url": "https://maas.example.com/ns/granite/v1",
            "system_message_text": "Answer the question.",
            "user_message_text": "Context: {reference_documents}\nQuestion: {question}",
            "context_template_text": "{document}",
        },
        "embedding": {
            "model_id": "ibm/slate-125m-english-rtrvr",
            "base_url": "https://maas.example.com/ns/slate/v1",
            "embedding_params": {"embedding_dimension": 768},
        },
        "vector_store_binding": {
            "provider_type": "milvus",
            "collection_name": "test_collection",
        },
        "retrieval": {
            "method": "simple",
            "number_of_chunks": 5,
            "search_mode": None,
            "ranker_strategy": None,
            "ranker_k": None,
            "ranker_alpha": None,
        },
        "chunking": {
            "method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
    },
}


class TestCreatePlaceholderMapping:
    """Verify ``create_placeholder_mapping`` extracts all expected keys."""

    @pytest.fixture
    def mapping(self) -> dict:
        """Mapping built from the realistic sample pattern data."""
        return create_placeholder_mapping(
            _SAMPLE_PATTERN_DATA,
            test_data_key="s3://bucket/test.jsonl",
            input_data_key="s3://bucket/docs/",
        )

    def test_pattern_name(self, mapping: dict):
        assert mapping["PATTERN_NAME"] == "pattern_001"

    def test_generation_fields(self, mapping: dict):
        assert mapping["FM_MODEL_ID"] == "ibm/granite-3.1-8b-instruct"
        assert mapping["FM_BASE_URL"] == "https://maas.example.com/ns/granite/v1"
        assert mapping["SYSTEM_MESSAGE"] == "Answer the question."
        assert mapping["CONTEXT_TEXT"] == "{document}"

    def test_embedding_fields(self, mapping: dict):
        assert mapping["EMBEDDING_MODEL_ID"] == "ibm/slate-125m-english-rtrvr"
        assert mapping["EMBEDDING_BASE_URL"] == "https://maas.example.com/ns/slate/v1"
        assert mapping["EMBEDDING_PARAMS"] == {"embedding_dimension": 768}

    def test_vector_store_fields(self, mapping: dict):
        assert mapping["PROVIDER_TYPE"] == "milvus"
        assert mapping["COLLECTION_NAME"] == "test_collection"

    def test_required_env_vars_rendered(self, mapping: dict):
        """The provider's env vars must render as a Markdown bullet list."""
        rendered = mapping["REQUIRED_ENV_VARS"]
        assert "- `MILVUS_URI`" in rendered
        assert "- `MILVUS_TOKEN`" in rendered
        assert "- `MILVUS_SERVER_CERT`" in rendered

    def test_required_env_vars_empty_for_unknown_provider(self):
        """An unknown or missing provider must yield an empty env-var block."""
        mapping = create_placeholder_mapping({})
        assert mapping["PROVIDER_TYPE"] == ""
        assert mapping["REQUIRED_ENV_VARS"] == ""

    def test_retrieval_fields(self, mapping: dict):
        assert mapping["RETRIEVAL_METHOD"] == "simple"
        assert mapping["NUMBER_OF_CHUNKS"] == 5

    def test_chunking_fields(self, mapping: dict):
        assert mapping["CHUNKING_METHOD"] == "recursive"
        assert mapping["CHUNK_SIZE"] == 512
        assert mapping["CHUNK_OVERLAP"] == 50

    def test_s3_keys(self, mapping: dict):
        assert mapping["TEST_DATA_KEY"] == "s3://bucket/test.jsonl"
        assert mapping["INPUT_DATA_KEY"] == "s3://bucket/docs/"

    def test_all_expected_keys_present(self, mapping: dict):
        """All documented placeholder names must appear in the mapping."""
        expected_keys = {
            "PATTERN_NAME",
            "FM_MODEL_ID",
            "SYSTEM_MESSAGE",
            "USER_MESSAGE",
            "CONTEXT_TEXT",
            "FM_BASE_URL",
            "EMBEDDING_MODEL_ID",
            "EMBEDDING_BASE_URL",
            "EMBEDDING_PARAMS",
            "PROVIDER_TYPE",
            "COLLECTION_NAME",
            "REQUIRED_ENV_VARS",
            "RETRIEVAL_METHOD",
            "NUMBER_OF_CHUNKS",
            "SEARCH_MODE",
            "RANKER_STRATEGY",
            "RANKER_K",
            "RANKER_ALPHA",
            "CHUNKING_METHOD",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "TEST_DATA_KEY",
            "INPUT_DATA_KEY",
        }
        assert expected_keys.issubset(set(mapping.keys()))

    def test_empty_output_data_uses_defaults(self):
        """An empty pattern dict must still produce a mapping with safe defaults."""
        mapping = create_placeholder_mapping({})

        assert mapping["PATTERN_NAME"] == ""
        assert mapping["FM_MODEL_ID"] == ""
        assert mapping["CHUNK_SIZE"] == 512
        assert mapping["CHUNK_OVERLAP"] == 50
        assert mapping["NUMBER_OF_CHUNKS"] == 5

    def test_base_urls_empty_when_not_provided(self):
        """When the pattern carries no model base URLs, the placeholders must be empty strings."""
        mapping = create_placeholder_mapping({})
        assert mapping["FM_BASE_URL"] == ""
        assert mapping["EMBEDDING_BASE_URL"] == ""


# ---------------------------------------------------------------------------
# generate_notebook_from_template
# ---------------------------------------------------------------------------


class TestGenerateNotebookFromTemplate:
    """Verify ``generate_notebook_from_template`` orchestrates load, fill, and save."""

    def test_calls_load_and_save(self, mocker, tmp_path: Path):
        """The function must load the template, format cells, and save to disk."""
        mock_cell = mocker.MagicMock()
        mock_cell.format_source.return_value = mock_cell

        mock_notebook = mocker.MagicMock()
        mock_notebook.cells = [mock_cell]

        mock_load = mocker.patch(
            "ai4rag.components.assets_generator.templates.Notebook.load",
            return_value=mock_notebook,
        )
        mock_save = mocker.patch("ai4rag.components.assets_generator.templates.Notebook.save")

        output_path = tmp_path / "output.ipynb"
        generate_notebook_from_template(
            notebook_template="maas_indexing",
            output_data=_SAMPLE_PATTERN_DATA,
            output_notebook_path=output_path,
        )

        mock_load.assert_called_once_with(
            notebook_name="maas_indexing_template.ipynb",
        )
        mock_cell.format_source.assert_called_once()
        mock_save.assert_called_once_with(output_path)

    def test_passes_s3_keys(self, mocker, tmp_path: Path):
        """S3 keys must propagate into the placeholder mapping."""
        mock_cell = mocker.MagicMock()
        mock_cell.format_source.return_value = mock_cell
        mock_notebook = mocker.MagicMock()
        mock_notebook.cells = [mock_cell]

        mocker.patch("ai4rag.components.assets_generator.templates.Notebook.load", return_value=mock_notebook)
        mocker.patch("ai4rag.components.assets_generator.templates.Notebook.save")
        mock_create = mocker.patch(
            "ai4rag.components.assets_generator.templates.create_placeholder_mapping",
            return_value={},
        )

        generate_notebook_from_template(
            notebook_template="maas_inference",
            output_data={},
            output_notebook_path=tmp_path / "out.ipynb",
            test_data_key="key/test",
            input_data_key="key/input",
        )

        mock_create.assert_called_once_with(
            {},
            test_data_key="key/test",
            input_data_key="key/input",
        )


def _read_notebook_text(path: Path) -> str:
    """Return the concatenated source of every cell in a generated notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"] for cell in notebook["cells"]
    )


@pytest.mark.parametrize("template", ["maas_indexing", "maas_inference"])
class TestGeneratedNotebookUsesDirectClients:
    """End-to-end checks that generated notebooks target the direct-client vector store API."""

    def test_generated_notebook_uses_new_api(self, template: str, tmp_path: Path):
        """Both templates must render the ``get_vector_store_config`` factory and drop the OGX store."""
        output_path = tmp_path / f"{template}.ipynb"
        generate_notebook_from_template(
            notebook_template=template,
            output_data=_SAMPLE_PATTERN_DATA,
            output_notebook_path=output_path,
        )
        text = _read_notebook_text(output_path)

        # New direct-client API is present, with the provider substituted from the pattern.
        assert "get_vector_store_config" in text
        assert "get_vector_store(" in text
        assert 'provider_type = "milvus"' in text
        assert 'collection_name = "test_collection"' in text

        # Old OGX vector-store API is fully removed.
        assert "OGXVectorStore" not in text
        assert "provider_id" not in text
        assert "reuse_collection_name" not in text

    def test_generated_notebook_lists_required_env_vars(self, template: str, tmp_path: Path):
        """The provider's required environment variables must be documented in the notebook."""
        output_path = tmp_path / f"{template}.ipynb"
        generate_notebook_from_template(
            notebook_template=template,
            output_data=_SAMPLE_PATTERN_DATA,
            output_notebook_path=output_path,
        )
        text = _read_notebook_text(output_path)

        assert "MILVUS_URI" in text

    def test_generated_notebook_has_no_unresolved_placeholders(self, template: str, tmp_path: Path):
        """No ``{PLACEHOLDER}`` tokens may survive substitution in the rendered notebook."""
        import re

        output_path = tmp_path / f"{template}.ipynb"
        generate_notebook_from_template(
            notebook_template=template,
            output_data=_SAMPLE_PATTERN_DATA,
            output_notebook_path=output_path,
        )
        text = _read_notebook_text(output_path)

        # Escaped literal braces ({{ }}) are intentional; only single-brace ALL-CAPS tokens are placeholders.
        leftover = sorted(set(re.findall(r"(?<!\{)\{[A-Z_]+\}(?!\})", text)))
        assert leftover == [], f"Unresolved placeholders: {leftover}"
