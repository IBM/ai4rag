# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

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
            "system_message_text": "Answer the question.",
            "user_message_text": "Context: {reference_documents}\nQuestion: {question}",
            "context_template_text": "{document}",
        },
        "embedding": {
            "model_id": "ibm/slate-125m-english-rtrvr",
            "embedding_params": {"embedding_dimension": 768},
        },
        "vector_store_binding": {
            "provider_id": "milvus",
            "vector_store_id": "test_collection",
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
            ogx_base_url="  https://ogx.example.com  ",
        )

    def test_pattern_name(self, mapping: dict):
        assert mapping["PATTERN_NAME"] == "pattern_001"

    def test_generation_fields(self, mapping: dict):
        assert mapping["FM_MODEL_ID"] == "ibm/granite-3.1-8b-instruct"
        assert mapping["SYSTEM_MESSAGE"] == "Answer the question."
        assert mapping["CONTEXT_TEXT"] == "{document}"

    def test_embedding_fields(self, mapping: dict):
        assert mapping["EMBEDDING_MODEL_ID"] == "ibm/slate-125m-english-rtrvr"
        assert mapping["EMBEDDING_PARAMS"] == {"embedding_dimension": 768}

    def test_vector_store_fields(self, mapping: dict):
        assert mapping["PROVIDER_ID"] == "milvus"
        assert mapping["COLLECTION_NAME"] == "test_collection"

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

    def test_ogx_url_stripped(self, mapping: dict):
        """Leading/trailing whitespace must be stripped from ogx_base_url."""
        assert mapping["OGX_CLIENT_BASE_URL"] == "https://ogx.example.com"

    def test_all_expected_keys_present(self, mapping: dict):
        """All documented placeholder names must appear in the mapping."""
        expected_keys = {
            "PATTERN_NAME",
            "FM_MODEL_ID",
            "SYSTEM_MESSAGE",
            "USER_MESSAGE",
            "CONTEXT_TEXT",
            "EMBEDDING_MODEL_ID",
            "EMBEDDING_PARAMS",
            "PROVIDER_ID",
            "COLLECTION_NAME",
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
            "OGX_CLIENT_BASE_URL",
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

    def test_ogx_url_empty_when_not_provided(self):
        """When ogx_base_url is not given, the placeholder must be an empty string."""
        mapping = create_placeholder_mapping({})
        assert mapping["OGX_CLIENT_BASE_URL"] == ""


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
            notebook_template="ogx_indexing",
            output_data=_SAMPLE_PATTERN_DATA,
            output_notebook_path=output_path,
        )

        mock_load.assert_called_once_with(
            notebook_name="ogx_indexing_template.ipynb",
        )
        mock_cell.format_source.assert_called_once()
        mock_save.assert_called_once_with(output_path)

    def test_passes_s3_keys_and_url(self, mocker, tmp_path: Path):
        """S3 keys and OGX URL must propagate into the placeholder mapping."""
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
            notebook_template="ogx_inference",
            output_data={},
            output_notebook_path=tmp_path / "out.ipynb",
            test_data_key="key/test",
            input_data_key="key/input",
            ogx_base_url="https://ogx.local",
        )

        mock_create.assert_called_once_with(
            {},
            test_data_key="key/test",
            input_data_key="key/input",
            ogx_base_url="https://ogx.local",
        )
