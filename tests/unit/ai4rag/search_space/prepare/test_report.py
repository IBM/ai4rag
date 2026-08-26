# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for ai4rag.search_space.prepare.report.

Covers the *write* half of the search-space-report contract:

* :meth:`SearchSpaceReport.save_json` — JSON persistence;
* :func:`build_search_space_report` — assembling the report from a prepared
  search space, including serializing the model dimensions and dropping values
  that only appeared in rule-rejected combinations.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from ai4rag.search_space.prepare.models import get_embedding_models, get_foundation_models
from ai4rag.search_space.prepare.report import SearchSpaceReport, build_search_space_report
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

_FM_ID = "publishers/ns/models/qwen3-8b"
_EM_ID = "publishers/ns/models/bge-m3"


def _client() -> MagicMock:
    """A restore-only serving client: passes the OpenAI gate, never hits network."""
    client = MagicMock(spec=OpenAI)
    client.base_url = "https://maas.example.com/v1"
    client.api_key = "secret-key"
    return client


def _foundation(model_id: str = _FM_ID):
    """Build a real foundation model via the restore path (no network calls)."""
    spec = {"model_id": model_id, "type": "generation", "params": {"temperature": 0.3}}
    return get_foundation_models(_client(), [spec], validate=False)[0]


def _embedding(model_id: str = _EM_ID, context_length: int = 8192):
    """Build a real embedding model via the restore path (no network calls)."""
    spec = {
        "model_id": model_id,
        "type": "embedding",
        "params": {"embedding_dimension": 768, "context_length": context_length},
    }
    return get_embedding_models(_client(), [spec], validate=False)[0]


# ---------------------------------------------------------------------------
# SearchSpaceReport.save_json
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_report() -> SearchSpaceReport:
    """A minimal report for persistence tests."""
    return SearchSpaceReport(
        search_space={
            "foundation_model": [{"model_id": "model-a"}],
            "embedding_model": [{"model_id": "emb-a"}],
            "chunk_size": [256, 512],
        }
    )


class TestSearchSpaceReportSaveJson:
    """JSON serialization of SearchSpaceReport."""

    def test_save_json_creates_file(self, simple_report, tmp_path: Path):
        """save_json creates a readable JSON file at the given path."""
        out_file = tmp_path / "report.json"
        simple_report.save_json(out_file)

        assert out_file.exists()
        assert isinstance(json.loads(out_file.read_text()), dict)

    def test_save_json_creates_parent_directories(self, simple_report, tmp_path: Path):
        """save_json creates intermediate directories if they do not exist."""
        out_file = tmp_path / "nested" / "dir" / "report.json"
        simple_report.save_json(out_file)

        assert out_file.exists()

    def test_save_json_preserves_search_space_keys(self, simple_report, tmp_path: Path):
        """All top-level search_space keys appear in the serialized JSON."""
        out_file = tmp_path / "report.json"
        simple_report.save_json(out_file)

        data = json.loads(out_file.read_text())
        assert set(data) >= {"foundation_model", "embedding_model", "chunk_size"}


# ---------------------------------------------------------------------------
# build_search_space_report
# ---------------------------------------------------------------------------


class TestBuildSearchSpaceReport:
    """Assemble a report from a prepared AI4RAGSearchSpace."""

    def _search_space(self, chunking_methods, chunk_sizes) -> AI4RAGSearchSpace:
        return AI4RAGSearchSpace(
            params=[
                Parameter(name=AI4RAGParamNames.FOUNDATION_MODEL, values=(_foundation(),)),
                Parameter(name=AI4RAGParamNames.EMBEDDING_MODEL, values=(_embedding(),)),
                Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, values=tuple(chunking_methods)),
                Parameter(name=AI4RAGParamNames.CHUNK_SIZE, values=tuple(chunk_sizes)),
            ]
        )

    def test_model_dimensions_are_serialized_specs(self):
        """foundation_model / embedding_model become lists of serialized model specs."""
        report = build_search_space_report(self._search_space(["hybrid"], [512]))

        fm = report.search_space["foundation_model"]
        em = report.search_space["embedding_model"]
        assert [m["model_id"] for m in fm] == [_FM_ID]
        assert fm[0]["type"] == "generation"
        assert [m["model_id"] for m in em] == [_EM_ID]
        assert em[0]["type"] == "embedding"

    def test_report_is_json_serializable(self, tmp_path: Path):
        """The assembled report round-trips through save_json without error."""
        report = build_search_space_report(self._search_space(["hybrid"], [512]))
        out_file = tmp_path / "report.json"
        report.save_json(out_file)

        data = json.loads(out_file.read_text())
        assert data["foundation_model"][0]["model_id"] == _FM_ID

    def test_rules_drop_values_from_only_invalid_combinations(self):
        """recursive + chunk_sizes [128, 256] leave no valid combination, so
        every non-model dimension collapses to an empty list.

        - overlap=0   filtered by _rule_chunk_overlap_for_chunking_method (recursive needs > 0)
        - overlap=128 filtered by _rule_chunk_size_bigger_than_chunk_overlap (256 > 2*128 is False)
        - overlap=256 filtered by _rule_chunk_size_bigger_than_chunk_overlap (256 > 2*256 is False)
        """
        report = build_search_space_report(self._search_space(["recursive"], [128, 256]))

        assert report.search_space["chunk_size"] == []
        assert report.search_space["chunk_overlap"] == []
        assert report.search_space["chunking_method"] == []
        # Model dimensions are taken from the search space directly, not from
        # combinations, so they remain populated even when no combination is valid.
        assert [m["model_id"] for m in report.search_space["foundation_model"]] == [_FM_ID]
