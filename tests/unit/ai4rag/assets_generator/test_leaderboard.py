# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from pathlib import Path

import pytest

from ai4rag.components.assets_generator import build_leaderboard_html
from ai4rag.components.assets_generator.leaderboard import _get_nested

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pattern(directory: Path, name: str, score: float) -> None:
    """Write a minimal ``pattern.json`` inside *directory*/*name*/."""
    subdir = directory / name
    subdir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "final_score": score,
        "scores": {"scores": {"faithfulness": {"mean": score}}},
        "settings": {
            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 50},
            "embedding": {"model_id": "ibm/slate-125m-english-rtrvr", "distance_metric": "cosine"},
            "retrieval": {"method": "simple", "number_of_chunks": 5},
            "generation": {"model_id": "ibm/granite-3.1-8b-instruct"},
        },
    }
    with (subdir / "pattern.json").open("w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# _get_nested
# ---------------------------------------------------------------------------


class TestGetNested:
    """Verify dotted-key resolution from flat or nested dicts."""

    def test_flat_key_present(self):
        assert _get_nested({"a.b": 1, "a": {"b": 2}}, "a.b") == 1

    def test_nested_key_resolution(self):
        """Dotted key falls through to nested lookup when flat key is absent."""
        assert _get_nested({"a": {"b": 42}}, "a.b") == 42

    def test_missing_key_returns_none(self):
        assert _get_nested({"x": 1}, "a.b") is None

    def test_empty_dict_returns_none(self):
        assert _get_nested({}, "a.b") is None

    def test_none_dict_returns_none(self):
        assert _get_nested(None, "a.b") is None

    def test_simple_key_without_dot(self):
        assert _get_nested({"key": "value"}, "key") == "value"

    def test_nested_value_not_dict(self):
        """If the outer key exists but its value is not a dict, return None."""
        assert _get_nested({"a": "string"}, "a.b") is None


# ---------------------------------------------------------------------------
# build_leaderboard_html
# ---------------------------------------------------------------------------


class TestBuildLeaderboardHtml:
    """Verify HTML leaderboard generation from pattern directories."""

    @pytest.fixture
    def patterns_dir(self, tmp_path: Path) -> Path:
        """Create a temp directory with two pattern subdirectories."""
        _write_pattern(tmp_path, "pattern_alpha", 0.92)
        _write_pattern(tmp_path, "pattern_beta", 0.88)
        return tmp_path

    def test_html_contains_pattern_names(self, patterns_dir: Path):
        """Both pattern names must appear in the generated HTML."""
        html = build_leaderboard_html(patterns_dir)

        assert "pattern_alpha" in html
        assert "pattern_beta" in html

    def test_html_contains_best_pattern_footer(self, patterns_dir: Path):
        """The best-scoring pattern must be highlighted in the footer."""
        html = build_leaderboard_html(patterns_dir)

        assert "Best pattern:" in html
        assert "pattern_alpha" in html

    def test_html_is_valid_document(self, patterns_dir: Path):
        """Output must be a complete HTML5 document."""
        html = build_leaderboard_html(patterns_dir)

        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_html_contains_metric_scores(self, patterns_dir: Path):
        """Numeric metric values must appear in the table body."""
        html = build_leaderboard_html(patterns_dir)

        assert "0.9200" in html
        assert "0.8800" in html

    def test_ranking_order(self, patterns_dir: Path):
        """The higher-scoring pattern must appear first in the HTML output."""
        html = build_leaderboard_html(patterns_dir)
        pos_alpha = html.index("pattern_alpha")
        pos_beta = html.index("pattern_beta")

        assert pos_alpha < pos_beta, "Higher-scoring pattern must be ranked first"

    def test_custom_optimization_metric(self, tmp_path: Path):
        """Using a different optimization_metric must still produce valid HTML."""
        _write_pattern(tmp_path, "p1", 0.75)

        html = build_leaderboard_html(tmp_path, optimization_metric="answer_correctness")

        assert "answer correctness" in html
        assert "p1" in html

    def test_pattern_count_displayed(self, patterns_dir: Path):
        """The number of patterns must appear in the subtitle."""
        html = build_leaderboard_html(patterns_dir)
        assert "2 pattern(s)" in html


class TestBuildLeaderboardHtmlEmptyDir:
    """Verify behaviour when the patterns directory contains no pattern files."""

    def test_empty_dir_produces_valid_html(self, tmp_path: Path):
        """An empty (but existing) directory must produce valid HTML with zero patterns."""
        html = build_leaderboard_html(tmp_path)

        assert html.strip().startswith("<!DOCTYPE html>")
        assert "0 pattern(s)" in html

    def test_dir_with_non_pattern_files(self, tmp_path: Path):
        """Files that are not in subdirectories with pattern.json are ignored."""
        (tmp_path / "README.md").write_text("hello")
        (tmp_path / "some_subdir").mkdir()
        # No pattern.json inside some_subdir

        html = build_leaderboard_html(tmp_path)
        assert "0 pattern(s)" in html


class TestBuildLeaderboardHtmlErrors:
    """Verify error handling for invalid inputs."""

    def test_nonexistent_dir_raises_file_not_found(self, tmp_path: Path):
        """A path that does not exist must raise FileNotFoundError."""
        fake = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="rag_patterns path is not a directory"):
            build_leaderboard_html(fake)

    def test_file_instead_of_dir_raises_file_not_found(self, tmp_path: Path):
        """Passing a file path (not a directory) must raise FileNotFoundError."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        with pytest.raises(FileNotFoundError):
            build_leaderboard_html(file_path)
