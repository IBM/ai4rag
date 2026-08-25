# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from unittest.mock import MagicMock

import pytest

from ai4rag.components.optimization.rag_templates_optimization import (
    DEFAULT_LLM_JUDGE_MODE,
    DEFAULT_MAX_RAG_PATTERNS,
    MIN_MAX_RAG_PATTERNS_RANGE,
    SUPPORTED_OPTIMIZATION_METRICS,
    _generate_output_artifacts,
    _validate_optimization_settings,
    run_rag_optimization,
)
from ai4rag.rag.vector_store.config import ChromaConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_maas_client() -> MagicMock:
    """Return a bare MagicMock standing in for a MaaS OpenAI client."""
    return MagicMock()


# ---------------------------------------------------------------------------
# _validate_optimization_settings
# ---------------------------------------------------------------------------


class TestValidateOptimizationSettings:
    """Tests for the _validate_optimization_settings helper."""

    def test_none_returns_empty_dict(self):
        """None input (no settings provided) must return an empty dict."""
        assert _validate_optimization_settings(None) == {}

    def test_non_dict_raises_type_error(self):
        """A non-dict value must raise TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            _validate_optimization_settings("not-a-dict")  # type: ignore[arg-type]

    def test_list_raises_type_error(self):
        """A list (common mis-use) must raise TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            _validate_optimization_settings([1, 2, 3])  # type: ignore[arg-type]

    def test_valid_settings_with_int_max_patterns(self):
        """A valid dict with an integer max_number_of_rag_patterns must pass."""
        settings = {"max_number_of_rag_patterns": 10}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_string_max_patterns_parsed_to_int(self):
        """The pipeline UI sends strings; they must be parsed without error."""
        settings = {"max_number_of_rag_patterns": "12"}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_string_with_whitespace_parsed(self):
        """Whitespace around the string value must be tolerated."""
        settings = {"max_number_of_rag_patterns": "  8  "}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_invalid_string_raises_value_error(self):
        """A non-numeric string must raise ValueError."""
        with pytest.raises(ValueError, match="valid integer"):
            _validate_optimization_settings({"max_number_of_rag_patterns": "abc"})

    def test_below_range_raises_value_error(self):
        """A value below the minimum allowed range must raise ValueError."""
        below_min = MIN_MAX_RAG_PATTERNS_RANGE[0] - 1
        with pytest.raises(ValueError, match="must be in range"):
            _validate_optimization_settings({"max_number_of_rag_patterns": below_min})

    def test_above_range_raises_value_error(self):
        """A value above the maximum allowed range must raise ValueError."""
        above_max = MIN_MAX_RAG_PATTERNS_RANGE[1] + 1
        with pytest.raises(ValueError, match="must be in range"):
            _validate_optimization_settings({"max_number_of_rag_patterns": above_max})

    def test_boundary_min_accepted(self):
        """The exact minimum boundary must be accepted."""
        settings = {"max_number_of_rag_patterns": MIN_MAX_RAG_PATTERNS_RANGE[0]}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_boundary_max_accepted(self):
        """The exact maximum boundary must be accepted."""
        settings = {"max_number_of_rag_patterns": MIN_MAX_RAG_PATTERNS_RANGE[1]}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_default_max_patterns_within_range(self):
        """The DEFAULT_MAX_RAG_PATTERNS constant must fall inside the allowed range."""
        lo, hi = MIN_MAX_RAG_PATTERNS_RANGE
        assert lo <= DEFAULT_MAX_RAG_PATTERNS <= hi

    def test_empty_dict_passes(self):
        """An empty dict (no overrides) must be accepted and returned."""
        result = _validate_optimization_settings({})
        assert result == {}

    def test_float_raises_type_error(self):
        """A float value must raise TypeError (after range check, it is not int)."""
        with pytest.raises(TypeError, match="must be an integer"):
            _validate_optimization_settings({"max_number_of_rag_patterns": 8.5})

    def test_extra_keys_preserved(self):
        """Settings with additional keys besides max_number_of_rag_patterns must pass through."""
        settings = {"max_number_of_rag_patterns": 10, "metric": "faithfulness"}
        result = _validate_optimization_settings(settings)
        assert result["metric"] == "faithfulness"


# ---------------------------------------------------------------------------
# run_rag_optimization -- input validation only
# ---------------------------------------------------------------------------


class TestRunRagOptimizationValidation:
    """Test input validation in run_rag_optimization.

    These tests verify that the function rejects bad inputs before
    reaching any heavy I/O or MaaS calls.
    """

    def test_test_data_key_not_json_raises(self, mock_maas_client):
        """A test_data_key not ending in .json must raise ValueError."""
        with pytest.raises(ValueError, match="JSON file"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="data.csv",
            )

    def test_empty_test_data_key_raises(self, mock_maas_client):
        """An empty test_data_key must raise ValueError."""
        with pytest.raises(ValueError, match="JSON file"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="",
            )

    def test_invalid_optimization_metric_raises(self, mock_maas_client):
        """An unsupported metric in optimization_settings must raise ValueError."""
        with pytest.raises(ValueError, match="is not supported"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="bench.json",
                optimization_settings={"metric": "nonexistent_metric"},
            )

    def test_supported_optimization_metrics_constant(self):
        """SUPPORTED_OPTIMIZATION_METRICS must contain unitxt and custom metrics."""
        assert "faithfulness" in SUPPORTED_OPTIMIZATION_METRICS
        assert "answer_correctness" in SUPPORTED_OPTIMIZATION_METRICS
        assert "context_correctness" in SUPPORTED_OPTIMIZATION_METRICS
        assert "overall_score" in SUPPORTED_OPTIMIZATION_METRICS
        assert "answer_relevance" not in SUPPORTED_OPTIMIZATION_METRICS

    def test_invalid_optimization_settings_type_raises(self, mock_maas_client):
        """Non-dict optimization_settings must raise TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="bench.json",
                optimization_settings="bad",  # type: ignore[arg-type]
            )

    def test_out_of_range_max_patterns_raises(self, mock_maas_client):
        """max_number_of_rag_patterns outside the allowed range must raise ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="bench.json",
                optimization_settings={"max_number_of_rag_patterns": 50},
            )


# ---------------------------------------------------------------------------
# run_rag_optimization -- inference_max_threads parameter
# ---------------------------------------------------------------------------


class TestRunRagOptimizationInferenceMaxThreads:
    """Tests for the inference_max_threads parameter on run_rag_optimization."""

    def test_inference_max_threads_has_default_of_ten(self):
        """The inference_max_threads parameter must have a default value of 10."""
        import inspect

        sig = inspect.signature(run_rag_optimization)
        param = sig.parameters["inference_max_threads"]
        assert param.default == 10

    def test_inference_max_threads_is_accepted(self, mock_maas_client):
        """Passing inference_max_threads alongside an invalid input must still raise
        the expected validation error (not a TypeError from an unknown param)."""
        with pytest.raises(ValueError, match="JSON file"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="bench.csv",
                inference_max_threads=4,
            )


# ---------------------------------------------------------------------------
# run_rag_optimization -- hybrid evaluation setup
# ---------------------------------------------------------------------------


class TestRunRagOptimizationEvaluation:
    """Tests for hybrid evaluation wiring on run_rag_optimization."""

    def test_optimization_metric_defaults_to_overall_score(self):
        import inspect

        sig = inspect.signature(run_rag_optimization)
        assert "evaluator" not in sig.parameters

        from ai4rag.components.optimization import rag_templates_optimization as module

        assert module.DEFAULT_METRIC == "overall_score"


# ---------------------------------------------------------------------------
# _generate_output_artifacts -- indexing pipeline_spec enrichment
# ---------------------------------------------------------------------------


def _make_pattern(name: str = "pattern_001") -> dict:
    """Return a minimal but complete raw pattern for artefact generation."""
    return {
        "payload": {
            "name": name,
            "settings": {
                "vector_store_binding": {
                    "provider_type": "milvus",
                    "collection_name": "pattern_001_collection",
                },
                "embedding": {
                    "model_id": "ibm/slate-125m-english-rtrvr",
                    "embedding_params": {"embedding_dimension": 768},
                },
                "chunking": {
                    "method": "recursive",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                },
            },
        },
        "evaluation_results": [{"question": "q?", "answer": "a"}],
    }


class TestGenerateOutputArtifactsIndexingSpec:
    """Cover the ``indexing_pipeline_params`` enrichment path (direct-client schema)."""

    @pytest.fixture(autouse=True)
    def _stub_notebook_generation(self, mocker):
        """Isolate pipeline-spec logic from real notebook rendering."""
        return mocker.patch("ai4rag.components.optimization.rag_templates_optimization.generate_notebook_from_template")

    def test_pipeline_spec_uses_provider_type_and_collection_name(self, tmp_path):
        """The indexing spec must source the vector store from the new binding schema."""
        patterns = _generate_output_artifacts(
            patterns_raw=[_make_pattern()],
            output_dir=tmp_path,
            input_data_key="s3://bucket/docs/",
            test_data_key="s3://bucket/test.json",
            indexing_pipeline_params={
                "maas_secret_name": "maas-secret",
                "input_data_secret_name": "s3-secret",
                "input_data_bucket_name": "docs-bucket",
                "input_data_key": "docs/",
                "batch_size": 64,
            },
        )

        params = patterns[0]["indexing"]["pipeline_spec"]["parameters"]
        assert params["provider_type"] == "milvus"
        assert params["collection_name"] == "pattern_001_collection"
        assert params["embedding_model_id"] == "ibm/slate-125m-english-rtrvr"
        assert params["chunk_size"] == 512

        # Old vector-store parameters must be gone.
        assert "vector_store_id" not in params
        assert "vector_io_provider_id" not in params

    def test_overrides_allow_collection_name_not_vector_store_id(self, tmp_path):
        """``overrides_allowed`` must expose the collection name, not the stale store id."""
        patterns = _generate_output_artifacts(
            patterns_raw=[_make_pattern()],
            output_dir=tmp_path,
            input_data_key="",
            test_data_key="",
            indexing_pipeline_params={"batch_size": 32},
        )

        overrides = patterns[0]["indexing"]["pipeline_spec"]["overrides_allowed"]
        assert "collection_name" in overrides
        assert "vector_store_id" not in overrides

    def test_no_indexing_spec_when_params_absent(self, tmp_path):
        """Without ``indexing_pipeline_params`` no indexing spec must be added."""
        patterns = _generate_output_artifacts(
            patterns_raw=[_make_pattern()],
            output_dir=tmp_path,
            input_data_key="",
            test_data_key="",
            indexing_pipeline_params=None,
        )

        assert "indexing" not in patterns[0]

    def test_pattern_json_written_to_disk(self, tmp_path):
        """The enriched pattern must be persisted as ``pattern.json``."""
        _generate_output_artifacts(
            patterns_raw=[_make_pattern()],
            output_dir=tmp_path,
            input_data_key="",
            test_data_key="",
            indexing_pipeline_params={"batch_size": 16},
        )

        pattern_json = tmp_path / "pattern_001" / "pattern.json"
        assert pattern_json.exists()
        persisted = json.loads(pattern_json.read_text(encoding="utf-8"))
        assert persisted["indexing"]["pipeline_spec"]["parameters"]["provider_type"] == "milvus"


# ---------------------------------------------------------------------------
# run_rag_optimization -- llm_judge_mode selection
# ---------------------------------------------------------------------------


class TestRunRagOptimizationLLMJudgeMode:
    """Tests for the llm_judge_mode parameter on run_rag_optimization."""

    def test_default_mode_is_base(self):
        """llm_judge_mode must default to 'base' (in-house LLM judge)."""
        import inspect

        sig = inspect.signature(run_rag_optimization)
        assert sig.parameters["llm_judge_mode"].default == DEFAULT_LLM_JUDGE_MODE
        assert DEFAULT_LLM_JUDGE_MODE == "base"

    def test_judge_enabled_removed(self):
        """The old binary judge_enabled parameter must no longer exist."""
        import inspect

        sig = inspect.signature(run_rag_optimization)
        assert "judge_enabled" not in sig.parameters

    @pytest.mark.parametrize("mode", ["base", "ragas", "all", "none"])
    def test_valid_modes_pass_validation(self, mock_maas_client, mode):
        """Valid modes must pass the mode check and fail later on real inputs."""
        # A non-JSON test_data_key proves we got past the llm_judge_mode check.
        with pytest.raises(ValueError, match="JSON file"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="bench.csv",
                llm_judge_mode=mode,
            )

    def test_invalid_mode_raises(self, mock_maas_client):
        """An unsupported llm_judge_mode must raise ValueError before any I/O."""
        with pytest.raises(ValueError, match="llm_judge_mode"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="bench.json",
                llm_judge_mode="judge",  # type: ignore[arg-type]
            )


class TestRunRagOptimizationWarmStartParams:
    """Tests that n_random_nodes and warm_start_strategy are forwarded to GAMOptSettings."""

    def test_default_n_random_nodes(self):
        """n_random_nodes must default to None (auto-computed)."""
        import inspect

        sig = inspect.signature(run_rag_optimization)
        assert sig.parameters["n_random_nodes"].default is None

    def test_default_warm_start_strategy(self):
        """warm_start_strategy must default to 'random'."""
        import inspect

        sig = inspect.signature(run_rag_optimization)
        assert sig.parameters["warm_start_strategy"].default == "random"

    def test_invalid_warm_start_strategy_raises(self, mock_maas_client):
        """An invalid warm_start_strategy must raise ValueError from GAMOptSettings."""
        with pytest.raises(ValueError, match="warm_start_strategy"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.json",
                output_dir="out",
                maas_client=mock_maas_client,
                vector_store_config=ChromaConfig(),
                test_data_key="bench.json",
                warm_start_strategy="invalid",  # type: ignore[arg-type]
            )

    def test_valid_warm_start_strategies_pass_validation(self, mock_maas_client):
        """Valid strategy values pass the warm_start_strategy check and fail later on real inputs."""
        for strategy in ("random", "greedy"):
            with pytest.raises(ValueError, match="JSON file"):
                run_rag_optimization(
                    extracted_text_path="dummy",
                    test_data_path="dummy.json",
                    search_space_report_path="dummy.json",
                    output_dir="out",
                    maas_client=mock_maas_client,
                    vector_store_config=ChromaConfig(),
                    test_data_key="bench.csv",
                    warm_start_strategy=strategy,
                )
