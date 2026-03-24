# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.search_space.src.default_search_space import get_default_ai4rag_search_space_parameters
from ai4rag.search_space.src.search_space import (
    AI4RAGSearchSpace,
    Parameter,
    SearchSpace,
    SearchSpaceValueError,
    _rule_adjust_window_to_retrieval_method,
    _rule_chunk_params_consistency_with_method,
    _rule_chunk_size_bigger_than_chunk_overlap,
    _rule_chunk_size_within_embedding_context_length,
    _rule_ranker_alpha_for_weighted_only,
    _rule_ranker_k_for_rrf_only,
    _rule_search_mode_ranker_consistency,
)


@pytest.fixture
def mocked_params() -> list[Parameter]:
    return [
        Parameter(name="a", param_type="I", v_min=1, v_max=5),
        Parameter(name="b", param_type="C", values=[6, 7, 8, 9, 10]),
    ]


@pytest.mark.parametrize(
    "combination, expected_value",
    (
        ({"chunk_size": 2048, "chunk_overlap": 512}, True),
        ({"chunk_size": 512, "chunk_overlap": 512}, False),
        ({"chunk_size": 256, "chunk_overlap": 512}, False),
    ),
)
def test_rule_chunk_size_bigger_than_chunk_overlap_returns(combination, expected_value):
    val = _rule_chunk_size_bigger_than_chunk_overlap(combination)
    assert val == expected_value


def test_rule_chunk_size_bigger_than_chunk_overlap_raises():
    with pytest.raises(SearchSpaceValueError):
        _ = _rule_chunk_size_bigger_than_chunk_overlap({"chunk_size": 512})


@pytest.mark.parametrize(
    "combination, expected_value",
    (
        ({"retrieval_method": "simple", "window_size": 0}, True),
        ({"retrieval_method": "simple", "window_size": 2}, False),
        ({"retrieval_method": "window", "window_size": 0}, False),
        ({"retrieval_method": "window", "window_size": 5}, True),
    ),
)
def test_rule_adjust_window_to_retrieval_method(combination, expected_value):
    val = _rule_adjust_window_to_retrieval_method(combination)
    assert val == expected_value


def test_rule_adjust_window_to_retrieval_method_raises():
    with pytest.raises(SearchSpaceValueError):
        _ = _rule_adjust_window_to_retrieval_method({"retrieval_method": "simple"})


class _MockEmbeddingModelWithParams:
    """Mock embedding model with params as a dataclass-like object."""

    def __init__(self, context_length):
        self.params = type("Params", (), {"context_length": context_length})()


class _MockEmbeddingModelWithDictParams:
    """Mock embedding model with params as a dict."""

    def __init__(self, context_length):
        self.params = {"context_length": context_length}


@pytest.mark.parametrize(
    "combination, expected_value",
    (
        # chunk_size=512 → estimated_tokens = 512 / 4 = 128, context=256 → True
        (
            {
                "chunk_size": 512,
                "embedding_model": _MockEmbeddingModelWithParams(256),
            },
            True,
        ),
        # chunk_size=2048 → estimated_tokens = 2048 / 4 = 512, context=256 → False
        (
            {
                "chunk_size": 2048,
                "embedding_model": _MockEmbeddingModelWithParams(256),
            },
            False,
        ),
        # Exact boundary: chunk_size=1024 → estimated_tokens = 256, context=256 → True
        (
            {
                "chunk_size": 1024,
                "embedding_model": _MockEmbeddingModelWithParams(256),
            },
            False,
        ),
    ),
)
def test_rule_chunk_size_within_embedding_context_length(combination, expected_value):
    val = _rule_chunk_size_within_embedding_context_length(combination)
    assert val == expected_value


def test_rule_chunk_size_within_embedding_context_length_no_context_length():
    """Rule returns True when context_length is not available on the embedding model."""
    mock_model = type("Model", (), {"params": type("Params", (), {"context_length": None})()})()
    combination = {"chunk_size": 2048, "chunk_overlap": 512, "embedding_model": mock_model}
    assert _rule_chunk_size_within_embedding_context_length(combination) is True


def test_rule_chunk_size_within_embedding_context_length_missing_fields():
    """Rule returns True when chunk_size, chunk_overlap or embedding_model are missing."""
    assert _rule_chunk_size_within_embedding_context_length({"chunk_size": 512}) is True
    assert _rule_chunk_size_within_embedding_context_length({}) is True


@pytest.mark.parametrize(
    "combination, expected_value",
    (
        # vector mode: all ranker params must be sentinels ("", 0, 1) or None for alpha
        ({"search_mode": "vector", "ranker_strategy": "", "ranker_k": 0, "ranker_alpha": 1}, True),
        # vector mode: ranker_alpha=None is also accepted as sentinel
        ({"search_mode": "vector", "ranker_strategy": "", "ranker_k": 0, "ranker_alpha": None}, True),
        # vector mode: ranker params absent (all None via .get) -> True
        ({"search_mode": "vector"}, True),
        # vector mode: ranker_strategy set -> False
        ({"search_mode": "vector", "ranker_strategy": "rrf", "ranker_k": 0, "ranker_alpha": 1}, False),
        # vector mode: ranker_k set -> False
        ({"search_mode": "vector", "ranker_strategy": "", "ranker_k": 60, "ranker_alpha": 1}, False),
        # vector mode: ranker_alpha not sentinel -> False
        ({"search_mode": "vector", "ranker_strategy": "", "ranker_k": 0, "ranker_alpha": 0.5}, False),
        # hybrid mode: ranker_strategy must be non-empty
        ({"search_mode": "hybrid", "ranker_strategy": "rrf", "ranker_k": 60, "ranker_alpha": 1}, True),
        ({"search_mode": "hybrid", "ranker_strategy": "weighted", "ranker_k": 60, "ranker_alpha": 0.5}, True),
        ({"search_mode": "hybrid", "ranker_strategy": "normalized", "ranker_k": 0, "ranker_alpha": 1}, True),
        # hybrid mode: empty ranker_strategy -> False
        ({"search_mode": "hybrid", "ranker_strategy": "", "ranker_k": 0, "ranker_alpha": 1}, False),
    ),
)
def test_rule_search_mode_ranker_consistency(combination, expected_value):
    val = _rule_search_mode_ranker_consistency(combination)
    assert val == expected_value


def test_rule_search_mode_ranker_consistency_missing_search_mode():
    """Rule returns True when search_mode is not present (backward compat)."""
    assert _rule_search_mode_ranker_consistency({"ranker_strategy": "rrf"}) is True


@pytest.mark.parametrize(
    "combination, expected_value",
    (
        # non-weighted strategy: alpha must be 1 (sentinel)
        ({"ranker_strategy": "rrf", "ranker_alpha": 1}, True),
        ({"ranker_strategy": "normalized", "ranker_alpha": 1}, True),
        # non-weighted strategy: alpha != 1 -> False
        ({"ranker_strategy": "rrf", "ranker_alpha": 0.5}, False),
        ({"ranker_strategy": "rrf", "ranker_alpha": 0}, False),
        # weighted strategy: alpha must not be 1
        ({"ranker_strategy": "weighted", "ranker_alpha": 0.5}, True),
        ({"ranker_strategy": "weighted", "ranker_alpha": 0.7}, True),
        ({"ranker_strategy": "weighted", "ranker_alpha": 0}, True),
        # weighted strategy: alpha == 1 -> False
        ({"ranker_strategy": "weighted", "ranker_alpha": 1}, False),
        # empty strategy (sentinel): alpha must be 1
        ({"ranker_strategy": "", "ranker_alpha": 1}, True),
        ({"ranker_strategy": "", "ranker_alpha": 0.5}, False),
    ),
)
def test_rule_ranker_alpha_for_weighted_only(combination, expected_value):
    val = _rule_ranker_alpha_for_weighted_only(combination)
    assert val == expected_value


def test_rule_ranker_alpha_for_weighted_only_missing_fields():
    """Rule returns True when fields are missing."""
    assert _rule_ranker_alpha_for_weighted_only({}) is True
    assert _rule_ranker_alpha_for_weighted_only({"ranker_strategy": "rrf"}) is True


@pytest.mark.parametrize(
    "combination, expected_value",
    (
        # rrf strategy: ranker_k must not be 0
        ({"ranker_strategy": "rrf", "ranker_k": 20}, True),
        ({"ranker_strategy": "rrf", "ranker_k": 60}, True),
        ({"ranker_strategy": "rrf", "ranker_k": 100}, True),
        # rrf strategy: ranker_k == 0 -> False
        ({"ranker_strategy": "rrf", "ranker_k": 0}, False),
        # non-rrf strategies: ranker_k must be 0 (sentinel)
        ({"ranker_strategy": "weighted", "ranker_k": 0}, True),
        ({"ranker_strategy": "normalized", "ranker_k": 0}, True),
        ({"ranker_strategy": "", "ranker_k": 0}, True),
        # non-rrf strategies: ranker_k != 0 -> False
        ({"ranker_strategy": "weighted", "ranker_k": 60}, False),
        ({"ranker_strategy": "normalized", "ranker_k": 20}, False),
        ({"ranker_strategy": "", "ranker_k": 100}, False),
    ),
)
def test_rule_ranker_k_for_rrf_only(combination, expected_value):
    val = _rule_ranker_k_for_rrf_only(combination)
    assert val == expected_value


def test_rule_ranker_k_for_rrf_only_missing_fields():
    """Rule returns True when fields are missing."""
    assert _rule_ranker_k_for_rrf_only({}) is True
    assert _rule_ranker_k_for_rrf_only({"ranker_strategy": "rrf"}) is True
    assert _rule_ranker_k_for_rrf_only({"ranker_k": 60}) is True


@pytest.mark.parametrize(
    "combination, expected_value",
    (
        ({"chunking_method": "markdown_header", "chunk_size": 0}, True),
        ({"chunking_method": "markdown_header", "chunk_size": 1024}, True),
        ({"chunking_method": "recursive", "chunk_size": 0}, False),
        ({"chunking_method": "recursive", "chunk_size": 512}, True),
        ({"chunking_method": "markdown", "chunk_size": 0}, False),
        ({"chunking_method": "markdown", "chunk_size": 1024}, True),
    ),
)
def test_rule_chunk_params_consistency_with_method(combination, expected_value):
    val = _rule_chunk_params_consistency_with_method(combination)
    assert val == expected_value


def test_rule_chunk_params_consistency_with_method_missing_fields():
    """Rule returns True when fields are missing."""
    assert _rule_chunk_params_consistency_with_method({}) is True
    assert _rule_chunk_params_consistency_with_method({"chunking_method": "recursive"}) is True
    assert _rule_chunk_params_consistency_with_method({"chunk_size": 512}) is True


def test_rule_chunk_size_bigger_than_chunk_overlap_skips_zero():
    """When chunk_size is 0 (sentinel), the rule passes."""
    assert _rule_chunk_size_bigger_than_chunk_overlap({"chunk_size": 0, "chunk_overlap": 0}) is True


class TestSearchSpace:
    def test_initialization(self, mocked_params):
        search_space = SearchSpace(params=mocked_params)

        assert search_space.as_list() == mocked_params
        assert search_space.as_dict() == {"a": mocked_params[0].all_values(), "b": mocked_params[1].all_values()}

    def test_get_item(self, mocked_params):
        search_space = SearchSpace(params=mocked_params)

        assert search_space["a"] == mocked_params[0]

    def test_params_setter_raises_error(self):
        with pytest.raises(SearchSpaceValueError):
            _ = SearchSpace(
                params=[
                    Parameter(name="a", param_type="C", values=[1, 2]),
                    Parameter(name="a", param_type="C", values=[3, 4]),
                ]
            )

    def test_combinations(self):
        search_space = SearchSpace(
            params=[
                Parameter(name="a", param_type="I", v_min=1, v_max=2),
                Parameter(name="b", param_type="C", values=[5, 6]),
            ],
        )
        assert search_space.combinations == [{"a": 1, "b": 5}, {"a": 1, "b": 6}, {"a": 2, "b": 5}, {"a": 2, "b": 6}]

    def test_apply_custom_rules(self, mocked_params):
        def _custom_rule(combination: dict) -> bool:
            if combination["a"] == 2 and combination["b"] == 6:
                return False
            return True

        search_space = SearchSpace(params=mocked_params, rules=[_custom_rule])

        assert (
            search_space.max_combinations == len(mocked_params[0].all_values()) * len(mocked_params[1].all_values()) - 1
        )


_HYBRID_PARAM_NAMES = {"search_mode", "ranker_strategy", "ranker_k", "ranker_alpha"}
_MOCK_FM = type("MockFM", (), {"__hash__": lambda self: 1, "model_id": "mock-fm"})()
_MOCK_EM = type("MockEM", (), {"__hash__": lambda self: 2, "model_id": "mock-em", "params": None})()
_REQUIRED_PARAMS = [
    Parameter(name="foundation_model", values=[_MOCK_FM]),
    Parameter(name="embedding_model", values=[_MOCK_EM]),
]


class TestGetDefaultSearchSpaceParameters:
    def test_ls_milvus_excludes_hybrid_params_by_default(self):
        params = get_default_ai4rag_search_space_parameters(vector_store_type="ls_milvus")
        param_names = {p.name for p in params}

        assert "search_mode" in param_names
        assert "ranker_strategy" not in param_names
        assert "ranker_k" not in param_names
        assert "ranker_alpha" not in param_names

        search_mode_param = next(p for p in params if p.name == "search_mode")
        assert search_mode_param.values == ("vector",)

    def test_chroma_excludes_hybrid_params(self):
        params = get_default_ai4rag_search_space_parameters(vector_store_type="chroma")
        param_names = {p.name for p in params}

        assert "search_mode" in param_names
        assert "ranker_strategy" not in param_names
        assert "ranker_k" not in param_names
        assert "ranker_alpha" not in param_names

        search_mode_param = next(p for p in params if p.name == "search_mode")
        assert search_mode_param.values == ("vector",)
        assert "hybrid" not in search_mode_param.values

    def test_default_is_ls_milvus(self):
        params_default = get_default_ai4rag_search_space_parameters()
        params_milvus = get_default_ai4rag_search_space_parameters(vector_store_type="ls_milvus")
        assert params_default == params_milvus

    def test_common_params_present_for_both_types(self):
        common_params = {
            "chunking_method",
            "chunk_size",
            "chunk_overlap",
            "retrieval_method",
            "window_size",
            "number_of_chunks",
            "search_mode",
        }
        for vs_type in ("ls_milvus", "chroma"):
            params = get_default_ai4rag_search_space_parameters(vector_store_type=vs_type)
            param_names = {p.name for p in params}
            assert common_params.issubset(param_names)


class TestAI4RAGSearchSpaceVectorStoreType:
    def test_ls_milvus_excludes_hybrid_params_by_default(self):
        ss = AI4RAGSearchSpace(params=list(_REQUIRED_PARAMS), vector_store_type="ls_milvus")
        param_names = {p.name for p in ss.params}
        assert "search_mode" in param_names
        assert not _HYBRID_PARAM_NAMES.intersection({"ranker_strategy", "ranker_k", "ranker_alpha"}).intersection(
            param_names
        )

    def test_chroma_excludes_hybrid_params(self):
        ss = AI4RAGSearchSpace(params=list(_REQUIRED_PARAMS), vector_store_type="chroma")
        param_names = {p.name for p in ss.params}
        assert "search_mode" in param_names
        assert not _HYBRID_PARAM_NAMES.intersection({"ranker_strategy", "ranker_k", "ranker_alpha"}).intersection(
            param_names
        )

    def test_chroma_search_mode_only_vector(self):
        ss = AI4RAGSearchSpace(params=list(_REQUIRED_PARAMS), vector_store_type="chroma")
        search_mode_param = ss["search_mode"]
        assert search_mode_param.values == ("vector",)

    def test_default_vector_store_type_is_ls_milvus(self):
        ss = AI4RAGSearchSpace(params=list(_REQUIRED_PARAMS))
        param_names = {p.name for p in ss.params}
        assert "search_mode" in param_names
        assert not _HYBRID_PARAM_NAMES.intersection({"ranker_strategy", "ranker_k", "ranker_alpha"}).intersection(
            param_names
        )

    def test_chroma_does_not_apply_hybrid_rules(self):
        ss = AI4RAGSearchSpace(params=list(_REQUIRED_PARAMS), vector_store_type="chroma")
        for combination in ss.combinations:
            assert "ranker_strategy" not in combination
            assert "ranker_k" not in combination
            assert "ranker_alpha" not in combination

    def test_ls_milvus_default_only_vector_mode(self):
        ss = AI4RAGSearchSpace(params=list(_REQUIRED_PARAMS), vector_store_type="ls_milvus")
        for combination in ss.combinations:
            assert combination.get("search_mode") == "vector"
            assert "ranker_strategy" not in combination
            assert "ranker_k" not in combination
            assert "ranker_alpha" not in combination

    def test_ls_milvus_user_provided_hybrid_params_apply_rules(self):
        hybrid_params = list(_REQUIRED_PARAMS) + [
            Parameter(name="search_mode", values=("vector", "hybrid")),
            Parameter(name="ranker_strategy", values=("", "rrf", "weighted")),
            Parameter(name="ranker_k", values=(0, 60)),
            Parameter(name="ranker_alpha", values=(1, 0.5)),
        ]
        ss = AI4RAGSearchSpace(params=hybrid_params, vector_store_type="ls_milvus")
        for combination in ss.combinations:
            search_mode = combination.get("search_mode")
            if search_mode == "vector":
                assert combination["ranker_strategy"] == ""
                assert combination["ranker_k"] == 0
                assert combination["ranker_alpha"] == 1
            elif search_mode == "hybrid":
                assert combination["ranker_strategy"] in ("rrf", "weighted", "normalized")
                if combination["ranker_strategy"] == "rrf":
                    assert combination["ranker_k"] > 0
                else:
                    assert combination["ranker_k"] == 0
                if combination["ranker_strategy"] == "weighted":
                    assert combination["ranker_alpha"] != 1
                else:
                    assert combination["ranker_alpha"] == 1
