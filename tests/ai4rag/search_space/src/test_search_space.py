# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.search_space.src.search_space import (
    SearchSpace,
    AI4RAGSearchSpace,
    _rule_adjust_window_to_retrieval_method,
    _rule_chunk_size_bigger_than_chunk_overlap,
    SearchSpaceValueError,
    Parameter,
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
    )
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
    )
)
def test_rule_adjust_window_to_retrieval_method(combination, expected_value):
    val = _rule_adjust_window_to_retrieval_method(combination)
    assert val == expected_value


def test_rule_adjust_window_to_retrieval_method_raises():
    with pytest.raises(SearchSpaceValueError):
        _ = _rule_adjust_window_to_retrieval_method({"retrieval_method": "simple"})


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
            search_space = SearchSpace(
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

        assert search_space.max_combinations == len(mocked_params[0].all_values()) * len(mocked_params[1].all_values()) - 1
