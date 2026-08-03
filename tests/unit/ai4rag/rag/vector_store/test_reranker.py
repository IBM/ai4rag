# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.rag.vector_store.reranker import WeightedInMemoryAggregator


class TestNormalizeScores:

    def test_normal_range(self):
        scores = {"a": 1.0, "b": 3.0, "c": 5.0}
        result = WeightedInMemoryAggregator._normalize_scores(scores)
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == pytest.approx(1.0)

    def test_identical_scores(self):
        scores = {"a": 2.0, "b": 2.0}
        result = WeightedInMemoryAggregator._normalize_scores(scores)
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    def test_empty_dict(self):
        assert WeightedInMemoryAggregator._normalize_scores({}) == {}

    def test_single_element(self):
        result = WeightedInMemoryAggregator._normalize_scores({"x": 42.0})
        assert result["x"] == 1.0


class TestWeightedRerank:

    def test_equal_alpha(self):
        vec = {"a": 1.0, "b": 3.0}
        kw = {"a": 3.0, "b": 1.0}
        result = WeightedInMemoryAggregator.weighted_rerank(vec, kw, alpha=0.5)
        assert result["a"] == pytest.approx(result["b"])

    def test_alpha_one_means_vector_only(self):
        vec = {"a": 1.0, "b": 5.0}
        kw = {"a": 5.0, "b": 1.0}
        result = WeightedInMemoryAggregator.weighted_rerank(vec, kw, alpha=1.0)
        assert result["b"] > result["a"]

    def test_alpha_zero_means_keyword_only(self):
        vec = {"a": 1.0, "b": 5.0}
        kw = {"a": 5.0, "b": 1.0}
        result = WeightedInMemoryAggregator.weighted_rerank(vec, kw, alpha=0.0)
        assert result["a"] > result["b"]

    def test_disjoint_ids(self):
        vec = {"a": 1.0}
        kw = {"b": 2.0}
        result = WeightedInMemoryAggregator.weighted_rerank(vec, kw, alpha=0.5)
        assert "a" in result
        assert "b" in result


class TestRRFRerank:

    def test_basic_rrf(self):
        vec = {"a": 10.0, "b": 5.0}
        kw = {"a": 5.0, "b": 10.0}
        result = WeightedInMemoryAggregator.rrf_rerank(vec, kw, k=60)
        assert result["a"] == pytest.approx(result["b"])

    def test_default_k(self):
        vec = {"x": 1.0}
        kw = {"x": 1.0}
        result = WeightedInMemoryAggregator.rrf_rerank(vec, kw)
        expected = 2.0 / (60.0 + 1.0)
        assert result["x"] == pytest.approx(expected)

    def test_disjoint_ids_penalized(self):
        vec = {"a": 1.0}
        kw = {"b": 1.0}
        result = WeightedInMemoryAggregator.rrf_rerank(vec, kw, k=60)
        both_present = WeightedInMemoryAggregator.rrf_rerank({"a": 1.0}, {"a": 1.0}, k=60)
        assert both_present["a"] > result["a"]
        assert both_present["a"] > result["b"]


class TestCombineSearchResults:

    def test_rrf_dispatch(self):
        vec = {"a": 10.0, "b": 5.0}
        kw = {"a": 5.0, "b": 10.0}
        result = WeightedInMemoryAggregator.combine_search_results(vec, kw, "rrf")
        assert "a" in result and "b" in result

    def test_weighted_dispatch(self):
        vec = {"a": 10.0, "b": 5.0}
        kw = {"a": 5.0, "b": 10.0}
        result = WeightedInMemoryAggregator.combine_search_results(vec, kw, "weighted", {"alpha": 0.7})
        assert "a" in result and "b" in result

    def test_normalized_falls_through_to_rrf(self):
        vec = {"a": 10.0}
        kw = {"a": 5.0}
        rrf_result = WeightedInMemoryAggregator.combine_search_results(vec, kw, "rrf")
        norm_result = WeightedInMemoryAggregator.combine_search_results(vec, kw, "normalized")
        assert rrf_result == norm_result

    def test_no_params_uses_defaults(self):
        vec = {"a": 1.0}
        kw = {"a": 2.0}
        result = WeightedInMemoryAggregator.combine_search_results(vec, kw)
        assert "a" in result
