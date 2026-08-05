# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

__all__ = ["WeightedInMemoryAggregator"]


class WeightedInMemoryAggregator:
    """Combines vector and keyword search scores using reranking strategies."""

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalize scores to the ``[0, 1]`` range.

        When all scores are equal (zero range) every entry maps to ``1.0``; an
        empty mapping returns an empty mapping.

        Parameters
        ----------
        scores : dict[str, float]
            Scores keyed by chunk id.

        Returns
        -------
        dict[str, float]
            Normalized scores keyed by chunk id.
        """
        if not scores:
            return {}
        min_score, max_score = min(scores.values()), max(scores.values())
        score_range = max_score - min_score
        if score_range > 0:
            return {doc_id: (score - min_score) / score_range for doc_id, score in scores.items()}
        return dict.fromkeys(scores, 1.0)

    @staticmethod
    def weighted_rerank(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        alpha: float = 0.5,
    ) -> dict[str, float]:
        """Weighted average of normalized vector and keyword scores.

        Parameters
        ----------
        vector_scores : dict[str, float]
            Scores from vector search keyed by chunk id.
        keyword_scores : dict[str, float]
            Scores from keyword search keyed by chunk id.
        alpha : float
            Blend factor: ``0`` = keyword only, ``1`` = vector only.

        Returns
        -------
        dict[str, float]
            Combined scores keyed by chunk id.
        """
        all_ids = set(vector_scores) | set(keyword_scores)
        norm_vec = WeightedInMemoryAggregator._normalize_scores(vector_scores)
        norm_kw = WeightedInMemoryAggregator._normalize_scores(keyword_scores)
        return {
            doc_id: (1 - alpha) * norm_kw.get(doc_id, 0.0) + alpha * norm_vec.get(doc_id, 0.0) for doc_id in all_ids
        }

    @staticmethod
    def rrf_rerank(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        k: float = 60.0,
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion of vector and keyword search results.

        Parameters
        ----------
        vector_scores : dict[str, float]
            Scores from vector search keyed by chunk id.
        keyword_scores : dict[str, float]
            Scores from keyword search keyed by chunk id.
        k : float
            RRF smoothing constant (default ``60.0``).

        Returns
        -------
        dict[str, float]
            Fused RRF scores keyed by chunk id.
        """
        vector_ranks = {
            doc_id: i + 1
            for i, (doc_id, _) in enumerate(sorted(vector_scores.items(), key=lambda x: x[1], reverse=True))
        }
        keyword_ranks = {
            doc_id: i + 1
            for i, (doc_id, _) in enumerate(sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True))
        }

        all_ids = set(vector_scores) | set(keyword_scores)
        return {
            doc_id: 1.0 / (k + vector_ranks.get(doc_id, float("inf")))
            + 1.0 / (k + keyword_ranks.get(doc_id, float("inf")))
            for doc_id in all_ids
        }

    @staticmethod
    def combine_search_results(
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        reranker_type: str = "rrf",
        reranker_params: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Dispatch to the appropriate reranking strategy.

        Parameters
        ----------
        vector_scores : dict[str, float]
            Scores from vector search keyed by chunk id.
        keyword_scores : dict[str, float]
            Scores from keyword search keyed by chunk id.
        reranker_type : str
            ``"rrf"``, ``"weighted"``, or ``"normalized"`` (falls through to RRF).
        reranker_params : dict[str, Any] | None
            Strategy-specific params: ``{"k": float}`` for RRF,
            ``{"alpha": float}`` for weighted.

        Returns
        -------
        dict[str, float]
            Combined scores keyed by chunk id, produced by the selected strategy.
        """
        if reranker_params is None:
            reranker_params = {}

        if reranker_type == "weighted":
            alpha = reranker_params.get("alpha", 0.5)
            return WeightedInMemoryAggregator.weighted_rerank(vector_scores, keyword_scores, alpha)

        k = reranker_params.get("k", 60.0)
        return WeightedInMemoryAggregator.rrf_rerank(vector_scores, keyword_scores, k)
