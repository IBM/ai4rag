# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import numpy as np

from ai4rag.evaluator.score_utils import enrich_with_overall_score


def test_enrich_with_overall_score():
    result = enrich_with_overall_score(
        {
            "scores": {
                "faithfulness": {"mean": 0.8, "ci_low": 0.7, "ci_high": 0.9},
                "answer_correctness": {"mean": 0.6, "ci_low": 0.5, "ci_high": 0.7},
                "context_correctness": {"mean": 0.4, "ci_low": 0.3, "ci_high": 0.5},
                "answer_relevance": {"mean": 0.9, "ci_low": 0.8, "ci_high": 1.0},
            },
            "question_scores": {
                "faithfulness": {"q1": 0.8, "q2": 1.0},
                "answer_correctness": {"q1": 0.6, "q2": 0.8},
                "context_correctness": {"q1": 0.4, "q2": 0.6},
                "answer_relevance": {"q1": 0.9, "q2": 0.7},
            },
        }
    )

    assert result["scores"]["overall_score"]["mean"] == round(float(np.mean([0.675, 0.775])), 4)
    assert result["question_scores"]["overall_score"]["q1"] == 0.675
    assert result["question_scores"]["overall_score"]["q2"] == 0.775
