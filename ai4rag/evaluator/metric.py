# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Literal

from ai4rag.utils.constants import ConstantMeta


@dataclass(frozen=True)
class RAGMetric:
    """Representation of a single metric that can be used in AI4RAG."""

    name: str
    evaluator: Literal["unitxt", "judge", "custom"]
    description: str


class Metrics(metaclass=ConstantMeta):
    """AI4RAG available metrics."""

    ANSWER_CORRECTNESS = RAGMetric(
        name="answer_correctness",
        evaluator="unitxt",
        description="Measures how accurately the generated answer matches the ground-truth reference answers.",
    )
    FAITHFULNESS = RAGMetric(
        name="faithfulness",
        evaluator="unitxt",
        description="Measures whether the generated answer is grounded in the retrieved context without hallucination.",
    )
    CONTEXT_CORRECTNESS = RAGMetric(
        name="context_correctness",
        evaluator="unitxt",
        description="Measures the relevance and correctness of the retrieved context passages for the given question.",
    )
    OVERALL_SCORE = RAGMetric(
        name="overall_score",
        evaluator="custom",
        description="Aggregate score computed as the mean of all other evaluated metrics.",
    )
