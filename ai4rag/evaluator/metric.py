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
    evaluator: Literal["unitxt", "judge", "ragas", "custom"]
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
        description="Measures whether the retrieved context passages match the ground-truth reference documents.",
    )

    JUDGE_ANSWER_RELEVANCE = RAGMetric(
        name="answer_relevance",
        evaluator="judge",
        description="LLM judge score for how directly and helpfully the response addresses the question.",
    )

    RAGAS_FAITHFULNESS = RAGMetric(
        name="faithfulness",
        evaluator="ragas",
        description="RAGAS score for how well the answer is grounded in the retrieved context without hallucination.",
    )
    RAGAS_ANSWER_RELEVANCY = RAGMetric(
        name="answer_relevancy",
        evaluator="ragas",
        description="RAGAS score for how relevant and on-topic the answer is to the question.",
    )
    RAGAS_CONTEXT_PRECISION = RAGMetric(
        name="context_precision",
        evaluator="ragas",
        description="RAGAS score for whether the retrieved contexts relevant to the ground truth are ranked highly.",
    )
    RAGAS_CONTEXT_RECALL = RAGMetric(
        name="context_recall",
        evaluator="ragas",
        description="RAGAS score for how much of the ground-truth answer is covered by the retrieved contexts.",
    )

    OVERALL_SCORE = RAGMetric(
        name="overall_score",
        evaluator="custom",
        description="Aggregate score computed as the mean of all other evaluated metrics.",
    )
