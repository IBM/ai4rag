# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from dataclasses import asdict, dataclass
from typing import Any, Generator

from ai4rag import logger
from ai4rag.evaluator.base_evaluator import EvaluationData, EvaluationMetricsResult


@dataclass(frozen=True)
class EvaluationResult:
    """
    Class holding data from single evaluation.

    Parameters
    ----------
    pattern_name : str
        Name of the RAG pattern created. Next names should be
        i.e. Pattern1, Pattern2, ..., PatternX

    collection : list[str]
        Name of the indexes/collections in the VectorStores for single
        configuration.

    indexing_params : dict[str, Any] | None
        Subspace of hyperparameters used during indexing stage in the
        Retrieval Augmented Generation.

    rag_params : dict[str, Any]
        Subspace of hyperparameters used during inference stage in the
        Retrieval Augmented Generation.

    scores : EvaluationMetricsResult
        Evaluation output with aggregate metrics and per-question scores.

    execution_time : float
        Time in seconds how long did experiment take to run.

    final_score : float
        Single score calculated for optimization process as the value to be minimized or maximized.
    """

    pattern_name: str
    collection: str
    indexing_params: dict[str, Any]
    rag_params: dict[str, Any]
    scores: EvaluationMetricsResult
    execution_time: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        """Cast instance to dict"""
        return asdict(self)


class ExperimentResults:
    """
    Class holding information about each run in the optimization process.

    Attributes
    ----------
    evaluations : list[EvaluationResult]
        Instances of each evaluation.

    evaluation_data : list[list[EvaluationData]]
        Data for evaluation, questions, ground truth answers and context references.

    Methods
    -------
    add_evaluation()
        Extend space with another evaluation step from the objective function.

    sorted_(reverse=False)
        Return sorted version of the evaluations.

    get_best_evaluations(k=None)
        Return k-best evaluations.
        If k is None all evaluations are returned sorted based on optimization metric score.
    """

    def __init__(self):
        self.evaluations: list[EvaluationResult] = []
        self.evaluation_data: list[list[EvaluationData]] = []

    def __len__(self) -> int:
        return len(self.evaluations)

    def __iter__(self) -> Generator[EvaluationResult, None, None]:
        yield from self.evaluations

    def __bool__(self) -> bool:
        return bool(self.evaluations)

    def add_evaluation(
        self,
        evaluation_data: list[EvaluationData],
        evaluation_result: EvaluationResult,
    ) -> None:
        """
        This method is responsible for adding searched point in the parameters space.

        Parameters
        ----------
        evaluation_data : list[EvaluationData]
            Data used to make evaluation. It is a list, as one instance
            of EvaluationData is per 1 question and answer.

        evaluation_result : EvaluationResult
            Instance containing information about single HPO run.
        """
        logger.info("Adding evaluation results and evaluation data to training results.")
        self.evaluations.append(evaluation_result)
        self.evaluation_data.append(evaluation_data)

    def evaluation_explored_or_cached(
        self, indexing_params: dict[str, Any], rag_params: dict[str, Any]
    ) -> float | None:
        """
        This method checks if an evaluation to certain params already exists.

        Parameters
        ----------
        indexing_params : dict[str, Any]
            Dictionary containing keys and values that are compared with
            previously used ones, to establish if the evaluation already done.

        rag_params : dict[str, Any]
            Dictionary containing keys and values that are compared with
            previously used ones, to establish if the evaluation already done.

        Returns
        -------
        float | None
            The final score of the evaluation if it exists, otherwise None.
        """
        for evaluation in self.evaluations:
            if evaluation.indexing_params == indexing_params and evaluation.rag_params == rag_params:
                return evaluation.final_score
        return None

    def get_existing_collection(self, indexing_params: dict[str, Any]) -> str | None:
        """
        This method checks if a collection to certain params already exists.
        If so, returns the collection name.

        Parameters
        ----------
        indexing_params : dict[str, Any]
            Dictionary containing keys and values that are compared with
            previously used ones, to establish if the index already created.

        Returns
        -------
        str | None
            The collection name of the index if it exists, otherwise None.
        """
        for evaluation in self.evaluations:
            if evaluation.indexing_params == indexing_params:
                return evaluation.collection
        return None

    @property
    def collection_names(self) -> list[str]:
        """
        Return list of collection_names from the HPO run.

        Returns
        -------
        list[str]
            Collections/indexes names created during all HPO runs.
        """
        ret = set()
        for ev in self.evaluations:
            ret.add(ev.collection)
        return list(ret)

    def get_best_evaluations(self, k: int | None = None) -> tuple[EvaluationResult]:
        """
        Return k-best evaluations.

        Parameters
        ----------
        k : int
            how many evaluations to return

        Returns
        -------
        tuple[EvaluationResult]
            the best evaluations from the AutoRAG experiment
        """
        evaluations = list(self.evaluations)
        sorted_evs = tuple(sorted(evaluations, key=lambda x: x.final_score, reverse=True))
        return sorted_evs[:k]

    @staticmethod
    def create_evaluation_results_json(
        evaluation_data: list[EvaluationData], evaluation_result: EvaluationResult
    ) -> list[dict[str, Any]]:
        """
        Create json made of evaluation results with proper data.

        Example file content::

            [
                {
                    "question_id": "0",
                    "answer": "<model's answer>",
                    "answer_contexts": [
                        {"text": "<content1_text>", "document_id": "document_1.pdf"},
                    ],
                    "metrics": [
                        {"name": "answer_correctness", "evaluator": "unitxt", "score": 0.79},
                        {"name": "overall_score", "evaluator": "custom", "score": 0.79},
                    ]
                },
            ]

        Parameters
        ----------
        evaluation_data : list[EvaluationData]
            Evaluation data to be used for the json generation.

        evaluation_result : EvaluationResult
            Evaluation result to be used for the json generation.

        Returns
        -------
        list[dict[str, Any]]
            json-like object with evaluation results.
        """
        scores_by_qid = {q["question_id"]: q["metrics"] for q in evaluation_result.scores["question_scores"]}

        data = []
        for ev in evaluation_data:
            data.append(
                {
                    "question": ev.question,
                    "correct_answers": ev.ground_truths,
                    "answer": ev.answer,
                    "answer_contexts": [
                        {"text": text, "document_id": doc_id} for text, doc_id in zip(ev.contexts, ev.context_ids)
                    ],
                    "metrics": [
                        {"name": m["name"], "evaluator": m["evaluator"], "score": m["value"]}
                        for m in scores_by_qid.get(ev.question_id, [])
                    ],
                }
            )

        return data
