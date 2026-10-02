# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ai4rag.core.experiment.exception_handler import EvaluationError
from ai4rag.core.experiment.results import EvaluationResult
from ai4rag.core.experiment.utils import merge_evaluation_results
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.evaluator.base_evaluator import (
    AggregateMetric,
    BaseEvaluator,
    ConfidenceInterval,
    EvaluationMetricsResult,
    QuestionMetric,
    QuestionScore,
)
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from ai4rag.evaluator.metric import Metrics, RAGMetric
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.rag.template.agentic_rag_template import AgenticRAG
from ai4rag.rag.template.simple_rag_template import SimpleRAG
from ai4rag.rag.vector_store.config import MilvusLiteConfig
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BENCHMARK_DF = pd.DataFrame(
    {
        "question": ["What is Python?"],
        "correct_answers": [["A programming language."]],
        "correct_answer_document_keys": [["doc1"]],
    }
)


def _make_result(
    metric_name: str,
    evaluator: str,
    mean: float,
    question_scores: dict[str, float],
) -> EvaluationMetricsResult:
    """Build a minimal EvaluationMetricsResult for testing."""
    return EvaluationMetricsResult(
        metrics=[
            AggregateMetric(
                name=metric_name,
                evaluator=evaluator,
                description="",
                scores=ConfidenceInterval(mean=mean, ci_low=None, ci_high=None),
            )
        ],
        question_scores=[
            QuestionScore(
                question_id=qid,
                metrics=[QuestionMetric(name=metric_name, evaluator=evaluator, value=val)],
            )
            for qid, val in question_scores.items()
        ],
    )


def _build_experiment(evaluators=None, optimization_metric=Metrics.FAITHFULNESS, metrics=None, rag_template=None):
    """Construct an AI4RAGExperiment with all heavy deps mocked out."""
    from ai4rag.core.experiment.experiment import AI4RAGExperiment

    kwargs = {}
    if evaluators is not None:
        kwargs["evaluators"] = evaluators
    if metrics is not None:
        kwargs["metrics"] = metrics
    if rag_template is not None:
        kwargs["rag_template"] = rag_template
    return AI4RAGExperiment(
        documents=[],
        benchmark_data=_BENCHMARK_DF,
        search_space=MagicMock(),
        vector_store_config=MilvusLiteConfig(db_path="./ai4rag.db"),
        optimizer_settings=MagicMock(),
        event_handler=MagicMock(),
        client=MagicMock(),
        optimization_metric=optimization_metric,
        **kwargs,
    )


def _make_llmaj_evaluator():
    model = MagicMock()
    model.model_id = "judge-model"
    return LLMaJEvaluator(model=model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluatorType:
    def test_base_evaluator_has_empty_default(self):
        assert BaseEvaluator.EVALUATOR_TYPE == ""

    def test_unitxt_evaluator_type(self):
        assert UnitxtEvaluator.EVALUATOR_TYPE == "unitxt"

    def test_llmaj_evaluator_type(self):
        assert LLMaJEvaluator.EVALUATOR_TYPE == "judge"


class TestRAGTemplateSelection:
    def test_agentic_rag_is_the_default(self):
        assert _build_experiment().rag_template is AgenticRAG

    def test_accepts_another_rag_template(self):
        assert _build_experiment(rag_template=SimpleRAG).rag_template is SimpleRAG

    def test_preselection_receives_selected_rag_template(self, mocker):
        experiment = _build_experiment(rag_template=SimpleRAG)
        selector_class = mocker.patch("ai4rag.core.experiment.mps.ModelsPreSelector")
        selector = selector_class.return_value
        selector.select_models.return_value = {"foundation_models": [], "embedding_models": []}

        experiment.run_pre_selection(foundation_models=[], embedding_models=[])

        assert selector_class.call_args.kwargs["rag_template"] is SimpleRAG

    @pytest.mark.parametrize("invalid_template", ["SimpleRAG", object])
    def test_rejects_non_template_classes(self, invalid_template):
        with pytest.raises(TypeError, match="rag_template must be a BaseRAGTemplate subclass"):
            _build_experiment(rag_template=invalid_template)


class TestEvaluatorsSetter:
    def test_default_evaluators_is_unitxt_only(self):
        exp = _build_experiment()
        assert len(exp.evaluators) == 1
        assert isinstance(exp.evaluators[0], UnitxtEvaluator)

    def test_explicit_evaluators_are_stored(self):
        evals = [UnitxtEvaluator(), _make_llmaj_evaluator()]
        exp = _build_experiment(evaluators=evals)
        assert len(exp.evaluators) == 2
        assert isinstance(exp.evaluators[0], UnitxtEvaluator)
        assert isinstance(exp.evaluators[1], LLMaJEvaluator)

    def test_setter_rejects_non_evaluator_instances(self):
        with pytest.raises(ValueError, match="BaseEvaluator"):
            _build_experiment(evaluators=[UnitxtEvaluator(), "not_an_evaluator"])


class TestDefaultMetrics:
    def test_unitxt_only_defaults(self):
        exp = _build_experiment()
        names = [m.name for m in exp.metrics]
        assert "answer_correctness" in names
        assert "faithfulness" in names
        assert "context_correctness" in names
        assert "overall_score" in names
        assert "answer_relevance" not in names

    def test_with_judge_evaluator_includes_answer_relevance(self):
        evals = [UnitxtEvaluator(), _make_llmaj_evaluator()]
        exp = _build_experiment(evaluators=evals)
        names = [m.name for m in exp.metrics]
        assert "answer_relevance" in names
        assert "overall_score" in names

    def test_explicit_metrics_override_defaults(self):
        exp = _build_experiment(metrics=(Metrics.FAITHFULNESS,))
        assert len(exp.metrics) == 1
        assert exp.metrics[0].name == "faithfulness"

    def test_metrics_ragmetric_instances(self):
        exp = _build_experiment(metrics=[Metrics.FAITHFULNESS, Metrics.ANSWER_CORRECTNESS])
        assert len(exp.metrics) == 2
        assert all(isinstance(m, RAGMetric) for m in exp.metrics)
        assert [m.name for m in exp.metrics] == ["faithfulness", "answer_correctness"]

    def test_metrics_string_raises(self):
        with pytest.raises(TypeError, match="RAGMetric instance selected from Metrics"):
            _build_experiment(metrics=["faithfulness"])

    def test_metrics_unknown_ragmetric_raises(self):
        unknown = RAGMetric(name="nonexistent", evaluator="unitxt", description="")
        with pytest.raises(ValueError, match="Unknown RAGMetric 'nonexistent'"):
            _build_experiment(metrics=[unknown])

    def test_metrics_wrong_type_element_raises(self):
        with pytest.raises(TypeError, match="RAGMetric instance selected from Metrics"):
            _build_experiment(metrics=[42])

    def test_metrics_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _build_experiment(metrics=[])


class TestMetricEvaluatorValidation:
    def test_unitxt_metric_with_unitxt_evaluator_passes(self):
        _build_experiment(optimization_metric=Metrics.FAITHFULNESS)

    def test_custom_metric_always_passes(self):
        _build_experiment(optimization_metric=Metrics.OVERALL_SCORE)

    def test_judge_metric_without_judge_evaluator_raises(self):
        with pytest.raises(ValueError, match="requires a 'judge' evaluator"):
            _build_experiment(optimization_metric=Metrics.JUDGE_ANSWER_RELEVANCE)

    def test_judge_metric_with_judge_evaluator_passes(self):
        evals = [UnitxtEvaluator(), _make_llmaj_evaluator()]
        _build_experiment(evaluators=evals, optimization_metric=Metrics.JUDGE_ANSWER_RELEVANCE)


class TestGAMPatternPublication:
    """Final GAM reporting publishes only the selected warm-start result."""

    def test_publishes_best_experiment_results_in_score_order(self, mocker):
        experiment = _build_experiment()
        for index, (name, score) in enumerate(
            (
                ("Pattern1-warm-start", 0.7),
                ("Pattern2-warm-start", 0.9),
                ("Pattern2", 0.2),
                ("Pattern3", 0.8),
            )
        ):
            experiment.results.add_evaluation(
                [],
                EvaluationResult(
                    pattern_name=name,
                    collection="collection",
                    indexing_params={},
                    rag_params={},
                    scores={"metrics": [], "question_scores": []},
                    execution_time=0.0,
                    final_score=score,
                ),
            )
        publish = mocker.patch.object(experiment, "_stream_finished_pattern")

        experiment._publish_best_warm_start_pattern()

        assert [call.kwargs["evaluation_result"].final_score for call in publish.call_args_list] == [0.9]
        assert publish.call_args_list[0].kwargs["pattern_name"] == "Pattern1"
        assert [call.kwargs["iteration"] for call in publish.call_args_list] == [0]
        published_patterns = pd.DataFrame(
            [
                {
                    "source_pattern": call.kwargs["evaluation_result"].pattern_name,
                    "published_pattern": call.kwargs.get("pattern_name")
                    or call.kwargs["evaluation_result"].pattern_name,
                    "score": call.kwargs["evaluation_result"].final_score,
                }
                for call in publish.call_args_list
            ]
        )
        print(f"\nPublished patterns:\n{published_patterns.to_string(index=False)}")

    def test_does_not_publish_an_unscored_warm_start_candidate(self, mocker):
        """A failed warm-start candidate cannot become final Pattern1."""
        experiment = _build_experiment()
        for name, score in (
            ("Pattern1-warm-start", None),
            ("Pattern2", 0.8),
        ):
            experiment.results.add_evaluation(
                [],
                EvaluationResult(
                    pattern_name=name,
                    collection="collection",
                    indexing_params={},
                    rag_params={},
                    scores={"metrics": [], "question_scores": []},
                    execution_time=0.0,
                    final_score=score,
                ),
            )

        publish = mocker.patch.object(experiment, "_stream_finished_pattern")
        experiment._publish_best_warm_start_pattern()

        publish.assert_not_called()

    def test_marks_warm_start_pattern_names(self):
        """Warm-start candidate names retain their phase in final artifacts."""
        experiment = _build_experiment()

        experiment._optimization_phase = "warm_start"
        assert experiment._create_pattern_name() == "Pattern1-warm-start"
        experiment._optimization_phase = "gam"
        assert experiment._create_pattern_name() == "Pattern1"

        experiment.results.add_evaluation(
            [],
            EvaluationResult(
                pattern_name="Pattern1-warm-start",
                collection="collection",
                indexing_params={},
                rag_params={},
                scores={"metrics": [], "question_scores": []},
                execution_time=0.0,
                final_score=0.5,
            ),
        )
        assert experiment._create_pattern_name() == "Pattern2"


class TestMultiModelGAMExperiment:
    """GAM experiment coverage for multiple foundation and embedding models."""

    def test_runs_with_three_embedding_models_and_two_foundation_models(self, mocker):
        """A failed model does not prevent balanced warm-start coverage of responding models."""
        from ai4rag.core.experiment.exception_handler import GenerationError
        from ai4rag.core.experiment.experiment import AI4RAGExperiment

        foundation_models = [MagicMock(model_id=f"llm-{index}") for index in range(2)]
        embedding_models = [MagicMock(model_id=f"embedding-{index}", params={}) for index in range(3)]
        search_space = AI4RAGSearchSpace(
            params=[
                Parameter(name=AI4RAGParamNames.FOUNDATION_MODEL, values=foundation_models),
                Parameter(name=AI4RAGParamNames.EMBEDDING_MODEL, values=embedding_models),
            ]
        )
        experiment = AI4RAGExperiment(
            documents=[],
            benchmark_data=_BENCHMARK_DF,
            search_space=search_space,
            vector_store_config=MilvusLiteConfig(db_path="./ai4rag.db"),
            optimizer_settings=GAMOptSettings(
                max_evals=15,
                max_iterations=7,
                n_random_nodes=6,
                warm_start_strategy="balanced",
                fields_to_balance=[
                    AI4RAGParamNames.FOUNDATION_MODEL,
                    AI4RAGParamNames.EMBEDDING_MODEL,
                ],
            ),
            event_handler=MagicMock(),
            n_mps_foundation_models=2,
            n_mps_embedding_models=3,
        )
        mock_gam = MagicMock()
        mock_gam.predict.side_effect = lambda values: [0.5] * len(values)
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)
        failed_models = []
        evaluated_phases = []

        def evaluate_pattern(params, **_):
            evaluated_phases.append(experiment._optimization_phase)
            foundation_model = params[AI4RAGParamNames.FOUNDATION_MODEL]
            if foundation_model.model_id == "llm-0":
                failed_models.append(foundation_model.model_id)
                raise GenerationError(ConnectionError("model is unavailable"), foundation_model.model_id)
            return 0.5

        evaluate = mocker.patch.object(experiment, "run_single_evaluation", side_effect=evaluate_pattern)

        experiment.search()

        assert evaluate.call_count == 15
        evaluated_pairs = {
            (
                call.args[0][AI4RAGParamNames.FOUNDATION_MODEL].model_id,
                call.args[0][AI4RAGParamNames.EMBEDDING_MODEL].model_id,
            )
            for call in evaluate.call_args_list[:6]
        }
        assert evaluated_pairs == {
            (foundation_model.model_id, embedding_model.model_id)
            for foundation_model in foundation_models
            for embedding_model in embedding_models
        }
        assert failed_models
        assert set(failed_models) == {"llm-0"}
        assert any(
            call.args[0][AI4RAGParamNames.FOUNDATION_MODEL].model_id == "llm-1" for call in evaluate.call_args_list
        )
        pattern_rows = []
        warm_start_count = 0
        gam_count = 0
        for call, phase in zip(evaluate.call_args_list, evaluated_phases):
            if phase == "warm_start":
                warm_start_count += 1
                pattern_name = f"Pattern{warm_start_count}-warm-start"
            else:
                gam_count += 1
                pattern_name = f"Pattern{gam_count + 1}"
            pattern_rows.append(
                {
                    "pattern_name": pattern_name,
                    "foundation_model": call.args[0][AI4RAGParamNames.FOUNDATION_MODEL].model_id,
                    "embedding_model": call.args[0][AI4RAGParamNames.EMBEDDING_MODEL].model_id,
                    "score": (None if call.args[0][AI4RAGParamNames.FOUNDATION_MODEL].model_id == "llm-0" else 0.5),
                }
            )
        patterns = pd.DataFrame(pattern_rows)
        print(f"\nEvaluated patterns:\n{patterns.to_string(index=False)}")


class TestResolveOptimizationScore:
    """Selecting the optimization metric's score from a pattern's results."""

    @staticmethod
    def _scores(*metrics) -> EvaluationMetricsResult:
        """Build a result-scores dict from ``(name, evaluator, mean)`` tuples."""
        return EvaluationMetricsResult(
            metrics=[
                AggregateMetric(
                    name=name,
                    evaluator=evaluator,
                    description="",
                    scores=ConfidenceInterval(mean=mean, ci_low=None, ci_high=None),
                )
                for name, evaluator, mean in metrics
            ],
            question_scores=[],
        )

    def test_returns_mean_of_matching_metric(self):
        experiment = _build_experiment(optimization_metric=Metrics.FAITHFULNESS)
        scores = self._scores(("faithfulness", "unitxt", 0.8))

        assert experiment._resolve_optimization_score(scores, "pattern_1") == 0.8

    def test_disambiguates_colliding_names_by_evaluator(self):
        """Both unitxt and ragas emit 'faithfulness'; the unitxt one must be chosen."""
        experiment = _build_experiment(optimization_metric=Metrics.FAITHFULNESS)
        scores = self._scores(("faithfulness", "ragas", 0.2), ("faithfulness", "unitxt", 0.9))

        assert experiment._resolve_optimization_score(scores, "pattern_1") == 0.9

    def test_none_mean_returns_none_not_error(self):
        """A produced-but-unscored metric is a failed iteration, not a fatal error."""
        experiment = _build_experiment(optimization_metric=Metrics.FAITHFULNESS)
        scores = self._scores(("faithfulness", "unitxt", None))

        assert experiment._resolve_optimization_score(scores, "pattern_1") is None

    def test_absent_metric_raises(self):
        """A metric that is not produced at all is a configuration error."""
        from ai4rag.core.experiment.utils import RAGExperimentError

        experiment = _build_experiment(optimization_metric=Metrics.FAITHFULNESS)
        scores = self._scores(("answer_correctness", "unitxt", 0.7))

        with pytest.raises(RAGExperimentError, match="not found in evaluation results"):
            experiment._resolve_optimization_score(scores, "pattern_1")

    def test_wrong_evaluator_only_raises(self):
        """A matching name under a different evaluator does not satisfy the target."""
        from ai4rag.core.experiment.utils import RAGExperimentError

        experiment = _build_experiment(optimization_metric=Metrics.FAITHFULNESS)
        scores = self._scores(("faithfulness", "ragas", 0.5))

        with pytest.raises(RAGExperimentError, match="not found in evaluation results"):
            experiment._resolve_optimization_score(scores, "pattern_1")


class TestWarmStartMetrics:
    @pytest.mark.parametrize(
        "optimization_metric, expected_metrics",
        [
            (Metrics.FAITHFULNESS, {"unitxt": [Metrics.FAITHFULNESS]}),
            (Metrics.RAGAS_FAITHFULNESS, {"ragas": [Metrics.RAGAS_FAITHFULNESS]}),
            (Metrics.RAGAS_CONTEXT_RECALL, {"ragas": [Metrics.RAGAS_CONTEXT_RECALL]}),
            (
                Metrics.OVERALL_SCORE,
                {
                    "unitxt": [Metrics.ANSWER_CORRECTNESS, Metrics.FAITHFULNESS, Metrics.CONTEXT_CORRECTNESS],
                    "ragas": [
                        Metrics.RAGAS_FAITHFULNESS,
                        Metrics.RAGAS_ANSWER_RELEVANCY,
                        Metrics.RAGAS_CONTEXT_PRECISION,
                        Metrics.RAGAS_CONTEXT_RECALL,
                    ],
                },
            ),
        ],
    )
    def test_only_objective_metrics_are_evaluated(self, mocker, optimization_metric, expected_metrics):
        evaluators = []
        for evaluator_type in ("unitxt", "ragas"):
            evaluator = MagicMock(spec=BaseEvaluator)
            evaluator.EVALUATOR_TYPE = evaluator_type
            evaluator.evaluate_metrics.side_effect = lambda evaluation_data, metrics: merge_evaluation_results(
                [
                    _make_result(
                        metric.name,
                        metric.evaluator,
                        0.2 if metric.evaluator == "ragas" else 0.8,
                        {"q1": 0.2 if metric.evaluator == "ragas" else 0.8},
                    )
                    for metric in metrics
                ]
            )
            evaluators.append(evaluator)

        experiment = _build_experiment(evaluators=evaluators, optimization_metric=optimization_metric)
        mocker.patch("ai4rag.core.experiment.experiment.build_evaluation_data", return_value=[])
        experiment._optimization_phase = "warm_start"

        scores, _ = experiment._evaluate_response([], "Pattern1-warm-start")

        for evaluator in evaluators:
            if evaluator.EVALUATOR_TYPE in expected_metrics:
                assert (
                    evaluator.evaluate_metrics.call_args.kwargs["metrics"] == expected_metrics[evaluator.EVALUATOR_TYPE]
                )
            else:
                evaluator.evaluate_metrics.assert_not_called()
        expected_score = (
            0.4571
            if optimization_metric == Metrics.OVERALL_SCORE
            else (0.2 if optimization_metric.evaluator == "ragas" else 0.8)
        )
        assert experiment._resolve_optimization_score(scores, "Pattern1-warm-start") == expected_score
        if optimization_metric == Metrics.OVERALL_SCORE:
            assert len(scores["metrics"]) == 8
            assert scores["question_scores"][0]["metrics"][-1]["value"] == expected_score
        else:
            assert [(metric["evaluator"], metric["name"]) for metric in scores["metrics"]] == [
                (optimization_metric.evaluator, optimization_metric.name)
            ]

        experiment._optimization_phase = "gam"
        experiment._evaluate_response([], "Pattern2")
        assert [len(evaluator.evaluate_metrics.call_args.kwargs["metrics"]) for evaluator in evaluators] == [3, 4]

    def test_published_best_warm_start_has_full_metrics(self, mocker):
        experiment = _build_experiment(metrics=[Metrics.FAITHFULNESS, Metrics.ANSWER_CORRECTNESS])
        evaluation_data = [MagicMock()]
        partial_scores = _make_result("faithfulness", "unitxt", 0.9, {"q1": 0.9})
        experiment.results.add_evaluation(
            evaluation_data,
            EvaluationResult(
                pattern_name="Pattern1-warm-start",
                collection="collection",
                indexing_params={},
                rag_params={},
                scores=partial_scores,
                execution_time=1.0,
                final_score=0.9,
            ),
        )
        remaining_scores = _make_result("answer_correctness", "unitxt", 0.5, {"q1": 0.5})
        evaluate = mocker.patch.object(experiment, "_evaluate_data", return_value=remaining_scores)
        publish = mocker.patch.object(experiment, "_stream_finished_pattern")

        experiment._publish_best_warm_start_pattern()

        evaluate.assert_called_once_with(evaluation_data, [Metrics.ANSWER_CORRECTNESS])
        assert [metric["name"] for metric in experiment.results.evaluations[0].scores["metrics"]] == [
            "faithfulness",
            "answer_correctness",
        ]
        assert experiment.results.evaluations[0].final_score == 0.9
        assert publish.call_args.kwargs["evaluation_result"].scores == experiment.results.evaluations[0].scores

    @pytest.mark.parametrize("error", [EvaluationError(RuntimeError("unavailable")), RuntimeError("unavailable")])
    def test_failed_completion_publishes_objective_score(self, mocker, caplog, error):
        experiment = _build_experiment(metrics=[Metrics.FAITHFULNESS, Metrics.RAGAS_FAITHFULNESS])
        partial_scores = _make_result("faithfulness", "unitxt", 0.9, {"q1": 0.9})
        result = EvaluationResult(
            pattern_name="Pattern1-warm-start",
            collection="collection",
            indexing_params={},
            rag_params={},
            scores=partial_scores,
            execution_time=1.0,
            final_score=0.9,
        )
        experiment.results.add_evaluation([MagicMock()], result)
        evaluate = mocker.patch.object(experiment, "_evaluate_data", side_effect=error)
        publish = mocker.patch.object(experiment, "_stream_finished_pattern")

        experiment._publish_best_warm_start_pattern()

        evaluate.assert_called_once_with(experiment.results.evaluation_data[0], [Metrics.RAGAS_FAITHFULNESS])
        assert experiment.results.evaluations[0] is result
        assert result.scores is partial_scores
        assert result.final_score == 0.9
        assert publish.call_args.kwargs["evaluation_result"] is result
        assert publish.call_args.kwargs["pattern_name"] == "Pattern1"
        assert "Unable to complete non-objective metrics" in caplog.text


class TestMergeEvaluationResults:
    def test_empty_list_returns_empty_result(self):
        merged = merge_evaluation_results([])
        assert merged["metrics"] == []
        assert merged["question_scores"] == []

    def test_single_result_returned_as_is(self):
        r = _make_result("faithfulness", "unitxt", 0.8, {"q1": 0.9, "q2": 0.7})
        merged = merge_evaluation_results([r])
        assert merged is r

    def test_two_results_merged(self):
        r1 = _make_result("faithfulness", "unitxt", 0.8, {"q1": 0.9, "q2": 0.7})
        r2 = _make_result("answer_relevance", "judge", 0.6, {"q1": 0.5, "q2": 0.7})
        merged = merge_evaluation_results([r1, r2])

        assert len(merged["metrics"]) == 2
        metric_names = {m["name"] for m in merged["metrics"]}
        assert metric_names == {"faithfulness", "answer_relevance"}

        q_scores = {qs["question_id"]: qs for qs in merged["question_scores"]}
        assert len(q_scores["q1"]["metrics"]) == 2
        assert len(q_scores["q2"]["metrics"]) == 2

    def test_question_ids_preserved(self):
        r1 = _make_result("f", "unitxt", 0.5, {"q1": 0.5, "q2": 0.5, "q3": 0.5})
        r2 = _make_result("a", "judge", 0.5, {"q1": 0.5, "q2": 0.5, "q3": 0.5})
        merged = merge_evaluation_results([r1, r2])
        qids = [qs["question_id"] for qs in merged["question_scores"]]
        assert qids == ["q1", "q2", "q3"]
