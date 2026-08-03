# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pandas as pd
import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from ai4rag.core.experiment.mps import (
    AI4RAGChunk,
    BaseEmbeddingModel,
    BaseFoundationModel,
    BenchmarkData,
    ChromaVectorStore,
    GenerationError,
    ModelsPreSelector,
    PreSelectorError,
)
from ai4rag.evaluator.metric import Metrics


@pytest.fixture
def benchmark_data() -> BenchmarkData:
    benchmark_data = BenchmarkData(
        benchmark_data=pd.DataFrame(
            {
                "question": ["Question 1", "Questions 2"],
                "correct_answers": [["Answer 1"], ["Answer 2"]],
                "correct_answer_document_ids": [["id_1_1"], ["id_2_1"]],
            }
        )
    )

    return benchmark_data


def _make_docling_doc(name: str, text: str) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
    return doc


@pytest.fixture
def documents() -> list[DoclingDocument]:
    return [
        _make_docling_doc("id_1_1", "Page content 1"),
        _make_docling_doc("id_2_1", "Page content 2"),
    ]


@pytest.fixture
def foundation_models(mocker):
    fm_list = [mocker.MagicMock(spec=BaseFoundationModel, model_id=f"foundation_model_{idx}") for idx in range(4)]
    return fm_list


@pytest.fixture
def embedding_models(mocker):
    em_list = [mocker.MagicMock(spec=BaseEmbeddingModel, model_id=f"embedding_model_{idx}") for idx in range(3)]
    return em_list


@pytest.fixture
def pre_selector_evaluation_results(embedding_models, foundation_models) -> list[dict]:
    results = []

    score = 0.05
    unitxt_metric_names = ("answer_correctness", "faithfulness", "context_correctness")

    for fm in foundation_models:
        for em in embedding_models:
            aggregate_metrics = [
                {
                    "name": name,
                    "evaluator": "unitxt",
                    "description": "",
                    "scores": {"mean": score, "ci_low": None, "ci_high": None},
                }
                for name in unitxt_metric_names
            ]
            overall_mean = round(score, 4)
            aggregate_metrics.append(
                {
                    "name": "overall_score",
                    "evaluator": "custom",
                    "description": "",
                    "scores": {"mean": overall_mean, "ci_low": None, "ci_high": None},
                }
            )
            question_scores = [
                {
                    "question_id": f"q{i}",
                    "metrics": [
                        *[{"name": name, "evaluator": "unitxt", "value": score} for name in unitxt_metric_names],
                        {"name": "overall_score", "evaluator": "custom", "value": overall_mean},
                    ],
                }
                for i in range(5)
            ]
            results.append(
                {
                    "embedding_model": em,
                    "foundation_model": fm,
                    "evaluation": {
                        "metrics": aggregate_metrics,
                        "question_scores": question_scores,
                    },
                }
            )
            score += 0.05

    return results


@pytest.fixture
def pre_selector(
    documents, benchmark_data, embedding_models, foundation_models, pre_selector_evaluation_results
) -> ModelsPreSelector:

    pre_selector = ModelsPreSelector(
        documents=documents,
        benchmark_data=benchmark_data,
        foundation_models=foundation_models,
        embedding_models=embedding_models,
    )
    pre_selector.evaluation_results = pre_selector_evaluation_results

    return pre_selector


def _make_evaluate_metrics_result(evaluation_data, metrics):
    """Build a minimal EvaluationMetricsResult for each requested metric."""
    question_ids = [ed.question_id or str(i) for i, ed in enumerate(evaluation_data)]
    score = 0.5
    return {
        "metrics": [
            {
                "name": m.name,
                "evaluator": m.evaluator,
                "description": m.description,
                "scores": {"mean": score, "ci_low": None, "ci_high": None},
            }
            for m in metrics
            if m.name in ("answer_correctness", "faithfulness", "context_correctness")
        ],
        "question_scores": [
            {
                "question_id": qid,
                "metrics": [
                    {"name": m.name, "evaluator": m.evaluator, "value": score}
                    for m in metrics
                    if m.name in ("answer_correctness", "faithfulness", "context_correctness")
                ],
            }
            for qid in question_ids
        ],
    }


@pytest.fixture
def fully_mocked_selector(mocker, documents, benchmark_data, embedding_models, foundation_models) -> ModelsPreSelector:
    mocker.patch("ai4rag.core.experiment.mps.ChromaVectorStore", autospec=True)

    def side_effect(**kwargs):
        questions = kwargs.pop("questions")
        res = []
        for question in questions:
            res.append({"question": question, "answer": question[::-1], "reference_documents": []})
        return res

    mocker.patch("ai4rag.core.experiment.mps.query_rag", side_effect=side_effect)

    selector = ModelsPreSelector(
        benchmark_data=benchmark_data,
        documents=documents,
        foundation_models=foundation_models,
        embedding_models=embedding_models,
    )
    mocker.patch.object(selector.evaluator, "evaluate_metrics", side_effect=_make_evaluate_metrics_result)

    return selector


class TestModelsPreSelectorInit:
    def test_default_search_mode(self, documents, benchmark_data, embedding_models, foundation_models):
        """Test that search_mode defaults to 'vector'."""
        selector = ModelsPreSelector(
            benchmark_data=benchmark_data,
            documents=documents,
            foundation_models=foundation_models,
            embedding_models=embedding_models,
        )

        assert selector.retrieval_params["search_mode"] == "vector"

    def test_default_metric_is_overall_score(self, documents, benchmark_data, embedding_models, foundation_models):
        """Test that the default optimization metric is OVERALL_SCORE."""
        selector = ModelsPreSelector(
            benchmark_data=benchmark_data,
            documents=documents,
            foundation_models=foundation_models,
            embedding_models=embedding_models,
        )

        assert selector.metric == Metrics.OVERALL_SCORE

    def test_custom_search_mode(self, documents, benchmark_data, embedding_models, foundation_models):
        """Test that search_mode can be set via kwargs."""
        selector = ModelsPreSelector(
            benchmark_data=benchmark_data,
            documents=documents,
            foundation_models=foundation_models,
            embedding_models=embedding_models,
            search_mode="hybrid",
        )

        assert selector.retrieval_params["search_mode"] == "hybrid"

    def test_search_mode_independent_of_retrieval_method(
        self, documents, benchmark_data, embedding_models, foundation_models
    ):
        """Test that search_mode is read from its own key, not retrieval_method."""
        selector = ModelsPreSelector(
            benchmark_data=benchmark_data,
            documents=documents,
            foundation_models=foundation_models,
            embedding_models=embedding_models,
            retrieval_method="window",
        )

        assert selector.retrieval_params["search_mode"] == "vector"
        assert selector.retrieval_params["method"] == "window"


class TestModelsPreSelector:
    def test_evaluate_patterns(self, fully_mocked_selector, caplog):

        fully_mocked_selector.evaluate_patterns()

        evaluated_fms = [e["foundation_model"] for e in fully_mocked_selector.evaluation_results]
        evaluated_ems = [e["embedding_model"] for e in fully_mocked_selector.evaluation_results]

        for fm in fully_mocked_selector.foundation_models:
            for em in fully_mocked_selector.embedding_models:
                assert em in evaluated_ems, f"{em.model_id} not in {evaluated_ems}"
                assert fm in evaluated_fms, f"{fm.model_id} not in {evaluated_fms}"
                assert (
                    f"Starting pre-evaluation of foundation model: {fm.model_id} and embedding model: {em.model_id}"
                    in caplog.text
                ), f"There are no proper pre-selection logs for {(em, fm)}"

    def test_evaluate_patterns_with_errors(self, mocker, fully_mocked_selector, caplog):
        gen_exc = GenerationError(exception=ValueError("Dummy val error"), model_id="some-inference-model")

        mocker.patch("ai4rag.core.experiment.mps.query_rag", side_effect=gen_exc)

        with pytest.raises(PreSelectorError) as err:
            fully_mocked_selector.evaluate_patterns()

        msg = "Foundation models pre-selection has failed. None of the given models has been successfully evaluated. "
        assert msg in str(err.value), "Proper message was not raised when all models failed."
        for model in fully_mocked_selector.foundation_models:
            expected_log = f"Pre-evaluation of '{model.model_id}' has failed."
            assert expected_log in caplog.text

    def test_evaluate_pattern_with_failing_embedding(self, mocker, fully_mocked_selector, caplog):
        vs = mocker.MagicMock(ChromaVectorStore)
        val_err = ValueError("Fake error in embeddings")
        vs.add_documents.side_effect = val_err
        mocker.patch("ai4rag.core.experiment.mps.ChromaVectorStore", return_value=vs)

        with pytest.raises(PreSelectorError) as err:
            fully_mocked_selector.evaluate_patterns()

        expected_msg = (
            "Foundation models pre-selection has failed. None of the given models has been successfully evaluated. "
        )

        assert expected_msg in str(err.value)

    def test_create_vector_store(self, mocker, fully_mocked_selector, caplog):
        vs = mocker.MagicMock(ChromaVectorStore)
        document = mocker.MagicMock(AI4RAGChunk)
        val_err = ValueError("Fake embeddings error")
        vs.add_documents.side_effect = val_err
        mocker.patch("ai4rag.core.experiment.mps.ChromaVectorStore", return_value=vs)
        mocked_em = mocker.MagicMock(BaseEmbeddingModel)
        mocked_em.model_id = "embedding_model_id"

        with pytest.raises(PreSelectorError) as err:
            fully_mocked_selector._create_vector_store(
                embedding_model=mocked_em, chunked_documents=[document], collection_name="ai4rag_mps_collection_1"
            )

        exp_msg = f"Failed to create in-memory vector index due to: {repr(val_err)}."
        assert exp_msg in caplog.text, "Warning after first embedding fail was not logged"
        assert str(err.value) == exp_msg

    def test_mean_based_scoring(self, pre_selector):
        top_models_with_scores = pre_selector._mean_based_scoring()
        scores = [r.get("score") for r in top_models_with_scores]
        sorted_scores = sorted(scores, reverse=True)

        assert scores == sorted_scores, "Scores were not sorted correctly."

    def test_select_models(self, pre_selector, caplog):
        n_em = 2
        n_fm = 3
        models = pre_selector.select_models(n_embedding_models=n_em, n_foundation_models=n_fm)
        assert len(models.get("foundation_models")) == n_fm
        assert len(models.get("embedding_models")) == n_em
        assert "Starting models pre-selection..." in caplog.text
        assert f"Selected the best {n_em} embedding model(s) and {n_fm} foundation model(s)." in caplog.text
