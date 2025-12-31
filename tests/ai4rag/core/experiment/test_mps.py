import pytest
import pandas as pd

from ai4rag.core.experiment.mps import (
    ModelsPreSelector,
    PreSelectorError,
    Document,
    GenerationError,
    BenchmarkData,
    FoundationModel,
    EmbeddingModel,
    ChromaVectorStore,
)


@pytest.fixture
def benchmark_data() -> BenchmarkData:
    benchmark_data = BenchmarkData(
        benchmark_data=pd.DataFrame(
            {
                "question": ["Question 1", "Questions 2"],
                "correct_answer": ["Answer 1", "Answer 2"],
                "correct_answer_document_ids": [["id_1_1"], ["id_2_1"]],
            }
        )
    )

    return benchmark_data


@pytest.fixture
def documents() -> list[Document]:
    docs = [
        Document(
            page_content="Page content 1",
            metadata={"document_id": "id_1_1"},
        ),
        Document(
            page_content="Page content 2",
            metadata={"document_id": "id_2_1"},
        ),
    ]
    return docs


@pytest.fixture
def foundation_models(mocker):
    fm_list = [mocker.MagicMock(spec=FoundationModel, model_id=f"foundation_model_{idx}") for idx in range(4)]
    return fm_list


@pytest.fixture
def embedding_models(mocker):
    em_list = [mocker.MagicMock(spec=EmbeddingModel, model_id=f"embedding_model_{idx}") for idx in range(3)]
    return em_list


@pytest.fixture
def pre_selector_evaluation_results(embedding_models, foundation_models) -> list[dict]:
    results = []

    score = 0.05

    for fm in foundation_models:
        for em in embedding_models:
            results.append(
                {
                    "embedding_model": em,
                    "foundation_model": fm,
                    "scores": {"answer_correctness": {"mean": score}},
                    "question_scores": {
                        "answer_correctness": {
                            "q0": score,
                            "q1": score,
                            "q2": score,
                            "q3": score,
                            "q4": score,
                        }
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
        embedding_model_id="fake_embedding_model_id",
        documents=documents,
        benchmark_data=benchmark_data,
        foundation_models=foundation_models,
        embedding_models=embedding_models,
        metric="answer_correctness",
    )
    pre_selector.evaluation_results = pre_selector_evaluation_results

    return pre_selector


@pytest.fixture
def fully_mocked_selector(
    mocker, documents, benchmark_data, embedding_models, foundation_models
) -> ModelsPreSelector:
    mocker.patch("ai4rag.core.experiment.mps.ChromaVectorStore", autospec=True)

    def side_effect(**kwargs):
        questions = kwargs.pop("questions")
        res = []
        for question in questions:
            res.append({"question": question, "answer": question[::-1], "reference_documents": []})
        return res

    mocker.patch("ai4rag.core.experiment.mps.query_inference_service", side_effect=side_effect)

    selector = ModelsPreSelector(
        agent="sequential",
        embedding_model_id="embedding_model_id",
        benchmark_data=benchmark_data,
        documents=documents,
        foundation_models=foundation_models,
        embedding_models=embedding_models,
        metric="answer_correctness",
    )

    return selector


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
                    f"Starting pre-evaluation of foundation model: {fm.model_id} and embedding model: {em.model_id}" in caplog.text
                ), f"There are no proper pre-selection logs for {(em, fm)}"

    def test_evaluate_patterns_with_errors(self, mocker, fully_mocked_selector, caplog):
        gen_exc = GenerationError(exception=ValueError("Dummy val error"), model_id="some-inference-model")

        mocker.patch("ai4rag.core.experiment.mps.query_inference_service", side_effect=gen_exc)

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
        document = mocker.MagicMock(Document)
        val_err = ValueError("Fake embeddings error")
        vs.add_documents.side_effect = val_err
        mocker.patch("ai4rag.core.experiment.mps.ChromaVectorStore", return_value=vs)
        mocked_em = mocker.MagicMock(EmbeddingModel)
        mocked_em.model_id = "embedding_model_id"

        with pytest.raises(PreSelectorError) as err:
            fully_mocked_selector._create_vector_store(embedding_model=mocked_em, chunked_documents=[document])

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
        models = pre_selector.select_models(n_em=n_em, n_fm=n_fm)
        assert len(models.get("foundation_models")) == n_fm
        assert len(models.get("embedding_models")) == n_em
        assert f"Selecting the best {n_em} embedding models and {n_fm} foundation models." in caplog.text
