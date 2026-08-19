# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional tests for the RAGAS evaluator.

Two levels of coverage that the unit tests deliberately skip:

1. ``TestRagasEvaluatorRealRagasPipeline`` drives the *real* ``ragas.evaluate``
   machinery through our :mod:`ai4rag.evaluator.ragas_adapters` wrappers, backed
   by local fake models (no network).  This is what catches RAGAS version drift:
   changes to the ``EvaluationDataset``/``SingleTurnSample`` schema, the
   ``evaluate()`` signature, the result-DataFrame column contract, or the
   ``BaseRagasLLM``/``BaseRagasEmbeddings`` interfaces our adapters implement.
   Exact scores are intentionally not asserted — matching every metric's prompt
   JSON schema would couple the test to RAGAS internals; instead we assert the
   pipeline runs end-to-end, delegates to our models, and returns well-formed
   results in the unit range.

2. ``TestRagasEvaluatorInExperiment`` wires a ``RagasEvaluator`` into a full
   :class:`AI4RAGExperiment` run (real Chroma, mocked search-space models) with
   only the RAGAS scoring step stubbed, verifying that RAGAS metrics are routed
   to the evaluator and land in the experiment results.
"""

import importlib.util

import pandas as pd
import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.random_opt import RandomOptimizer, RandomOptSettings
from ai4rag.evaluator.base_evaluator import EvaluationData
from ai4rag.evaluator.metric import Metrics
from ai4rag.evaluator.ragas_evaluator import RagasEvaluator
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.rag.vector_store.config import ChromaConfig
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.mocks import MockedEmbeddingModel, MockedFoundationModel

ragas_installed = importlib.util.find_spec("ragas") is not None
requires_ragas = pytest.mark.skipif(not ragas_installed, reason="ragas not installed")

_EMBEDDING_DIMENSION = 64


# ---------------------------------------------------------------------------
# Local fakes: accept RAGAS's chat/embedding call shapes and record delegation.
# ---------------------------------------------------------------------------


class _SpyFoundationModel:
    """Minimal foundation model that satisfies the RAGAS LLM adapter.

    Returns fixed, permissively-shaped JSON for every completion and counts the
    calls, so a test can prove RAGAS drove generation through our adapter.  The
    ``**kwargs`` on :meth:`chat` mirror the real ``OpenAIFoundationModel.chat``
    signature the adapter relies on.
    """

    model_id = "spy-fm"

    def __init__(self) -> None:
        self.chat_calls = 0

    def chat(self, messages, **kwargs):  # pylint: disable=unused-argument
        self.chat_calls += 1

        class _Message:
            # A grab-bag of the fields RAGAS metric prompts look for; unmatched
            # keys are simply ignored by RAGAS, unparseable metrics become NaN.
            content = (
                '{"question": "What is Python?", "noncommittal": 0, '
                '"statements": ["Python is a programming language."]}'
            )

        class _Choice:
            message = _Message()

        return [_Choice()]


class _SpyEmbeddingModel:
    """Minimal embedding model that satisfies the RAGAS embeddings adapter."""

    model_id = "spy-em"

    def __init__(self) -> None:
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text):  # pylint: disable=unused-argument
        self.query_calls += 1
        return [0.1] * 8

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[0.1] * 8 for _ in texts]


@pytest.fixture
def evaluation_data() -> list[EvaluationData]:
    """Two single-turn samples with question, answer, context and reference."""
    return [
        EvaluationData(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
            context_ids=["doc1"],
            ground_truths=["Python is a programming language."],
            ground_truths_context_ids=["doc1"],
            question_id="q1",
        ),
        EvaluationData(
            question="What is a vector store?",
            answer="A vector store indexes embeddings for similarity search.",
            contexts=["A vector store holds embeddings and supports nearest-neighbour search."],
            context_ids=["doc2"],
            ground_truths=["A vector store stores embeddings for similarity search."],
            ground_truths_context_ids=["doc2"],
            question_id="q2",
        ),
    ]


@requires_ragas
class TestRagasEvaluatorRealRagasPipeline:
    """Exercise the genuine ``ragas.evaluate`` path through our adapters."""

    def test_runs_end_to_end_and_delegates_to_models(self, evaluation_data):
        """A real RAGAS run must drive both the LLM and embedding adapters.

        ``answer_relevancy`` needs both a generation model (to produce candidate
        questions) and an embedding model (to score their similarity), so a
        successful run proves both adapters are wired into RAGAS correctly.
        """
        fm = _SpyFoundationModel()
        em = _SpyEmbeddingModel()
        evaluator = RagasEvaluator(model=fm, embedding_model=em, timeout=120, max_workers=1)

        result = evaluator.evaluate_metrics(evaluation_data, [Metrics.RAGAS_ANSWER_RELEVANCY])

        # RAGAS actually invoked our adapters rather than its own client.
        assert fm.chat_calls > 0, "RAGAS did not delegate generation to the foundation-model adapter"
        assert em.query_calls > 0, "RAGAS did not delegate embeddings to the embedding-model adapter"
        assert em.document_calls > 0, "RAGAS did not embed documents through the embedding-model adapter"

        # The result keeps the shared shape and evaluator attribution.
        names = {m["name"]: m for m in result["metrics"]}
        assert set(names) == {"answer_relevancy"}
        assert names["answer_relevancy"]["evaluator"] == "ragas"

    def test_scores_are_none_or_in_unit_range(self, evaluation_data):
        """Every metric mean and per-question value must be ``None`` or in [0, 1]."""
        fm = _SpyFoundationModel()
        em = _SpyEmbeddingModel()
        evaluator = RagasEvaluator(model=fm, embedding_model=em, timeout=120, max_workers=1)

        metrics = [
            Metrics.RAGAS_FAITHFULNESS,
            Metrics.RAGAS_ANSWER_RELEVANCY,
            Metrics.RAGAS_CONTEXT_PRECISION,
            Metrics.RAGAS_CONTEXT_RECALL,
        ]
        result = evaluator.evaluate_metrics(evaluation_data, metrics)

        # All requested metrics are present with the RAGAS attribution.
        assert {m["name"] for m in result["metrics"]} == {m.name for m in metrics}
        for metric in result["metrics"]:
            assert metric["evaluator"] == "ragas"
            mean = metric["scores"]["mean"]
            assert mean is None or 0.0 <= mean <= 1.0, f"mean {mean!r} out of range for {metric['name']}"

        # question_scores is emitted per question, and any value present is valid.
        question_ids = {qs["question_id"] for qs in result["question_scores"]}
        assert question_ids == {"q1", "q2"}
        for qs in result["question_scores"]:
            for metric in qs["metrics"]:
                assert 0.0 <= metric["value"] <= 1.0, f"per-question value {metric['value']!r} out of range"


# ---------------------------------------------------------------------------
# Experiment-level integration: RAGAS metrics routed through the pipeline.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def documents():
    """3 DoclingDocuments long enough to chunk under the default chunk size."""
    paragraph = (
        "This document covers topic {topic}. "
        "The concept of {topic} is central to understanding the broader subject. "
        "Researchers have studied {topic} extensively over many decades. "
        "Practical applications of {topic} span engineering, science, and industry. "
    )
    long_content = (paragraph * 8).strip()
    docs = []
    for i in range(3):
        doc = DoclingDocument(name=f"doc_{i}")
        doc.add_text(label=DocItemLabel.PARAGRAPH, text=long_content.format(topic=f"topic_{i}"))
        docs.append(doc)
    return docs


@pytest.fixture(scope="module")
def benchmark_data():
    """3-record benchmark dataset referencing documents doc_0 through doc_2."""
    return pd.DataFrame(
        {
            "question": [
                "What is topic_0 about?",
                "Describe the applications of topic_1.",
                "What have researchers discovered about topic_2?",
            ],
            "correct_answers": [
                ["topic_0 is central to understanding the broader subject."],
                ["Applications of topic_1 span engineering, science, and industry."],
                ["Researchers have studied topic_2 extensively over many decades."],
            ],
            "correct_answer_document_ids": [["doc_0"], ["doc_1"], ["doc_2"]],
        }
    )


@requires_ragas
class TestRagasEvaluatorInExperiment:
    """RAGAS evaluator wired into a full AI4RAGExperiment run."""

    def test_ragas_metric_flows_into_experiment_results(self, documents, benchmark_data, monkeypatch):
        """A ``RagasEvaluator`` in the evaluator list must score RAGAS metrics.

        The RAGAS scoring call itself is stubbed (``_run_ragas`` returns a fixed
        per-sample DataFrame) so the test needs no live LLM, but everything
        around it is real: metric routing by evaluator type, the evaluator
        contract, result merging, and the experiment loop.
        """
        ragas_evaluator = RagasEvaluator(model=_SpyFoundationModel(), embedding_model=_SpyEmbeddingModel())

        def _fake_run_ragas(dataset, ragas_metrics):  # pylint: disable=unused-argument
            # One row per benchmark question; a constant so the mean is exact.
            return pd.DataFrame({"faithfulness": [1.0, 1.0, 1.0]})

        monkeypatch.setattr(ragas_evaluator, "_run_ragas", _fake_run_ragas)

        foundation_models = [MockedFoundationModel(model_id=f"mock-fm-{i}", params=None) for i in range(2)]
        embedding_models = [
            MockedEmbeddingModel(model_id=f"mock-em-{i}", params={"embedding_dimension": _EMBEDDING_DIMENSION})
            for i in range(2)
        ]
        search_space = AI4RAGSearchSpace(
            vector_store_type="chroma",
            params=[
                Parameter(name="foundation_model", param_type="C", values=foundation_models),
                Parameter(name="embedding_model", param_type="C", values=embedding_models),
            ],
        )

        experiment = AI4RAGExperiment(
            documents=documents,
            benchmark_data=benchmark_data,
            search_space=search_space,
            vector_store_config=ChromaConfig(),
            optimizer_settings=RandomOptSettings(max_evals=2),
            event_handler=LocalEventHandler(),
            evaluators=[UnitxtEvaluator(), ragas_evaluator],
            metrics=(Metrics.RAGAS_FAITHFULNESS,),
            optimization_metric=Metrics.RAGAS_FAITHFULNESS,
        )

        experiment.search(optimizer=RandomOptimizer, skip_mps=True)

        assert len(experiment.results) > 0, "Experiment produced no evaluation results"
        for evaluation in experiment.results:
            ragas_metrics = [
                m for m in evaluation.scores["metrics"] if m["name"] == "faithfulness" and m["evaluator"] == "ragas"
            ]
            assert ragas_metrics, f"RAGAS faithfulness missing from results for {evaluation.pattern_name}"
            assert ragas_metrics[0]["scores"]["mean"] == 1.0
