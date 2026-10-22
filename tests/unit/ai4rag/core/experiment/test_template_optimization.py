# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Candidate templates retain their identity throughout experiment execution."""

from unittest.mock import MagicMock

import httpx
import pytest
from openai import InternalServerError
from pydantic import ValidationError

from ai4rag.core.experiment.exception_handler import IndexingError
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.random_opt import RandomOptimizer, RandomOptSettings
from ai4rag.evaluator.metric import Metrics
from ai4rag.rag.template import AgenticRAG, SequentialRAG
from ai4rag.rag.vector_store.config import MilvusLiteConfig, Neo4jConfig
from ai4rag.rag.vector_store.neo4j import Neo4jGraphRetrievalConfig
from ai4rag.search_space.prepare.input_payload_types import AI4RAGConstraints
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from tests.unit.ai4rag.core.experiment.test_experiment import _BENCHMARK_DF, _make_result


def make_space(provider="milvus_lite", templates=None, count=1):
    models = [MagicMock(model_id=f"fm-{index}") for index in range(count)]
    for model in models:
        model.params.temperature = 0.2
        model.params.max_completion_tokens = 128
    params = [
        Parameter(name="foundation_model", values=models),
        Parameter(name="embedding_model", values=[MagicMock(model_id="em", params={})]),
    ]
    if templates is not None:
        params.append(Parameter(name="template_id", values=templates))
    return AI4RAGSearchSpace(params=params, vector_store_type=provider)


def make_experiment(space, **kwargs):
    config = (
        Neo4jConfig(uri="bolt://localhost:7687", username="neo4j", password="password")
        if space.vector_store_type == "neo4j"
        else MilvusLiteConfig(db_path="./test.db")
    )
    return AI4RAGExperiment(
        documents=[],
        benchmark_data=_BENCHMARK_DF,
        search_space=space,
        vector_store_config=config,
        optimizer_settings=RandomOptSettings(max_evals=2),
        event_handler=MagicMock(),
        optimization_metric=Metrics.FAITHFULNESS,
        **kwargs,
    )


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("milvus", "agentic_rag"),
        ("milvus_lite", "agentic_rag"),
        ("pgvector", "agentic_rag"),
        ("neo4j", "agentic_graph_rag"),
    ],
)
def test_backend_defaults(provider, expected):
    assert make_space(provider)["template_id"].values == (expected,)


@pytest.mark.parametrize("value", [[], ["simple_rag"], ["react_rag"], ["react_graph_rag"], [1], "agentic_rag"])
def test_invalid_payload_templates(value):
    with pytest.raises(ValidationError):
        AI4RAGConstraints(template_ids=value)


@pytest.mark.parametrize("value", [None, ["agentic_rag", "sequential_rag"], ["agentic_graph_rag"]])
def test_payload_templates(value):
    assert AI4RAGConstraints(template_ids=value).template_ids == value


@pytest.mark.parametrize(
    "provider,templates",
    [("neo4j", ("agentic_rag",)), ("milvus", ("agentic_graph_rag",)), ("milvus", ("simple_rag",)), ("milvus", (1,))],
)
def test_incompatible_templates(provider, templates):
    with pytest.raises(ValueError):
        make_space(provider, templates)


@pytest.mark.parametrize("provider,mode", [("milvus", "graph"), ("neo4j", "vector"), ("neo4j", "hybrid")])
def test_incompatible_search_modes_are_rejected_at_construction(provider, mode):
    space = make_space(provider)
    params = [param for param in space.as_list() if param.name != "search_mode"]
    params.append(Parameter(name="search_mode", values=(mode,)))
    with pytest.raises(ValueError, match="requires search modes"):
        AI4RAGSearchSpace(params=params, vector_store_type=provider)


def test_non_categorical_template_parameter_is_rejected_at_construction():
    space = make_space()
    params = [param for param in space.as_list() if param.name != "template_id"]
    params.append(Parameter(name="template_id", param_type="I", v_min=1, v_max=2))
    with pytest.raises(ValueError, match="categorical"):
        AI4RAGSearchSpace(params=params, vector_store_type=space.vector_store_type)


def mock_execution(mocker, experiment):
    store = MagicMock(collection_name="collection")
    mocker.patch("ai4rag.core.experiment.experiment.get_vector_store", return_value=store)
    mocker.patch("ai4rag.core.experiment.experiment.LangChainChunker")
    mocker.patch("ai4rag.core.experiment.experiment.DoclingChunker")
    retriever = mocker.patch("ai4rag.core.experiment.experiment.Retriever")
    query = mocker.patch("ai4rag.core.experiment.experiment.query_rag", return_value=[])
    mocker.patch.object(
        experiment, "_evaluate_response", return_value=(_make_result("faithfulness", "unitxt", 0.7, {}), [])
    )
    mocker.patch.object(AgenticRAG, "__init__", return_value=None)
    mocker.patch.object(SequentialRAG, "__init__", return_value=None)
    return store, retriever, query


def test_templates_have_distinct_cache_entries_and_reuse_index(mocker):
    experiment = make_experiment(make_space(templates=("agentic_rag", "sequential_rag")))
    store, _, query = mock_execution(mocker, experiment)
    candidate = experiment.search_space.combinations[0]
    for template in ("agentic_rag", "sequential_rag", "agentic_rag"):
        assert experiment.run_single_evaluation({**candidate, "template_id": template}) == 0.7
    assert query.call_count == 2
    assert isinstance(query.call_args_list[0].kwargs["rag"], AgenticRAG)
    assert isinstance(query.call_args_list[1].kwargs["rag"], SequentialRAG)
    assert store.add_documents.call_count == 1
    assert [result.rag_params["template_id"] for result in experiment.results] == ["agentic_rag", "sequential_rag"]
    assert [
        call.kwargs["payload"]["template_id"] for call in experiment.event_handler.on_pattern_creation.call_args_list
    ] == ["agentic_rag", "sequential_rag"]


def test_failed_indexing_never_constructs_or_invokes_agent(mocker):
    """An exhausted embedding failure stays an indexing error before generation."""
    experiment = make_experiment(make_space())
    store, retriever, query = mock_execution(mocker, experiment)
    agent_init = mocker.patch.object(AgenticRAG, "__init__", return_value=None)
    failure = InternalServerError(
        "Embedding service unavailable",
        response=httpx.Response(500, request=httpx.Request("POST", "https://maas.example/v1/embeddings")),
        body=None,
    )
    store.add_documents.side_effect = failure

    with pytest.raises(IndexingError) as error:
        experiment.run_single_evaluation(experiment.search_space.combinations[0])

    assert error.value.__cause__ is failure
    agent_init.assert_not_called()
    retriever.assert_not_called()
    query.assert_not_called()


@pytest.mark.parametrize("config", [None, {"route_k": 4, "include_entity_neighbors": False}])
def test_graph_execution_persists_effective_settings(mocker, config):
    experiment = make_experiment(make_space("neo4j"), graph_retrieval_config=config)
    _, retriever, query = mock_execution(mocker, experiment)
    experiment.run_single_evaluation(experiment.search_space.combinations[0])
    expected = Neo4jGraphRetrievalConfig.from_mapping(config or {}).to_search_kwargs()
    assert retriever.call_args.kwargs["search_kwargs"] == expected
    assert isinstance(query.call_args.kwargs["rag"], AgenticRAG)
    payload = experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]
    assert payload["template_id"] == "agentic_graph_rag"
    assert all(payload["settings"]["retrieval"][key] == value for key, value in expected.items())


def test_deferred_publication_uses_stored_template(mocker):
    experiment = make_experiment(make_space(templates=("sequential_rag",)))
    mock_execution(mocker, experiment)
    experiment._optimization_phase = "warm_start"
    experiment.run_single_evaluation(experiment.search_space.combinations[0], publish_pattern=False)
    experiment.event_handler.on_pattern_creation.assert_not_called()
    experiment._publish_best_warm_start_pattern()
    assert experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]["template_id"] == "sequential_rag"


@pytest.mark.parametrize(
    "provider,templates",
    [
        ("milvus_lite", ("sequential_rag",)),
        ("milvus_lite", ("agentic_rag", "sequential_rag")),
        ("neo4j", ("agentic_graph_rag",)),
    ],
)
@pytest.mark.parametrize("skip", [False, True])
def test_preselection_thresholds_apply_to_all_templates(mocker, provider, templates, skip):
    experiment = make_experiment(make_space(provider, templates, count=4))
    selector = mocker.patch("ai4rag.core.experiment.mps.ModelsPreSelector")
    selector.return_value.select_models.return_value = {
        "foundation_models": experiment.search_space["foundation_model"].values[:3],
        "embedding_models": experiment.search_space["embedding_model"].values,
    }
    mocker.patch.object(RandomOptimizer, "search")
    experiment.search(optimizer=RandomOptimizer, skip_mps=skip)
    assert selector.call_count == int(not skip)
    if not skip:
        assert selector.call_args.kwargs["metric"] == Metrics.OVERALL_SCORE
        assert "rag_template" not in selector.call_args.kwargs
    assert experiment.search_space["template_id"].values == templates


def test_valid_known_observation_reaches_gam(mocker):
    from ai4rag.core.hpo.gam_opt import GAMOptimizer, GAMOptSettings

    experiment = make_experiment(make_space(templates=("sequential_rag",)))
    observation = {**experiment.search_space.combinations[0], "score": 0.5}
    experiment.known_observations = [observation]
    experiment.optimizer_settings = GAMOptSettings(max_evals=2)
    loaded = mocker.spy(GAMOptimizer, "_load_known_observations")
    mocker.patch.object(GAMOptimizer, "search")
    experiment.search()
    assert loaded.call_args.args[1] == [observation]
