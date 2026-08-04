# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional tests for AI4RAG experiment runs against a live OGX server."""

import os
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv
from ogx_client import OgxClient

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.rag.vector_store.config import ChromaConfig, MilvusConfig, PGVectorConfig
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.file_store import FileStore
from dev_utils.utils import read_benchmark_from_json

load_dotenv(find_dotenv(".env.local"))


DATA_PATH = os.environ.get("AI4RAG_TEST_DATA_PATH")
OUTPUT_PATH = os.environ.get("AI4RAG_TEST_OUTPUT_PATH")


pytestmark = pytest.mark.skipif(
    DATA_PATH is None,
    reason="AI4RAG_TEST_DATA_PATH environment variable not set",
)


@pytest.fixture(scope="module")
def client():
    return OgxClient(
        base_url=os.environ["OGX_CLIENT_BASE_URL"],
        api_key=os.environ["OGX_CLIENT_API_KEY"],
    )


@pytest.fixture(scope="module")
def documents():
    documents_path = Path(os.path.join(DATA_PATH, "documents"))
    file_store = FileStore(documents_path)
    return file_store.load_as_documents()


@pytest.fixture(scope="module")
def benchmark_data():
    benchmark_data_path = Path(os.path.join(DATA_PATH, "benchmark_data.json"))
    return read_benchmark_from_json(benchmark_data_path)


@pytest.fixture(scope="module")
def foundation_model(client):
    model_id = os.environ.get("AI4RAG_TEST_FOUNDATION_MODEL", "vllm-inference-llama-3-1/redhataillama-31-8b-instruct")
    return OGXFoundationModel(model_id=model_id, client=client)


@pytest.fixture(scope="module")
def embedding_model(client):
    model_id = os.environ.get("AI4RAG_TEST_EMBEDDING_MODEL", "vllm-embedding/granite-278m-multilingual-1")
    dimension = int(os.environ.get("AI4RAG_TEST_EMBEDDING_DIMENSION", "768"))
    context_length = int(os.environ.get("AI4RAG_TEST_EMBEDDING_CONTEXT_LENGTH", "512"))
    return OGXEmbeddingModel(
        model_id=model_id,
        client=client,
        params={"embedding_dimension": dimension, "context_length": context_length},
    )


def _make_event_handler(test_name):
    if OUTPUT_PATH:
        return LocalEventHandler(output_path=os.path.join(OUTPUT_PATH, test_name))
    return LocalEventHandler()


class TestExperimentChroma:
    """Run experiment with chroma vector store and OGX models."""

    def test_experiment_chroma_ogx_models(self, client, documents, benchmark_data, foundation_model, embedding_model):
        search_space = AI4RAGSearchSpace(
            vector_store_type="chroma",
            params=[
                Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
                Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
            ],
        )

        optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=3)

        experiment = AI4RAGExperiment(
            client=client,
            documents=documents,
            benchmark_data=benchmark_data,
            search_space=search_space,
            optimizer_settings=optimizer_settings,
            event_handler=_make_event_handler("chroma_ogx_models"),
            vector_store_config=ChromaConfig(),
        )

        experiment.search(skip_mps=True)

        assert len(experiment.results) > 0

        best_evals = experiment.results.get_best_evaluations(k=1)
        assert len(best_evals) == 1

        best_eval = best_evals[0]
        assert best_eval.final_score is not None
        assert 0 <= best_eval.final_score <= 1


class TestExperimentMilvus:
    """Run experiment with a direct Milvus vector store client and OGX models."""

    def test_experiment_milvus_ogx_models(self, client, documents, benchmark_data, foundation_model, embedding_model):
        search_space = AI4RAGSearchSpace(
            vector_store_type="milvus",
            params=[
                Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
                Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
            ],
        )

        optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=3)

        experiment = AI4RAGExperiment(
            client=client,
            documents=documents,
            benchmark_data=benchmark_data,
            search_space=search_space,
            optimizer_settings=optimizer_settings,
            event_handler=_make_event_handler("milvus_ogx_models"),
            vector_store_config=MilvusConfig.from_env(),
        )

        experiment.search(skip_mps=True)

        assert len(experiment.results) > 0

        best_evals = experiment.results.get_best_evaluations(k=1)
        assert len(best_evals) == 1

        best_eval = best_evals[0]
        assert best_eval.final_score is not None
        assert 0 <= best_eval.final_score <= 1


class TestExperimentPGVector:
    """Run experiment with PG vector store and OGX models."""

    def test_experiment_pgvector_ogx_models(self, client, documents, benchmark_data, foundation_model, embedding_model):
        search_space = AI4RAGSearchSpace(
            vector_store_type="pgvector",
            params=[
                Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
                Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
            ],
        )

        optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=3)

        experiment = AI4RAGExperiment(
            client=client,
            documents=documents,
            benchmark_data=benchmark_data,
            search_space=search_space,
            optimizer_settings=optimizer_settings,
            event_handler=_make_event_handler("pgvector_ogx_models"),
            vector_store_config=PGVectorConfig.from_env(),
        )

        experiment.search(skip_mps=True)

        assert len(experiment.results) > 0

        best_evals = experiment.results.get_best_evaluations(k=1)
        assert len(best_evals) == 1

        best_eval = best_evals[0]
        assert best_eval.final_score is not None
        assert 0 <= best_eval.final_score <= 1


class TestExperimentChromaWithKnownObservations:
    """Run experiment with chroma, OGX models, and known observations."""

    def test_experiment_chroma_known_observations(
        self, client, documents, benchmark_data, foundation_model, embedding_model
    ):
        known_observations = [
            {
                "foundation_model": foundation_model,
                "embedding_model": embedding_model,
                "chunk_size": 1024,
                "chunk_overlap": 128,
                "chunking_method": "recursive",
                "retrieval_method": "simple",
                "search_mode": "vector",
                "window_size": 0,
                "number_of_chunks": 3,
                "score": 0.4,
            },
            {
                "foundation_model": foundation_model,
                "embedding_model": embedding_model,
                "chunk_size": 2048,
                "chunk_overlap": 128,
                "chunking_method": "recursive",
                "retrieval_method": "simple",
                "search_mode": "vector",
                "window_size": 0,
                "number_of_chunks": 3,
                "score": 0.5,
            },
            {
                "foundation_model": foundation_model,
                "embedding_model": embedding_model,
                "chunk_size": 2048,
                "chunk_overlap": 256,
                "chunking_method": "recursive",
                "retrieval_method": "simple",
                "search_mode": "vector",
                "window_size": 0,
                "number_of_chunks": 3,
                "score": 0.6,
            },
        ]

        search_space = AI4RAGSearchSpace(
            vector_store_type="chroma",
            params=[
                Parameter(name="foundation_model", param_type="C", values=[foundation_model]),
                Parameter(name="embedding_model", param_type="C", values=[embedding_model]),
            ],
        )

        optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=3)

        experiment = AI4RAGExperiment(
            client=client,
            documents=documents,
            benchmark_data=benchmark_data,
            search_space=search_space,
            optimizer_settings=optimizer_settings,
            event_handler=_make_event_handler("chroma_known_observations"),
            vector_store_config=ChromaConfig(),
            known_observations=known_observations,
        )

        experiment.search(skip_mps=True)

        assert len(experiment.results) > 0

        best_evals = experiment.results.get_best_evaluations(k=1)
        assert len(best_evals) == 1

        best_eval = best_evals[0]
        assert best_eval.final_score is not None
        assert 0 <= best_eval.final_score <= 1
