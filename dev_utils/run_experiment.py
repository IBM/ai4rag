# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Sample script to run ai4rag experiment"""

from pathlib import Path

from ogx_client import OgxClient

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.file_store import FileStore
from dev_utils.mocks import MockedEmbeddingModel, MockedFoundationModel
from dev_utils.utils import read_benchmark_from_json

if __name__ == "__main__":
    _filepath = Path(__file__)
    from dotenv import find_dotenv, load_dotenv

    # load_dotenv(find_dotenv())

    api_key = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ2NkVNNU9iR3g2TzcwMTZHTjFrMWQ4N2sxSkVUZFBvY2ZKV2ZLVmt4VGlRIn0.eyJleHAiOjE3ODE3MzAyMTksImlhdCI6MTc4MTY5NDIxOSwianRpIjoib25ydHJvOjY1NWNhZGU1LTJmMDMtMzg2OS0zOTcxLWM2ZWJjZWRjY2E5OCIsImlzcyI6Imh0dHBzOi8va2V5Y2xvYWstYWktZW5nLXByb2QuYXBwcy5yb3NhLmFpLWVuZy1wcm9kLndyaDcucDMub3BlbnNoaWZ0YXBwcy5jb20vcmVhbG1zL29neCIsImF1ZCI6WyJvZ3hfY2xpZW50IiwiYWNjb3VudCJdLCJzdWIiOiJhMzZlOGMyNC04YjAyLTQyMTEtYTNkMC00YTZkNWQ4YmJkMDkiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJvZ3hfY2xpZW50Iiwic2lkIjoibGRYZTB5NnhMd3UtZV9mX1pXejdrZ0YxIiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLW9neCJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiVW5jbGUgU2FtIiwicHJlZmVycmVkX3VzZXJuYW1lIjoib2d4X3VzZXJuYW1lIiwiZ2l2ZW5fbmFtZSI6IlVuY2xlIiwiZmFtaWx5X25hbWUiOiJTYW0iLCJlbWFpbCI6ImFzZGZhZHNAZmFkc2YuYWRzZmFzIn0.WbPYQ8yiexwAJ9q7wV_pUOW4HPoX5Hvzkt4WzMsHC8jz1fXgYsKPbrnkKpq40kpYT2l7l-26Ew9ptbC9YTE_MlRqG92Ktjb7mXqjAUlrNefdzUDHrkNsMBGFWZwT2y_dEsf-GWcrMNqHbkBTqvO8Jyv-9xrpxUBmz7AdzTgsBNx64UVf0fGbNlJmBrCYhuS3P64_k2-X9c9O5oXl8yXaUaC28BUYTpVlCfhAtRStlH3Q-wi8YhUIIADLlAjnpJgmLBbsyhBbQO6wQ08tCufb1CEI9EYlWFgLk5uwNqsSFHMr6uH-frwPE43Nf1b2Fl1cZSRi-b0BcY53drybt50d3w"
    base_url = "https://server-ogx.apps.rosa.ai-eng-prod.wrh7.p3.openshiftapps.com"
    client = OgxClient(base_url=base_url, api_key=api_key)
    # client = OgxClient()

    # change to direct to your local documents path
    documents_path = Path("autorag/documents")

    # change to direct to your benchmark_data.json
    benchmark_data_path = Path("autorag/watsonx_benchmark.json")

    file_store = FileStore(documents_path)
    documents = file_store.load_as_documents()
    benchmark_data = read_benchmark_from_json(benchmark_data_path)

    # Configure optimizer
    optimizer_settings = GAMOptSettings(max_evals=3, n_random_nodes=4)

    # Edit configurations of search space
    search_space = AI4RAGSearchSpace(
        vector_store_type="ogx",
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[OGXFoundationModel(model_id="vllm-inference-gpu-llama/llama-31-8b-instruct", client=client)],
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[
                    OGXEmbeddingModel(
                        model_id="vllm-embedding/bge-m3",
                        client=client,
                        params={"embedding_dimension": 1024, "context_length": 8192},
                    )
                ],
            ),
            Parameter(name="search_mode", values=["hybrid"]),
        ],
    )

    # search_space = AI4RAGSearchSpace(
    #     vector_store_type="chroma",
    #     params=[
    #         Parameter(
    #             name="foundation_model",
    #             param_type="C",
    #             values=[MockedFoundationModel(model_id="mocked_fm_1"), MockedFoundationModel(model_id="mocked_fm_2")],
    #         ),
    #         Parameter(
    #             name="embedding_model",
    #             param_type="C",
    #             values=[
    #                 MockedEmbeddingModel(
    #                     model_id="ollama/nomic-embed-text:latest", params={"embedding_dimension": 768}
    #                 ),
    #             ],
    #         ),
    #     ]
    # )

    experiment = AI4RAGExperiment(
        client=client,
        documents=documents,
        benchmark_data=benchmark_data,
        search_space=search_space,
        optimizer_settings=optimizer_settings,
        # event_handler=LocalEventHandler(output_path=_filepath.parent / "local" / "hybrid_test"),
        event_handler=LocalEventHandler(output_path="dev_utils/local_run"),
        vector_store_type="ogx",
        ogx_vector_io_provider_id="milvus-remote",
    )

    experiment.search(skip_mps=True)

    best_eval = experiment.results.get_best_evaluations(k=1)[0]

    print(best_eval)
