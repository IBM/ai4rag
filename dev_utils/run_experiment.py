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
    import os
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    print()

    # client = OgxClient(base_url="http://localhost:8321")
    client = OgxClient(
        api_key=os.environ["OGX_CLIENT_API_KEY"],
        base_url=os.environ["OGX_CLIENT_BASE_URL"],
    )

    # change to direct to your local documents path
    documents_path = _filepath.parents[1] / "local" / "data" / "documents"

    # change to direct to your benchmark_data.json
    benchmark_data_path = (
        _filepath.parents[1] / "local" / "data" / "benchmark_data_4q.json"
    )

    file_store = FileStore(documents_path)
    documents = file_store.load_as_documents()
    benchmark_data = read_benchmark_from_json(benchmark_data_path)

    # Configure optimizer
    optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=4)

    # Edit configurations of search space
    search_space = AI4RAGSearchSpace(
        vector_store_type="ogx",
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[
                    OGXFoundationModel(
                        model_id="vllm-inference-gpu-mistral/redhataimistral-small-24b-inst",
                        client=client,
                    )
                ],
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
        event_handler=LocalEventHandler(),
        vector_store_type="ogx",
        ogx_vector_io_provider_id="pgvector",
    )

    experiment.search(skip_mps=True)

    best_eval = experiment.results.get_best_evaluations(k=1)[0]

    print(best_eval)
