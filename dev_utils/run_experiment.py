# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Sample script to run ai4rag experiment"""

from pathlib import Path

from llama_stack_client import LlamaStackClient

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.file_store import FileStore
from dev_utils.mocks import MockedEmbeddingModel, MockedFoundationModel
from dev_utils.utils import read_benchmark_from_json

if __name__ == "__main__":
    _filepath = Path(__file__)
    # from dotenv import load_dotenv, find_dotenv
    # load_dotenv(find_dotenv())

    # client = LlamaStackClient(base_url="http://localhost:8321")
    client = LlamaStackClient()

    # change to direct to your local documents path
    documents_path = _filepath.parents[1] / "local" / "data" / "watsonx_sample" / "documents"

    # change to direct to your benchmark_data.json
    benchmark_data_path = _filepath.parents[1] / "local" / "data" / "watsonx_sample" / "watsonx_benchmark.json"

    file_store = FileStore(documents_path)
    documents = file_store.load_as_documents()
    benchmark_data = read_benchmark_from_json(benchmark_data_path)

    # Configure optimizer
    optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=2)

    # Edit configurations of search space
    search_space = AI4RAGSearchSpace(
        vector_store_type="chroma",
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[LSFoundationModel(model_id="vllm-inference-llama-3-1/redhataillama-31-8b-instruct", client=client)],
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[
                    LSEmbeddingModel(
                        model_id="vllm-embedding/granite-278m-multilingual-1",
                        client=client,
                        params={"embedding_dimension": 768, "context_length": 512},
                    )
                ],
            ),
        ]
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
        event_handler=LocalEventHandler(output_path=_filepath.parent / "local" / "chroma_experiment"),
        vector_store_type="chroma",
    )

    experiment.search(skip_mps=True)

    best_eval = experiment.results.get_best_evaluations(k=1)[0]

    print(best_eval)

    print(best_eval.rag_pattern.generate("What is greedy decoding?"))
