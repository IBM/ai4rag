# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Sample script to run ai4rag experiment"""

from llama_stack_client import LlamaStackClient
from pathlib import Path

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.rag.foundation_models.foundation_model import LSFoundationModel
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

from dev_utils.file_store import FileStore
from dev_utils.local_event_handler import LocalEventHandler
from dev_utils.utils import read_benchmark_from_json


if __name__ == "__main__":
    _filepath = Path(__file__)
    client = LlamaStackClient(base_url="http://localhost:8321")

    # change to direct to your local documents path
    documents_path = _filepath.parents[1] / "local" / "data" / "watsonx_sample" / "documents"

    # change to direct to your benchmark_data.json
    benchmark_data_path = _filepath.parents[1] / "local" / "data" / "watsonx_sample" / "watsonx_benchmark.json"

    file_store = FileStore(documents_path)
    documents = file_store.load_as_documents()
    benchmark_data = read_benchmark_from_json(benchmark_data_path)

    # Configure optimiser
    optimiser_settings = GAMOptSettings(max_evals=4, n_random_nodes=2)

    # Edit configurations of search space
    search_space = AI4RAGSearchSpace(
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[LSFoundationModel(model_id="ollama/llama3.2:3b", client=client)],
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[LSEmbeddingModel(model_id="ollama/nomic-embed-text:latest", client=client)],
            ),
        ]
    )

    experiment = AI4RAGExperiment(
        client=client,
        documents=documents,
        benchmark_data=benchmark_data,
        search_space=search_space,
        optimiser_settings=optimiser_settings,
        event_handler=LocalEventHandler(),
        output_path=_filepath.parent / "local" / "results",
        vector_store_type="chroma",
    )

    best = experiment.search(skip_mps=True)

    print(best)
