# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Sample script to run ai4rag experiment"""

import os
from pathlib import Path

from openai import OpenAI

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler.event_handler import LocalEventHandler
from dev_utils.file_store import FileStore
from dev_utils.utils import read_benchmark_from_json

if __name__ == "__main__":
    _filepath = Path(__file__)
    foundation_client = OpenAI(
        base_url=os.getenv("FOUNDATION_OPENAI_BASE_URL"), api_key=os.getenv("FOUNDATION_OPENAI_API_KEY")
    )
    embedding_client = OpenAI(
        base_url=os.getenv("EMBEDDING_OPENAI_BASE_URL"), api_key=os.getenv("EMBEDDING_OPENAI_API_KEY")
    )

    # change to direct to your local documents path
    documents_path = _filepath.parents[1] / "local" / "data" / "watsonx_sample" / "documents"

    # change to direct to your benchmark_data.json
    benchmark_data_path = _filepath.parents[1] / "local" / "data" / "watsonx_sample" / "benchmark_data.json"

    file_store = FileStore(documents_path)
    documents = file_store.load_as_documents()
    benchmark_data = read_benchmark_from_json(benchmark_data_path)

    # Configure optimizer
    optimizer_settings = GAMOptSettings(max_evals=4, n_random_nodes=2)

    # Edit configurations of search space
    search_space = AI4RAGSearchSpace(
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[OpenAIFoundationModel(model_id="redhataillama-31-8b-instruct", client=foundation_client)],
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[
                    OpenAIEmbeddingModel(
                        model_id="granite-278m-multilingual-1",
                        client=embedding_client,
                        params={"embedding_dimension": 768, "context_length": 8192},
                    )
                ],
            ),
        ]
    )

    experiment = AI4RAGExperiment(
        documents=documents,
        benchmark_data=benchmark_data,
        search_space=search_space,
        optimizer_settings=optimizer_settings,
        event_handler=LocalEventHandler(output_path=_filepath.parent / "local" / "chroma_mocks"),
        vector_store_type="chroma",
    )

    experiment.search(skip_mps=True)

    print(experiment.results.get_best_evaluations(1))
