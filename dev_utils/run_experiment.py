# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Sample script to run ai4rag experiment"""

from pathlib import Path

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.rag.vector_store.config import get_vector_store_config
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.file_store import FileStore
from dev_utils.utils import build_maas_model, create_dev_maas_client, read_benchmark_from_json

if __name__ == "__main__":
    _filepath = Path(__file__)
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(".env.local"))

    client = create_dev_maas_client()

    # change to direct to your local documents path
    documents_path = _filepath.parents[1] / "local" / "data" / "rh_summit_2026" / "documents"

    # change to direct to your benchmark_data.json
    benchmark_data_path = _filepath.parents[1] / "local" / "data" / "rh_summit_2026" / "benchmark_data_4q.json"

    file_store = FileStore(documents_path)
    documents = file_store.load_as_documents()
    benchmark_data = read_benchmark_from_json(benchmark_data_path)

    # Configure optimizer
    optimizer_settings = GAMOptSettings(max_evals=8, n_random_nodes=4)

    # Edit configurations of search space
    search_space = AI4RAGSearchSpace(
        vector_store_type="milvus",
        params=[
            Parameter(
                name="foundation_model",
                param_type="C",
                values=[build_maas_model(client, model_id="qwen3-8b-fp8-dynamic", model_type="llm")],
            ),
            Parameter(
                name="embedding_model",
                param_type="C",
                values=[build_maas_model(client, model_id="bge-m3", model_type="embedding")],
            ),
        ],
    )

    vs_config = get_vector_store_config(provider="milvus")

    experiment = AI4RAGExperiment(
        documents=documents,
        benchmark_data=benchmark_data,
        search_space=search_space,
        optimizer_settings=optimizer_settings,
        event_handler=LocalEventHandler(output_path=_filepath.parent / "local" / "chunkers"),
        # event_handler=LocalEventHandler(),
        vector_store_config=vs_config,
    )

    experiment.search(skip_mps=True)

    best_eval = experiment.results.get_best_evaluations(k=1)[0]

    print(best_eval)
