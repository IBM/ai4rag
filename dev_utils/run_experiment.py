# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Sample script to run ai4rag experiment"""

from pathlib import Path

from llama_stack_client import LlamaStackClient

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.search_space.prepare.prepare_search_space import prepare_search_space_with_llama_stack
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.file_store import FileStore
from dev_utils.utils import read_benchmark_from_json

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    _filepath = Path(__file__)
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

    search_space = prepare_search_space_with_llama_stack(payload={}, client=client)

    experiment = AI4RAGExperiment(
        client=client,
        documents=documents,
        benchmark_data=benchmark_data,
        search_space=search_space,
        optimizer_settings=optimizer_settings,
        event_handler=LocalEventHandler(output_path=_filepath.parent / "local" / "chroma_mocks"),
        vector_store_type="ls_milvus",
    )

    experiment.search(skip_mps=True)

    best_eval = experiment.results.get_best_evaluations(k=1)[0]

    print(best_eval)

    print(best_eval.rag_pattern.generate("What is greedy decoding?"))
