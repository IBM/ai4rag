# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = [
    "get_default_ai4rag_search_space_parameters",
]

# Note: "" and 0 are sentinels for unused params; ranker_alpha uses 1 as sentinel (0 means 100% sparse)
_default_chunking_methods = ("recursive", "hybrid")
_default_chunk_sizes = (512, 1024, 2048)
_default_chunk_overlaps = (0, 128, 256)
_default_retrieval_methods = ("simple",)
_default_window_sizes = (0,)
_default_numbers_of_chunks = (3, 5, 10)
_default_search_modes = ("vector", "hybrid")
_default_ranker_strategies = ("", "rrf", "weighted")
_default_ranker_k = (0, 60)
_default_ranker_alpha = (1, 0.5)


def get_default_ai4rag_search_space_parameters() -> list[Parameter]:
    """Return the default search space parameters for an AI4RAG experiment.

    Returns
    -------
    list[Parameter]
        Parameters that will be used for creating AI4RAGSearchSpace.
    """
    default_search_space_parameters = [
        Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, values=_default_chunking_methods),
        Parameter(name=AI4RAGParamNames.CHUNK_SIZE, values=_default_chunk_sizes),
        Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, values=_default_chunk_overlaps),
        Parameter(name=AI4RAGParamNames.RETRIEVAL_METHOD, values=_default_retrieval_methods),
        Parameter(name=AI4RAGParamNames.WINDOW_SIZE, values=_default_window_sizes),
        Parameter(name=AI4RAGParamNames.NUMBER_OF_CHUNKS, values=_default_numbers_of_chunks),
        Parameter(name=AI4RAGParamNames.SEARCH_MODE, values=_default_search_modes),
        Parameter(name=AI4RAGParamNames.RANKER_STRATEGY, values=_default_ranker_strategies),
        Parameter(name=AI4RAGParamNames.RANKER_K, values=_default_ranker_k),
        Parameter(name=AI4RAGParamNames.RANKER_ALPHA, values=_default_ranker_alpha),
    ]

    return default_search_space_parameters
