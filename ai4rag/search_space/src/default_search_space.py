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
_default_chunking_methods = ("recursive",)
_default_chunk_sizes = (1024, 2048, 4096)
_default_chunk_overlaps = (128, 256)
_default_retrieval_methods = ("simple",)
_default_window_sizes = (0,)
_default_chroma_retrieval_methods = ("simple", "window")
_default_chroma_window_sizes = (0, 1, 3, 5)
_default_numbers_of_chunks = (3, 5, 10)
_default_search_modes = ("vector", "hybrid")  # currently off as llama stack has issue with hybrid search mode
_default_ranker_strategies = ("", "rrf", "weighted")  # to extend with normalized
_default_ranker_k = (0, 60)  # currently off as llama stack has issue with hybrid search mode
_default_ranker_alpha = (1, 0.5)  # currently off as llama stack has issue with hybrid search mode


def get_default_ai4rag_search_space_parameters(vector_store_type: str = "ls_milvus") -> list[Parameter]:
    """
    Function to return default search space containing experiment parameters.

    Parameters
    ----------
    vector_store_type : str, default="ls_milvus"
        Type of vector store. When "chroma", hybrid search parameters are excluded
        since ChromaDB does not support hybrid search.

    Returns
    -------
    list[Parameter]
        Parameters that will be used for creating AI4RAGSearchSpace
    """

    if vector_store_type == "chroma":
        retrieval_methods = _default_chroma_retrieval_methods
        window_sizes = _default_chroma_window_sizes
    else:
        retrieval_methods = _default_retrieval_methods
        window_sizes = _default_window_sizes

    default_search_space_parameters = [
        Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, values=_default_chunking_methods),
        Parameter(name=AI4RAGParamNames.CHUNK_SIZE, values=_default_chunk_sizes),
        Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, values=_default_chunk_overlaps),
        Parameter(name=AI4RAGParamNames.RETRIEVAL_METHOD, values=retrieval_methods),
        Parameter(name=AI4RAGParamNames.WINDOW_SIZE, values=window_sizes),
        Parameter(name=AI4RAGParamNames.NUMBER_OF_CHUNKS, values=_default_numbers_of_chunks),
        Parameter(name=AI4RAGParamNames.INCLUDE_CHUNK_METADATA, values=(False,)),
    ]

    if vector_store_type == "chroma":
        default_search_space_parameters.append(
            Parameter(name=AI4RAGParamNames.SEARCH_MODE, values=("vector",)),
        )
    else:
        default_search_space_parameters.extend(
            [
                Parameter(name=AI4RAGParamNames.SEARCH_MODE, values=("vector",)),
            ]
        )

    return default_search_space_parameters
