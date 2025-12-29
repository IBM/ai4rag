# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = [
    "get_default_ai4rag_search_space_parameters",
]

_default_chunking_methods = ("recursive",)
_default_chunk_sizes = (1024, 2048)
_default_chunk_overlaps = (256, 512)
_default_retrieval_methods = ("simple", "window")
_default_window_sizes = (0, 1, 2, 3, 4, 5)
_default_numbers_of_chunks = (3, 5, 10)


def get_default_ai4rag_search_space_parameters() -> list[Parameter]:
    """
    Function to return default search space containing experiment parameters.

    Returns
    -------
    list[Parameter]
        Parameters that will be used for creating AI4RAGSearchSpace
    """

    default_search_space_parameters = [
        Parameter(
            name=AI4RAGParamNames.CHUNKING_METHOD,
            param_type="C",
            values=_default_chunking_methods,
        ),
        Parameter(
            name=AI4RAGParamNames.CHUNK_SIZE,
            param_type="C",
            values=_default_chunk_sizes,
        ),
        Parameter(
            name=AI4RAGParamNames.CHUNK_OVERLAP,
            param_type="C",
            values=_default_chunk_overlaps,
        ),
        Parameter(
            name=AI4RAGParamNames.RETRIEVAL_METHOD,
            param_type="C",
            values=_default_retrieval_methods,
        ),
        Parameter(name=AI4RAGParamNames.WINDOW_SIZE, param_type="C", values=_default_window_sizes),
        Parameter(
            name=AI4RAGParamNames.NUMBER_OF_CHUNKS,
            param_type="C",
            values=_default_numbers_of_chunks,
        ),
    ]

    return default_search_space_parameters
