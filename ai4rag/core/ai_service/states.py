# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import TypedDict


class AI4RAGState(TypedDict):
    """
    State object for the Sequential RAG pipeline.

    Parameters
    ----------
    question : str
        The input question to be answered.

    retrieved_documents : list
        A list of documents retrieved during the retrieval phase.

    response : dict
        The final response, including the answer and any associated metadata.
    """

    question: str
    retrieved_documents: list
    response: dict
