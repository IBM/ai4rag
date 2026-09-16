# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from ai4rag.rag.template.agentic_rag_template import AgenticRAG
from ai4rag.rag.template.base_template import BaseRAGTemplate


def test_agentic_rag_is_a_base_template(mocker):
    """AgenticRAG exposes the common RAG template contract."""
    foundation_model = mocker.MagicMock()
    retriever = mocker.MagicMock()

    rag = AgenticRAG(foundation_model=foundation_model, retriever=retriever)

    assert isinstance(rag, BaseRAGTemplate)
    assert rag.foundation_model is foundation_model
    assert rag.retriever is retriever
