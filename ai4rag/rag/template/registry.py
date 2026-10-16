# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Registry for RAG templates supported by persisted patterns and notebooks."""

from dataclasses import dataclass

from .base_template import BaseRAGTemplate
from .react_rag_template import ReActRAG
from .sequential_rag_template import SequentialRAG


@dataclass(frozen=True)
class RAGTemplateSpec:
    """Describe one template that can be persisted in a pattern.

    Parameters
    ----------
    template_id : str
        Stable identifier stored in ``pattern.json``.
    template_class : type[BaseRAGTemplate]
        Implementation class used by the experiment and generated notebook.
    import_path : str
        Module from which the notebook imports ``template_class``.
    """

    template_id: str
    template_class: type[BaseRAGTemplate]
    import_path: str


_TEMPLATE_SPECS = (
    RAGTemplateSpec("react_rag", ReActRAG, "ai4rag.rag.template.react_rag_template"),
    # Graph retrieval is a provider concern. It uses the same agentic template
    # implementation while retaining a distinct, published pattern identifier.
    RAGTemplateSpec("react_graph_rag", ReActRAG, "ai4rag.rag.template.react_rag_template"),
    RAGTemplateSpec("sequential_rag", SequentialRAG, "ai4rag.rag.template.sequential_rag_template"),
)
_SPECS_BY_ID = {spec.template_id: spec for spec in _TEMPLATE_SPECS}


def get_template_spec(template_id: str) -> RAGTemplateSpec:
    """Return the allowlisted specification for a persisted template ID.

    Raises
    ------
    ValueError
        If the ID is unsupported, so notebook generation never imports an
        arbitrary module named by pattern data.
    """
    try:
        return _SPECS_BY_ID[template_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported template_id {template_id!r}. Expected one of {sorted(_SPECS_BY_ID)}.") from exc
