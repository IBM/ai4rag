# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Registry for RAG templates supported by persisted patterns and notebooks."""

from dataclasses import dataclass

from .agentic_rag_template import AgenticRAG
from .base_template import BaseRAGTemplate
from .simple_rag_template import SimpleRAG


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
    RAGTemplateSpec("agentic_rag", AgenticRAG, "ai4rag.rag.template.agentic_rag_template"),
    # Graph retrieval is a provider concern. It uses the same agentic template
    # implementation while retaining a distinct, published pattern identifier.
    RAGTemplateSpec("agentic_graph_rag", AgenticRAG, "ai4rag.rag.template.agentic_rag_template"),
    RAGTemplateSpec("simple_rag", SimpleRAG, "ai4rag.rag.template.simple_rag_template"),
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
    except KeyError as exc:
        raise ValueError(
            f"Unsupported template_id {template_id!r}. Expected one of {sorted(_SPECS_BY_ID)}."
        ) from exc


def template_id_for_class(template_class: type[BaseRAGTemplate]) -> str:
    """Return the canonical pattern ID for an experiment template class."""
    for spec in _TEMPLATE_SPECS:
        if spec.template_class is template_class and spec.template_id != "agentic_graph_rag":
            return spec.template_id
    raise ValueError(
        f"Unsupported RAG template class {template_class.__name__!r}. "
        f"Expected one of {[spec.template_class.__name__ for spec in _TEMPLATE_SPECS]}."
    )
