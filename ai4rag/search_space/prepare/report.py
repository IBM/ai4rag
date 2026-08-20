# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Serialize a prepared search space into the JSON report exchanged between steps.

The *search-space report* is the file-format contract that ties the AutoRAG
pipeline steps together: search-space preparation writes it, model
pre-selection may rewrite it (trimming the model lists), and the optimization
step restores it. Its *read* half lives in :mod:`ai4rag.search_space.prepare.models`
(``get_foundation_models`` / ``get_embedding_models`` in restore mode); this
module is its *write* half, so the two directions sit side by side in the same
package.

The report is a plain ``dict`` keyed by search-space parameter name. Non-model
parameters map to their de-duplicated value lists (taken from the rule-filtered
combinations, so invalid combinations never leak downstream); ``foundation_model``
and ``embedding_model`` map to lists of serialized model specs (see
:func:`ai4rag.search_space.prepare.models.serialize_model`).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4rag.search_space.prepare.models import serialize_model
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

__all__ = ["SearchSpaceReport", "build_search_space_report"]

_logger = logging.getLogger("search-space-report")

_MODEL_KEYS = ("foundation_model", "embedding_model")


@dataclass
class SearchSpaceReport:
    """Serialized representation of a prepared search space.

    Attributes
    ----------
    search_space : dict[str, Any]
        Verbose representation of the search space: non-model parameters as
        de-duplicated value lists and ``foundation_model`` / ``embedding_model``
        as lists of serialized model specs. This dict is what
        :func:`save_json` writes and what the optimization step restores.
    """

    search_space: dict[str, Any]

    def save_json(self, path: str | Path) -> None:
        """Write the report to a JSON file suitable as optimization input.

        Parameters
        ----------
        path
            Destination file path; parent directories are created as needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.search_space, f, indent=2, ensure_ascii=False)


def build_search_space_report(search_space: AI4RAGSearchSpace) -> SearchSpaceReport:
    """Assemble a :class:`SearchSpaceReport` from a prepared search space.

    Non-model parameters are reduced to their distinct values *as they appear in
    the rule-filtered combinations* — so any value that only occurred in
    combinations rejected by the search-space rules is dropped, and downstream
    steps never see an invalid configuration. The model dimensions are serialized
    verbatim from the search space's own model instances via
    :func:`~ai4rag.search_space.prepare.models.serialize_model`.

    Parameters
    ----------
    search_space : AI4RAGSearchSpace
        The search space produced by
        :func:`~ai4rag.search_space.prepare.prepare_search_space.prepare_search_space_with_maas`.

    Returns
    -------
    SearchSpaceReport
        The serialized report, ready to persist with
        :meth:`SearchSpaceReport.save_json`.
    """
    valid_combinations = search_space.combinations
    if not valid_combinations:
        _logger.warning("No valid combinations remain after applying search space rules.")

    non_model_keys = [param.name for param in search_space.params if param.name not in _MODEL_KEYS]
    verbose_repr: dict[str, Any] = {
        key: list(dict.fromkeys(combo[key] for combo in valid_combinations)) for key in non_model_keys
    }

    for key in _MODEL_KEYS:
        verbose_repr[key] = [serialize_model(model) for model in search_space[key].values]

    return SearchSpaceReport(search_space=verbose_repr)
