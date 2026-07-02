# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import logging
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from ogx_client import OgxClient

from ai4rag import handler
from ai4rag.components.utils.docling_io import load_docling_documents
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.search_space.prepare.prepare_search_space import prepare_search_space_with_ogx
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

_logger = logging.getLogger("search-space-preparation")
_logger.addHandler(handler)

SUPPORTED_METRICS = ("faithfulness", "answer_correctness", "context_correctness")

_DEFAULT_METRIC = "faithfulness"
_DEFAULT_TOP_N_GENERATION = 3
_DEFAULT_TOP_K_EMBEDDING = 2
_DEFAULT_SAMPLE_SIZE = 5
_DEFAULT_SEED = 17


def _serialize_model(model: BaseFoundationModel | BaseEmbeddingModel) -> dict[str, Any]:
    """Convert a model instance to a plain dictionary with all its settings.

    Captures model identifier, type discriminator, inference parameters,
    and — for foundation models — the detected language.
    """
    is_embedding = isinstance(model, BaseEmbeddingModel)

    params = model.params
    if is_dataclass(params):
        params_dict = {
            field.name: getattr(params, field.name)
            for field in fields(params)
            if getattr(params, field.name) is not None
        }
    elif hasattr(params, "model_dump"):
        params_dict = params.model_dump()
    elif hasattr(params, "dict"):
        params_dict = params.dict()
    else:
        params_dict = {}

    result: dict[str, Any] = {
        "model_id": model.model_id,
        "type": "embedding" if is_embedding else "generation",
        "params": params_dict,
    }

    if not is_embedding:
        if hasattr(model, "language") and model.language is not None:
            result["language"] = model.language.to_dict()
        result["system_message_text"] = model.system_message_text
        result["user_message_text"] = model.user_message_text
        result["context_template_text"] = model.context_template_text

    return result


@dataclass
class SearchSpaceReport:
    """Result of the search-space preparation step.

    Attributes
    ----------
    search_space : dict[str, Any]
        Verbose representation of the search space, including selected
        model lists and non-model parameter ranges.
    selected_models : dict[str, list]
        Foundation and embedding model lists that survived pre-selection.
    """

    search_space: dict[str, Any]
    selected_models: dict[str, list]

    def save_json(self, path: str | Path) -> None:
        """Serialize the report to a JSON file.

        The file is suitable as input for the RAG optimization step.

        Parameters
        ----------
        path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.search_space, f, indent=2)


def prepare_search_space_report(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    test_data_path: str | Path,
    extracted_text_path: str | Path,
    ogx_client: OgxClient,
    embedding_models: list[str] | None = None,
    generation_models: list[str] | None = None,
    metric: str = _DEFAULT_METRIC,
    top_n_generation: int = _DEFAULT_TOP_N_GENERATION,
    top_k_embedding: int = _DEFAULT_TOP_K_EMBEDDING,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    random_seed: int = _DEFAULT_SEED,
    chunking_methods: list[str] | None = None,
    inference_max_threads: int = 10,
    **kwargs: Any,
) -> SearchSpaceReport:
    """Run model pre-selection and prepare a search-space report.

    Builds an :class:`AI4RAGSearchSpace` from the given model lists, runs
    :class:`ModelsPreSelector` when the number of models exceeds the
    configured caps, detects the benchmark language, and returns a
    structured report.

    Parameters
    ----------
    test_data_path
        Path to a JSON file containing benchmark questions and expected
        answers.
    extracted_text_path
        Path to a single DoclingDocument JSON file or a directory of such
        files.
    ogx_client
        An authenticated :class:`OgxClient` instance.
    embedding_models
        Embedding model identifiers.  ``None`` uses the server defaults.
    generation_models
        Generation model identifiers.  ``None`` uses the server defaults.
    metric
        Quality metric for intermediate pattern evaluation.  Must be one
        of ``"faithfulness"``, ``"answer_correctness"``, or
        ``"context_correctness"``.
    top_n_generation
        Maximum number of generation models to retain.
    top_k_embedding
        Maximum number of embedding models to retain.
    sample_size
        Number of benchmark records sampled for model pre-selection.
    random_seed
        Seed for reproducible sampling.
    chunking_methods
        When provided, constrains the ``chunking_method`` dimension of the
        search space to only these methods (e.g. ``["recursive"]`` or
        ``["hybrid"]``).  ``None`` uses the platform defaults (both
        ``"recursive"`` and ``"hybrid"``).
    inference_max_threads
        Maximum number of concurrent threads used when querying the
        RAG service during benchmark evaluation.  Lower values reduce
        per-request concurrency (useful when each request carries more
        retrieved context).  Defaults to ``10``.

    Returns
    -------
    SearchSpaceReport
        Structured report containing the verbose search space, selected
        models, and detected language.

    Raises
    ------
    ValueError
        If *metric* is not one of the supported values.
    TypeError
        If *embedding_models* or *generation_models* contain invalid entries.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Metric {metric!r} is not supported. Supported metrics are {list(SUPPORTED_METRICS)}.")

    _validate_model_list(embedding_models, "embedding_models")
    _validate_model_list(generation_models, "generation_models")
    _validate_chunking_methods(chunking_methods)

    preset = kwargs.get("preset", None)
    # Build payload and create search space via OGX
    payload: dict[str, list[dict[str, str]]] = {}
    if generation_models:
        payload["foundation_models"] = [{"model_id": gm} for gm in generation_models]
    if embedding_models:
        payload["embedding_models"] = [{"model_id": em} for em in embedding_models]

    # Load benchmark data and documents
    benchmark_df = pd.read_json(Path(test_data_path))
    benchmark_data = BenchmarkData(benchmark_df)
    documents = load_docling_documents(extracted_text_path)

    search_space = prepare_search_space_with_ogx(payload, client=ogx_client, benchmark_data=benchmark_df)

    # this is a tmp approach -- optimal solution is to change the `prepare_search_space_with_ogx`
    # func to accept different params than models only
    # but this will take more time which currently we do not have
    if preset == "speed":
        speed_parameters = [
            search_space["foundation_model"],
            search_space["embedding_model"],
            Parameter("chunk_size", "C", values=[128, 256]),
            Parameter("chunking_method", "C", values=chunking_methods),
        ]
        # recreate the search space with constrained chunk_sizes and models
        # validated by all of the checks in `prepare_search_space_with_ogx` func
        search_space = AI4RAGSearchSpace(params=speed_parameters)

    # Run model pre-selection when the number of models exceeds the caps
    fm_values = search_space["foundation_model"].values
    em_values = search_space["embedding_model"].values

    if len(fm_values) > top_n_generation or len(em_values) > top_k_embedding:
        mps = ModelsPreSelector(
            benchmark_data=benchmark_data.get_random_sample(n_records=sample_size, random_seed=random_seed),
            documents=documents,
            foundation_models=search_space._search_space["foundation_model"].values,  # pylint: disable=protected-access
            embedding_models=search_space._search_space["embedding_model"].values,  # pylint: disable=protected-access
            metric=metric,
            max_threads=inference_max_threads,
        )
        mps.evaluate_patterns()
        selected = mps.select_models(
            n_embedding_models=top_k_embedding,
            n_foundation_models=top_n_generation,
        )
        selected_models = {
            "foundation_model": selected["foundation_models"],
            "embedding_model": selected["embedding_models"],
        }
    else:
        selected_models = {
            "foundation_model": list(fm_values),
            "embedding_model": list(em_values),
        }

    # Build verbose representation with serialized model dicts
    verbose_repr: dict[str, Any] = {
        k: v.all_values()
        for k, v in search_space._search_space.items()  # pylint: disable=protected-access
        if k not in ("foundation_model", "embedding_model")
    }
    verbose_repr["foundation_model"] = [_serialize_model(m) for m in selected_models["foundation_model"]]
    verbose_repr["embedding_model"] = [_serialize_model(m) for m in selected_models["embedding_model"]]

    if chunking_methods is not None:
        available = set(verbose_repr["chunking_method"])
        unsupported = [m for m in chunking_methods if m not in available]
        if unsupported:
            raise ValueError(
                f"Unsupported chunking methods: {unsupported!r}. " f"Available methods: {sorted(available)!r}."
            )
        verbose_repr["chunking_method"] = chunking_methods
        _logger.info("Chunking methods constrained to: %s", verbose_repr["chunking_method"])

    return SearchSpaceReport(
        search_space=verbose_repr,
        selected_models=selected_models,
    )


def _validate_model_list(models: list[str] | None, name: str) -> None:
    """Validate that a model list, if provided, contains only non-empty strings."""
    if models is None:
        return
    if not isinstance(models, list):
        raise TypeError(f"{name} must be a list.")
    for i, m in enumerate(models):
        if not m:
            raise TypeError(f"{name}[{i}] must be a non-empty string.")


def _validate_chunking_methods(methods: list[str] | None) -> None:
    """Validate that chunking methods, if provided, are non-empty strings."""
    if methods is None:
        return
    if not isinstance(methods, list):
        raise TypeError("chunking_methods must be a list.")
    if not methods:
        raise ValueError("chunking_methods must not be empty when provided.")
    for i, m in enumerate(methods):
        if not isinstance(m, str) or not m.strip():
            raise TypeError(f"chunking_methods[{i}] must be a non-empty string.")
