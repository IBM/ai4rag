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
from ai4rag.components.optimization.judge_selection import (
    JudgeSelectionContext,
    build_evaluation_block,
    select_judge_model,
)
from ai4rag.components.utils.docling_io import load_docling_documents
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.evaluator.base_evaluator import SUPPORTED_OPTIMIZATION_METRICS, MetricType
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.search_space.prepare.prepare_search_space import prepare_search_space_with_ogx

_logger = logging.getLogger("search-space-preparation")
_logger.addHandler(handler)

SUPPORTED_METRICS = SUPPORTED_OPTIMIZATION_METRICS
MPS_SELECTION_METRICS = frozenset(
    {
        MetricType.FAITHFULNESS,
        MetricType.ANSWER_CORRECTNESS,
        MetricType.CONTEXT_CORRECTNESS,
    }
)

_DEFAULT_METRIC = MetricType.FAITHFULNESS
_DEFAULT_TOP_N_GENERATION = 3
_DEFAULT_TOP_K_EMBEDDING = 2
_DEFAULT_SAMPLE_SIZE = 5
_DEFAULT_SEED = 17


def _mps_metric_for(metric: str) -> str:
    """Map an optimization metric to the Unitxt metric used by MPS."""
    if metric in MPS_SELECTION_METRICS:
        return metric
    if metric in (MetricType.OVERALL_SCORE, MetricType.ANSWER_RELEVANCE):
        return MetricType.FAITHFULNESS
    raise ValueError(f"Metric {metric!r} cannot be mapped to an MPS selection metric.")


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
    chunk_sizes: list[int] | None = None,
    chunk_overlaps: list[int] | None = None,
    inference_max_threads: int = 10,
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
        Optimization metric passed from the pipeline.  Accepts all supported
        optimization metrics (including ``"overall_score"`` and
        ``"answer_relevance"``).  Model pre-selection (MPS) uses Unitxt
        metrics only; ``"overall_score"`` and ``"answer_relevance"`` are
        mapped to ``"faithfulness"`` for that step.
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
    chunk_sizes
        When provided, constrains the ``chunk_size`` dimension of the
        search space to only these sizes (e.g. ``[256, 512]``).  ``None``
        uses the platform defaults.
    chunk_overlaps
        When provided, constrains the ``chunk_overlap`` dimension of the
        search space to only these values (e.g. ``[0, 128]``).  ``None``
        uses the platform defaults.
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
    pydantic.ValidationError
        If *chunking_methods*, *chunk_sizes*, or *chunk_overlaps* fail structural
        validation (wrong type, empty list, or invalid element types).
    SearchSpaceValueError
        If *chunking_methods* contains values not in
        :attr:`~ai4rag.utils.constants.ChunkingConstraints.METHODS`, or
        *chunk_sizes* contains values outside
        ``[ChunkingConstraints.MIN_CHUNK_SIZE, ChunkingConstraints.MAX_CHUNK_SIZE]``,
        or *chunk_overlaps* contains values outside
        ``[ChunkingConstraints.MIN_CHUNK_OVERLAP, ChunkingConstraints.MAX_CHUNK_OVERLAP]``.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Metric {metric!r} is not supported. Supported metrics are {sorted(SUPPORTED_METRICS)}.")

    mps_metric = _mps_metric_for(metric)
    if mps_metric != metric:
        _logger.info(
            "MPS model pre-selection uses %r (optimization metric is %r).",
            mps_metric,
            metric,
        )

    _validate_model_list(embedding_models, "embedding_models")
    _validate_model_list(generation_models, "generation_models")

    # Build payload and create search space via OGX
    payload: dict[str, Any] = {}
    if generation_models:
        payload["foundation_models"] = [{"model_id": gm} for gm in generation_models]
    if embedding_models:
        payload["embedding_models"] = [{"model_id": em} for em in embedding_models]
    if chunking_methods is not None:
        payload["chunking_methods"] = chunking_methods
    if chunk_sizes is not None:
        payload["chunk_sizes"] = chunk_sizes
    if chunk_overlaps is not None:
        payload["chunk_overlaps"] = chunk_overlaps

    # Load benchmark data and documents
    benchmark_df = pd.read_json(Path(test_data_path))
    benchmark_data = BenchmarkData(benchmark_df)
    documents = load_docling_documents(extracted_text_path)

    search_space = prepare_search_space_with_ogx(
        payload,
        client=ogx_client,
        benchmark_data=benchmark_df,
    )
    _logger.info(
        "Search space chunking_method=%s chunk_size=%s chunk_overlap=%s",
        list(search_space["chunking_method"].values),
        list(search_space["chunk_size"].values),
        list(search_space["chunk_overlap"].values),
    )

    selected_models = _select_models_with_mps(
        search_space=search_space,
        benchmark_data=benchmark_data,
        documents=documents,
        top_n_generation=top_n_generation,
        top_k_embedding=top_k_embedding,
        sample_size=sample_size,
        random_seed=random_seed,
        metric=mps_metric,
        inference_max_threads=inference_max_threads,
    )

    verbose_repr = _build_verbose_search_space(
        search_space=search_space,
        selected_models=selected_models,
        benchmark_data=benchmark_data,
        documents=documents,
        ogx_client=ogx_client,
        inference_max_threads=inference_max_threads,
        chunking_methods=chunking_methods,
    )

    return SearchSpaceReport(
        search_space=verbose_repr,
        selected_models=selected_models,
    )


def _select_models_with_mps(  # pylint: disable=too-many-arguments
    *,
    search_space: Any,
    benchmark_data: BenchmarkData,
    documents: list,
    top_n_generation: int,
    top_k_embedding: int,
    sample_size: int,
    random_seed: int,
    metric: str,
    inference_max_threads: int,
) -> dict[str, list]:
    """Run MPS when model counts exceed caps; otherwise return full model lists."""
    fm_values = search_space["foundation_model"].values
    em_values = search_space["embedding_model"].values

    if len(fm_values) <= top_n_generation and len(em_values) <= top_k_embedding:
        return {"foundation_model": list(fm_values), "embedding_model": list(em_values)}

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
    return {
        "foundation_model": selected["foundation_models"],
        "embedding_model": selected["embedding_models"],
    }


def _build_verbose_search_space(  # pylint: disable=too-many-arguments
    *,
    search_space: Any,
    selected_models: dict[str, list],
    benchmark_data: BenchmarkData,
    documents: list,
    ogx_client: OgxClient,
    inference_max_threads: int,
    chunking_methods: list[str] | None,
) -> dict[str, Any]:
    """Build the verbose search-space dict including evaluation metadata."""
    valid_combinations = search_space.combinations
    if not valid_combinations:
        _logger.warning("No valid combinations remain after applying search space rules.")
    non_model_keys = [p.name for p in search_space.params if p.name not in ("foundation_model", "embedding_model")]
    verbose_repr: dict[str, Any] = {
        key: list(dict.fromkeys(combo[key] for combo in valid_combinations)) for key in non_model_keys
    }
    verbose_repr["foundation_model"] = [_serialize_model(m) for m in selected_models["foundation_model"]]
    verbose_repr["embedding_model"] = [_serialize_model(m) for m in selected_models["embedding_model"]]

    if chunking_methods is not None:
        available = set(verbose_repr.get("chunking_method", []))
        if available:
            unsupported = [m for m in chunking_methods if m not in available]
            if unsupported:
                raise ValueError(
                    f"Unsupported chunking methods: {unsupported!r}. Available methods: {sorted(available)!r}."
                )
            verbose_repr["chunking_method"] = chunking_methods
            _logger.info("Chunking methods constrained to: %s", verbose_repr["chunking_method"])

    verbose_repr["evaluation"] = _resolve_evaluation_block(
        selected_models=selected_models,
        benchmark_data=benchmark_data,
        documents=documents,
        ogx_client=ogx_client,
        inference_max_threads=inference_max_threads,
    )
    return verbose_repr


def _resolve_evaluation_block(
    *,
    selected_models: dict[str, list],
    benchmark_data: BenchmarkData,
    documents: list,
    ogx_client: OgxClient,
    inference_max_threads: int,
) -> list[dict[str, Any]]:
    """Auto-select a judge model and build the hybrid evaluation artifact."""
    ogx_api_key = getattr(ogx_client, "api_key", "") or ""
    resolved_judge_id = select_judge_model(
        JudgeSelectionContext(
            generation_models=selected_models["foundation_model"],
            embedding_models=selected_models["embedding_model"],
            benchmark_data=benchmark_data,
            documents=documents,
            ogx_base_url=ogx_client.base_url,
            ogx_api_key=ogx_api_key,
            max_threads=inference_max_threads,
        )
    )
    if not resolved_judge_id:
        raise ValueError("Failed to resolve judge model during search-space preparation.")
    return build_evaluation_block(resolved_judge_id)


def _validate_model_list(models: list[str] | None, name: str) -> None:
    """Validate that a model list, if provided, contains only non-empty strings."""
    if models is None:
        return
    if not isinstance(models, list):
        raise TypeError(f"{name} must be a list.")
    for i, m in enumerate(models):
        if not m:
            raise TypeError(f"{name}[{i}] must be a non-empty string.")
