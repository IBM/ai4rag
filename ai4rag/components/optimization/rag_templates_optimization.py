# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import logging
import os
from dataclasses import dataclass
from json import dump as json_dump
from pathlib import Path
from typing import Any

import pandas as pd
from ogx_client import OgxClient

from ai4rag import handler
from ai4rag.components.assets_generator import build_pattern_json, generate_notebook_from_template
from ai4rag.components.utils.docling_io import load_docling_documents
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.evaluator.judge_selection import select_judge_model
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from ai4rag.evaluator.metric import Metrics
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel
from ai4rag.rag.foundation_models.base_model import Language
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler.event_handler import KFPEventHandler

_logger = logging.getLogger("rag-templates-optimization")
_logger.addHandler(handler)

DEFAULT_MAX_RAG_PATTERNS = 8
MIN_MAX_RAG_PATTERNS_RANGE = (4, 20)
DEFAULT_METRIC = Metrics.OVERALL_SCORE.name
SUPPORTED_OPTIMIZATION_METRICS = frozenset(
    {
        Metrics.FAITHFULNESS.name,
        Metrics.ANSWER_CORRECTNESS.name,
        Metrics.CONTEXT_CORRECTNESS.name,
        Metrics.OVERALL_SCORE.name,
    }
)


@dataclass
class OptimizationResult:
    """Output of a complete RAG optimization run.

    Attributes
    ----------
    patterns : list[dict]
        Pattern definitions for each evaluated RAG configuration.
    evaluations : list
        Raw evaluation result objects from the experiment.
    """

    patterns: list[dict]
    evaluations: list


def run_rag_optimization(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    extracted_text_path: str | Path,
    test_data_path: str | Path,
    search_space_report_path: str | Path,
    output_dir: str | Path,
    ogx_client: OgxClient,
    vector_io_provider_id: str,
    test_data_key: str = "",
    input_data_key: str = "",
    optimization_settings: dict | None = None,
    inference_max_threads: int = 10,
    indexing_pipeline_params: dict | None = None,
    judge_enabled: bool = True,
) -> OptimizationResult:
    """Run a full AI4RAG optimization experiment and generate output artefacts.

    Orchestrates the end-to-end workflow: load documents, reconstruct the
    search space from a JSON report, run the experiment, then generate
    per-pattern outputs (``pattern.json``, notebooks, evaluation results).

    Parameters
    ----------
    extracted_text_path
        Path to a folder of DoclingDocument JSON files (or a single file).
    test_data_path
        Path to a benchmark JSON file with questions and expected answers.
    search_space_report_path
        Path to the JSON report produced by the search-space preparation step.
    output_dir
        Root directory where per-pattern output folders are written.
    ogx_client
        An authenticated :class:`OgxClient` instance.
    vector_io_provider_id
        Vector I/O provider identifier registered in OGX.
    test_data_key
        Object-storage key for the test data file, embedded into generated
        notebooks.
    input_data_key
        Object-storage key for the documents directory, embedded into
        generated notebooks.
    optimization_settings
        Optional dictionary with ``"metric"`` and/or
        ``"max_number_of_rag_patterns"`` overrides.
    inference_max_threads
        Maximum number of concurrent threads used when querying the
        RAG service during benchmark evaluation.  Lower values reduce
        per-request concurrency (useful when each request carries more
        retrieved context).  Defaults to ``10``.
    indexing_pipeline_params : dict | None, default=None
        Parameters required to enhance pattern.json with indexing pipeline
        settings.
    judge_enabled : bool, default=True
        Whether LLM as a Judge metrics should be calculated.

    Returns
    -------
    OptimizationResult
        Contains the list of pattern definitions, raw evaluations, and the
        total number of parameter combinations explored.

    Raises
    ------
    ValueError
        If ``test_data_key`` does not point to a JSON file,
        ``vector_io_provider_id`` is empty, or the optimization metric is
        not supported.
    TypeError
        If ``optimization_settings`` has invalid types.
    """
    # --- Input validation ---
    if not isinstance(test_data_key, str) or not test_data_key.strip() or not test_data_key.lower().endswith(".json"):
        raise ValueError("test_data_key must point to a JSON file.")

    if not isinstance(vector_io_provider_id, str) or not vector_io_provider_id.strip():
        raise ValueError("vector_io_provider_id must be a non-empty string.")
    vector_io_provider_id = vector_io_provider_id.strip()

    settings = _validate_optimization_settings(optimization_settings)
    optimization_metric = settings.get("metric") or DEFAULT_METRIC
    if optimization_metric not in SUPPORTED_OPTIMIZATION_METRICS:
        raise ValueError(
            f"Optimization metric {optimization_metric} is not supported. "
            f"Select one of {sorted(SUPPORTED_OPTIMIZATION_METRICS)}."
        )

    documents = load_docling_documents(extracted_text_path)
    benchmark_data = pd.read_json(Path(test_data_path))
    benchmark_data_obj = BenchmarkData(benchmark_data)

    # --- Reconstruct search space from report ---
    with open(search_space_report_path, "r", encoding="utf-8") as f:
        search_space_raw: dict[str, Any] = json.load(f)

    foundation_models: list[OGXFoundationModel] = []
    embedding_models: list[OGXEmbeddingModel] = []
    params: list[Parameter] = []

    for param_name, values in search_space_raw.items():
        if param_name == "foundation_model":
            values = [_deserialize_model(m, ogx_client) for m in values]
            foundation_models = values
        elif param_name == "embedding_model":
            values = [_deserialize_model(m, ogx_client) for m in values]
            embedding_models = values
        params.append(Parameter(param_name, "C", values=values))

    search_space = AI4RAGSearchSpace(params=params)

    if judge_enabled:
        # --- Select judge model and build evaluators ---
        judge_model = select_judge_model(
            generation_models=foundation_models,
            embedding_models=embedding_models,
            benchmark_data=benchmark_data_obj,
            documents=documents,
            max_threads=inference_max_threads,
        )
        _logger.info("Judge model selected: %s", judge_model.model_id)

        evaluators = [UnitxtEvaluator(), LLMaJEvaluator(model=judge_model)]
    else:
        evaluators = [UnitxtEvaluator()]

    # --- Configure experiment ---
    max_rag_patterns = settings.get("max_number_of_rag_patterns", DEFAULT_MAX_RAG_PATTERNS)
    if isinstance(max_rag_patterns, str):
        max_rag_patterns = int(max_rag_patterns.strip())
    optimizer_settings = GAMOptSettings(max_evals=max_rag_patterns)

    event_handler = KFPEventHandler()

    rag_exp = AI4RAGExperiment(
        client=ogx_client,
        event_handler=event_handler,
        optimizer_settings=optimizer_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_type="ogx",
        documents=documents,
        optimization_metric=optimization_metric,
        ogx_vector_io_provider_id=vector_io_provider_id,
        inference_max_threads=inference_max_threads,
        evaluators=evaluators,
    )

    # --- Run the optimization loop ---
    rag_exp.search()

    # --- Generate output artefacts ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = _generate_output_artifacts(
        patterns_raw=event_handler.patterns,
        output_dir=output_dir,
        input_data_key=input_data_key,
        test_data_key=test_data_key,
        indexing_pipeline_params=indexing_pipeline_params,
    )

    return OptimizationResult(
        patterns=patterns,
        evaluations=list(rag_exp.results.evaluations),
    )


def _generate_output_artifacts(
    patterns_raw: list[dict],
    output_dir: Path,
    input_data_key: str,
    test_data_key: str,
    indexing_pipeline_params: dict | None,
) -> list[dict]:
    """Write per-pattern artefacts (JSON, notebooks, evaluation results)."""
    ogx_base_url = (os.environ.get("OGX_CLIENT_BASE_URL") or "").strip()
    patterns: list[dict] = []

    for pattern in patterns_raw:
        patt_dir = output_dir / pattern.get("payload").get("name")
        patt_dir.mkdir(parents=True, exist_ok=True)

        pattern_data = build_pattern_json(
            pattern=pattern.get("payload"),
            indexing_pipeline_params=indexing_pipeline_params,
        )

        generate_notebook_from_template(
            "ogx_indexing",
            pattern_data,
            patt_dir / "indexing.ipynb",
            input_data_key=input_data_key,
            ogx_base_url=ogx_base_url,
        )
        generate_notebook_from_template(
            "ogx_inference",
            pattern_data,
            patt_dir / "inference.ipynb",
            test_data_key=test_data_key,
            ogx_base_url=ogx_base_url,
        )

        with (patt_dir / "pattern.json").open("w", encoding="utf-8") as f:
            json_dump(pattern_data, f, indent=2)

        with (patt_dir / "evaluation_results.json").open("w", encoding="utf-8") as f:
            json_dump(pattern.get("evaluation_results", []), f, indent=2)

        patterns.append(pattern_data)

    return patterns


def _deserialize_model(data: dict[str, Any], ogx_client: OgxClient) -> OGXEmbeddingModel | OGXFoundationModel:
    """Reconstruct a model instance from its serialized dictionary.

    Parameters
    ----------
    data
        Dictionary produced by :func:`_serialize_model` in the search-space
        preparation step.
    ogx_client
        Client bound to the reconstructed model instance.
    """
    model_id = data["model_id"]
    params = data.get("params", {})

    if data["type"] == "embedding":
        return OGXEmbeddingModel(client=ogx_client, model_id=model_id, params=params)

    language = Language(**data["language"]) if data.get("language") else None
    return OGXFoundationModel(
        client=ogx_client,
        model_id=model_id,
        params=params,
        language=language,
        system_message_text=data.get("system_message_text"),
        user_message_text=data.get("user_message_text"),
        context_template_text=data.get("context_template_text"),
    )


def _evaluation_result_fallback(eval_data_list: list, evaluation_result: Any) -> list[dict[str, Any]]:
    """Build ``evaluation_results.json``-style list when ``question_scores`` is missing or incomplete.

    This is a safety net for older experiment results that may not contain
    per-question score breakdowns.
    """
    question_scores = (evaluation_result.scores or {}).get("question_scores") or []
    scores_by_qid = {q["question_id"]: q["metrics"] for q in question_scores if isinstance(q, dict)}

    out: list[dict[str, Any]] = []
    for ev in eval_data_list:
        answer_contexts: list[dict[str, str]] = []
        if getattr(ev, "contexts", None) and getattr(ev, "context_ids", None):
            answer_contexts = [{"text": t, "document_id": doc_id} for t, doc_id in zip(ev.contexts, ev.context_ids)]
        qid = getattr(ev, "question_id", None)
        metrics = [
            {"name": m["name"], "evaluator": m["evaluator"], "score": m["value"]} for m in scores_by_qid.get(qid, [])
        ]
        out.append(
            {
                "question": getattr(ev, "question", ""),
                "correct_answers": getattr(ev, "ground_truths", None),
                "answer": getattr(ev, "answer", ""),
                "answer_contexts": answer_contexts,
                "metrics": metrics,
            }
        )
    return out


def _validate_optimization_settings(optimization_settings: dict | None) -> dict:
    """Validate and normalize optimization settings.

    Returns
    -------
    dict
        Validated settings dictionary (empty dict when input is ``None``).

    Raises
    ------
    TypeError
        If settings or ``max_number_of_rag_patterns`` have wrong types.
    ValueError
        If ``max_number_of_rag_patterns`` is out of the allowed range or
        cannot be parsed as an integer.
    """
    if optimization_settings is None:
        return {}

    if not isinstance(optimization_settings, dict):
        raise TypeError("optimization_settings must be a dictionary.")

    max_rag_patterns = optimization_settings.get("max_number_of_rag_patterns", DEFAULT_MAX_RAG_PATTERNS)
    if isinstance(max_rag_patterns, str):
        try:
            max_rag_patterns = int(max_rag_patterns.strip())
        except ValueError as exc:
            raise ValueError(
                "optimization_settings.max_number_of_rag_patterns must be a valid integer "
                f"(e.g. from the pipeline UI); got {max_rag_patterns!r}."
            ) from exc

    if not isinstance(max_rag_patterns, int):
        raise TypeError("optimization_settings.max_number_of_rag_patterns must be an integer.")

    if not MIN_MAX_RAG_PATTERNS_RANGE[0] <= max_rag_patterns <= MIN_MAX_RAG_PATTERNS_RANGE[1]:
        raise ValueError(
            f"optimization_settings.max_number_of_rag_patterns must be in range "
            f"{MIN_MAX_RAG_PATTERNS_RANGE[0]} to {MIN_MAX_RAG_PATTERNS_RANGE[1]}."
        )

    return optimization_settings
