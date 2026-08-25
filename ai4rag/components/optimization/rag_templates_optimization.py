# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import logging
from dataclasses import dataclass
from json import dump as json_dump
from pathlib import Path
from typing import Any, Literal, get_args

import pandas as pd
from openai import OpenAI

from ai4rag import handler
from ai4rag.components.assets_generator import generate_notebook_from_template
from ai4rag.components.utils.docling_io import load_docling_documents
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.gam_opt import GAMOptSettings
from ai4rag.evaluator.judge_selection import select_judge_model
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from ai4rag.evaluator.metric import Metrics, RAGMetric
from ai4rag.evaluator.ragas_evaluator import RagasEvaluator
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.vector_store.config import ChromaConfig, MilvusConfig, PGVectorConfig
from ai4rag.search_space.prepare.models import get_embedding_models, get_foundation_models
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.event_handler.event_handler import KFPEventHandler

_logger = logging.getLogger("rag-templates-optimization")
_logger.addHandler(handler)

DEFAULT_MAX_RAG_PATTERNS = 8
MIN_MAX_RAG_PATTERNS_RANGE = (4, 20)

# LLM-as-a-judge evaluator selection for the optimization run. The
# reference-based ``UnitxtEvaluator`` always runs; this only controls the
# LLM-as-a-judge evaluators:
#   "base"  -> in-house LLM judge (LLMaJEvaluator)
#   "ragas" -> RagasEvaluator (RAGAS LLM-as-a-judge metrics)
#   "all"  -> judge + ragas
#   "none"  -> Unitxt only
LLMJudgeMode = Literal["base", "ragas", "all", "none"]
DEFAULT_LLM_JUDGE_MODE: LLMJudgeMode = "base"
DEFAULT_METRIC = Metrics.OVERALL_SCORE.name
# The optimization target is resolved to a concrete ``RAGMetric`` instance here.
# By assumption only the unitxt metrics (plus the custom ``overall_score``) drive
# optimization, so ambiguous names like "faithfulness" bind to the unitxt variant
# rather than the RAGAS one.
SUPPORTED_OPTIMIZATION_METRICS: dict[str, RAGMetric] = {
    Metrics.FAITHFULNESS.name: Metrics.FAITHFULNESS,
    Metrics.ANSWER_CORRECTNESS.name: Metrics.ANSWER_CORRECTNESS,
    Metrics.CONTEXT_CORRECTNESS.name: Metrics.CONTEXT_CORRECTNESS,
    Metrics.OVERALL_SCORE.name: Metrics.OVERALL_SCORE,
}


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
    maas_client: OpenAI,
    vector_store_config: ChromaConfig | MilvusConfig | PGVectorConfig,
    test_data_key: str = "",
    input_data_key: str = "",
    optimization_settings: dict | None = None,
    inference_max_threads: int = 10,
    indexing_pipeline_params: dict | None = None,
    llm_judge_mode: LLMJudgeMode = DEFAULT_LLM_JUDGE_MODE,
    n_random_nodes: int = 4,
    warm_start_strategy: Literal["mode_balanced", "model_mode_balanced"] = "mode_balanced",
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
    maas_client
        An authenticated OpenAI-compatible :class:`~openai.OpenAI` client shared
        by every model restored from the search-space report; it serves chat and
        embeddings for all of them.
    vector_store_config
        Connection config for the vector store backend. Its type (via
        ``config.provider``) determines whether Chroma, Milvus, or PGVector
        is used.
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
    llm_judge_mode : {"base", "ragas", "all", "none"}, default="base"
        Which LLM-as-a-judge evaluators to run in addition to the always-present
        reference-based ``UnitxtEvaluator``:

        - ``"base"``  — the in-house LLM judge (``answer_relevance``).
        - ``"ragas"`` — the RAGAS LLM-as-a-judge metrics (faithfulness, answer
          relevancy, context precision/recall).
        - ``"all"``  — the LLM judge and RAGAS together.
        - ``"none"``  — no LLM-as-a-judge evaluators; Unitxt metrics only.

        Any mode other than ``"none"`` requires at least one foundation model
        and one embedding model in the search space.
    n_random_nodes : int, default=4
        Number of random configurations to evaluate before starting GAM iterations.
        Passed directly to :class:`~ai4rag.core.hpo.gam_opt.GAMOptSettings`.
    warm_start_strategy : {"mode_balanced", "model_mode_balanced"}, default="mode_balanced"
        Controls how the initial random nodes are ordered across buckets.
        ``"mode_balanced"`` round-robins across ``search_mode`` values;
        ``"model_mode_balanced"`` round-robins across
        ``(foundation_model, embedding_model, search_mode)`` triples.
        Passed directly to :class:`~ai4rag.core.hpo.gam_opt.GAMOptSettings`.

    Returns
    -------
    OptimizationResult
        Contains the list of pattern definitions, raw evaluations, and the
        total number of parameter combinations explored.

    Raises
    ------
    ValueError
        If ``test_data_key`` does not point to a JSON file,
        ``llm_judge_mode`` is invalid, or the optimization metric is not
        supported.
    TypeError
        If ``optimization_settings`` has invalid types.
    """
    # --- Input validation ---
    valid_modes = list(get_args(LLMJudgeMode))
    if llm_judge_mode not in valid_modes:
        raise ValueError(f"llm_judge_mode {llm_judge_mode!r} is not supported. Select one of {valid_modes}.")

    if not isinstance(test_data_key, str) or not test_data_key.strip() or not test_data_key.lower().endswith(".json"):
        raise ValueError("test_data_key must point to a JSON file.")

    settings = _validate_optimization_settings(optimization_settings)
    optimization_metric_name = settings.get("metric") or DEFAULT_METRIC
    if optimization_metric_name not in SUPPORTED_OPTIMIZATION_METRICS:
        raise ValueError(
            f"Optimization metric {optimization_metric_name} is not supported. "
            f"Select one of {sorted(SUPPORTED_OPTIMIZATION_METRICS)}."
        )
    # The experiment expects a concrete RAGMetric instance, so resolve the
    # configured name to its (unitxt / custom) variant here.
    optimization_metric = SUPPORTED_OPTIMIZATION_METRICS[optimization_metric_name]

    documents = load_docling_documents(extracted_text_path)
    benchmark_data = pd.read_json(Path(test_data_path))
    benchmark_data_obj = BenchmarkData(benchmark_data)

    # --- Reconstruct search space from report ---
    with open(search_space_report_path, "r", encoding="utf-8") as f:
        search_space_raw: dict[str, Any] = json.load(f)

    # Restore the models exactly as selected during search-space preparation.
    # Each serialized spec carries its inference params, detected language and
    # prompts, all reused verbatim (validate=False); every model is bound to the
    # single shared `maas_client`.
    foundation_models: list[OpenAIFoundationModel] = get_foundation_models(
        maas_client, search_space_raw.get("foundation_model", []), validate=False
    )
    embedding_models: list[OpenAIEmbeddingModel] = get_embedding_models(
        maas_client, search_space_raw.get("embedding_model", []), validate=False
    )

    params: list[Parameter] = []
    for param_name, values in search_space_raw.items():
        if param_name == "foundation_model":
            values = foundation_models
        elif param_name == "embedding_model":
            values = embedding_models
        params.append(Parameter(param_name, "C", values=values))

    search_space = AI4RAGSearchSpace(params=params)

    evaluators = _build_evaluators(
        llm_judge_mode=llm_judge_mode,
        foundation_models=foundation_models,
        embedding_models=embedding_models,
        benchmark_data=benchmark_data_obj,
        documents=documents,
        inference_max_threads=inference_max_threads,
    )

    # --- Configure experiment ---
    max_rag_patterns = settings.get("max_number_of_rag_patterns", DEFAULT_MAX_RAG_PATTERNS)
    if isinstance(max_rag_patterns, str):
        max_rag_patterns = int(max_rag_patterns.strip())
    optimizer_settings = GAMOptSettings(
        max_evals=max_rag_patterns,
        n_random_nodes=n_random_nodes,
        warm_start_strategy=warm_start_strategy,
    )

    event_handler = KFPEventHandler()

    rag_exp = AI4RAGExperiment(
        event_handler=event_handler,
        optimizer_settings=optimizer_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_config=vector_store_config,
        documents=documents,
        optimization_metric=optimization_metric,
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


def _build_evaluators(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    llm_judge_mode: LLMJudgeMode,
    foundation_models: list[OpenAIFoundationModel],
    embedding_models: list[OpenAIEmbeddingModel],
    benchmark_data: BenchmarkData,
    documents: list,
    inference_max_threads: int,
) -> list:
    """Build the evaluator list for the given ``llm_judge_mode``.

    The reference-based :class:`UnitxtEvaluator` always runs; the mode only
    controls which LLM-as-a-judge evaluators are added on top (see
    :data:`LLMJudgeMode`).

    Parameters
    ----------
    llm_judge_mode : LLMJudgeMode
        One of ``"base"``, ``"ragas"``, ``"all"`` or ``"none"``.
    foundation_models : list[OpenAIFoundationModel]
        Foundation models from the search space; the first is used by RAGAS and
        the judge selection considers all of them.
    embedding_models : list[OpenAIEmbeddingModel]
        Embedding models from the search space; the first is used by RAGAS.
    benchmark_data : BenchmarkData
        Benchmark data used when selecting the judge model.
    documents : list
        Parsed documents used when selecting the judge model.
    inference_max_threads : int
        Concurrency used during judge-model selection.

    Returns
    -------
    list
        The configured evaluator instances.

    Raises
    ------
    ValueError
        If an LLM-based evaluator is requested but no foundation or embedding
        model is available.
    """
    use_judge = llm_judge_mode in ("base", "all")
    use_ragas = llm_judge_mode in ("ragas", "all")

    evaluators: list = [UnitxtEvaluator()]

    if (use_judge or use_ragas) and (not foundation_models or not embedding_models):
        raise ValueError(
            f"llm_judge_mode={llm_judge_mode!r} requires at least one foundation model and one embedding model."
        )

    if use_judge:
        judge_model = select_judge_model(
            generation_models=foundation_models,
            embedding_models=embedding_models,
            benchmark_data=benchmark_data,
            documents=documents,
            max_threads=inference_max_threads,
        )
        _logger.info("Judge model selected: %s", judge_model.model_id)
        evaluators.append(LLMaJEvaluator(model=judge_model))

    if use_ragas:
        # RAGAS runs as an independent cross-check on its own generation model
        # rather than reusing the selected judge model.
        ragas_model = foundation_models[0]
        evaluators.append(RagasEvaluator(model=ragas_model, embedding_model=embedding_models[0]))
        _logger.info("RAGAS evaluator enabled with model: %s", ragas_model.model_id)

    return evaluators


def _generate_output_artifacts(
    patterns_raw: list[dict],
    output_dir: Path,
    input_data_key: str,
    test_data_key: str,
    indexing_pipeline_params: dict | None,
) -> list[dict]:
    """Write per-pattern artefacts (JSON, notebooks, evaluation results)."""
    patterns: list[dict] = []

    for pattern in patterns_raw:
        patt_dir = output_dir / pattern.get("payload").get("name")
        patt_dir.mkdir(parents=True, exist_ok=True)

        pattern_data = pattern.get("payload")
        if indexing_pipeline_params:
            settings = pattern_data["settings"]
            vector_store_binding = settings["vector_store_binding"]
            pattern_data["indexing"] = {
                "pipeline_spec": {
                    "pipeline_name": indexing_pipeline_params.get("pipeline_name", "documents_indexing_pipeline"),
                    "parameters": {
                        "maas_secret_name": indexing_pipeline_params.get("maas_secret_name"),
                        "vector_db_secret_name": indexing_pipeline_params.get("vector_db_secret_name"),
                        "input_data_secret_name": indexing_pipeline_params.get("input_data_secret_name"),
                        "input_data_bucket_name": indexing_pipeline_params.get("input_data_bucket_name"),
                        "input_data_key": indexing_pipeline_params.get("input_data_key"),
                        "batch_size": indexing_pipeline_params.get("batch_size"),
                        "provider_type": vector_store_binding["provider_type"],
                        "collection_name": vector_store_binding["collection_name"],
                        "embedding_model_id": settings["embedding"]["model_id"],
                        "embedding_params": settings["embedding"]["embedding_params"],
                        "chunking_method": settings["chunking"]["method"],
                        "chunk_size": settings["chunking"]["chunk_size"],
                        "chunk_overlap": settings["chunking"]["chunk_overlap"],
                    },
                    "overrides_allowed": [
                        "input_data_secret_name",
                        "input_data_bucket_name",
                        "input_data_key",
                        "collection_name",
                        "batch_size",
                    ],
                }
            }

        generate_notebook_from_template(
            "maas_indexing",
            pattern_data,
            patt_dir / "indexing.ipynb",
            input_data_key=input_data_key,
        )
        generate_notebook_from_template(
            "maas_inference",
            pattern_data,
            patt_dir / "inference.ipynb",
            test_data_key=test_data_key,
        )

        with (patt_dir / "pattern.json").open("w", encoding="utf-8") as f:
            json_dump(pattern_data, f, indent=2, ensure_ascii=False)

        with (patt_dir / "evaluation_results.json").open("w", encoding="utf-8") as f:
            json_dump(pattern.get("evaluation_results", []), f, indent=2, ensure_ascii=False)

        patterns.append(pattern_data)

    return patterns


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
