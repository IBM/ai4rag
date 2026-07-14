# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence

import pandas as pd
from docling_core.types.doc import DoclingDocument
from ogx_client import OgxClient

from ai4rag import logger
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.exception_handler import (
    AI4RAGError,
    AssetSaveError,
    ExperimentExceptionHandler,
    IndexingError,
    VectorStoreInitializationError,
)
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.core.experiment.results import EvaluationResult, ExperimentResults
from ai4rag.core.experiment.utils import (
    RAGExperimentError,
    RAGParamsType,
    build_evaluation_data,
    get_chunking_params,
    get_retrieval_params,
    merge_evaluation_results,
    query_rag,
)
from ai4rag.core.hpo.base_optimizer import BaseOptimizer, OptimizationError, OptimizerSettings
from ai4rag.core.hpo.gam_opt import GAMOptimizer
from ai4rag.core.hpo.random_opt import FailedIterationError
from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData, EvaluationMetricsResult
from ai4rag.evaluator.custom_metrics import apply_custom_metrics
from ai4rag.evaluator.metric import Metrics, RAGMetric
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.rag.chunking import DoclingChunker, LangChainChunker
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.template.simple_rag_template import SimpleRAG
from ai4rag.rag.vector_store.get_vector_store import get_vector_store
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames, ExperimentStep
from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel


# pylint: disable=too-many-instance-attributes
class AI4RAGExperiment:
    """
    Class responsible for conducting AutoRAG experiment, that consists of
    finding the best hyperparameters for several steps/stages.

    AI4RAGExperiment is essentially an orchestrator for the RAG Patterns
    hyperparameters optimization for the desired metric. It requires from
    user to provide fully defined search space on which the experiment will
    be executed.

    AI4RAG uses 'BaseRAGTemplate' inheriting classes as definitions on how
    to build and utilize RAG Pattern with the given search space nodes.

    Parameters
    ----------
    documents : list[DoclingDocument]
        List of parsed docling documents to embed in vector db and use as context in RAG.

    benchmark_data : pd.DataFrame | BenchmarkData
        Structure with 3 columns: 'question', 'correct_answers' and - if applicable - 'correct_answer_document_ids'.

    search_space : AI4RAGSearchSpace
        Grid of parameters used during hyperparameter optimization.

    vector_store_type : str
        Specific type of Vector Data Base that will be used during the experiment.
        Supported values: ``"ogx"`` and ``"chroma"``.

    ogx_vector_io_provider_id : str | None
        Provider ID for OGX vector store (e.g., ``"milvus"``, ``"qdrant"``).
        Required when ``vector_store_type="ogx"``.

    optimizer_settings : OptimizerSettings
        Settings for the optimizer to be used during the experiment.

    client : OgxClient | Any
        Instance of the OGX client or other client allowing to communicate
        with the available vector store providers.

    event_handler : BaseEventHandler
        Instance satisfying BaseEventHandler's interface to stream pattern evaluation
        results and intermediate status updates. EventHandler is an entrypoint to configure
        custom logging and assets handling.

    optimization_metric : RAGMetric | str, default=Metrics.FAITHFULNESS
        Metric used for calculating the final score that drives optimization.

    Other Parameters
    ----------------
    metrics : Sequence[RAGMetric]
        Metrics evaluated during the AutoRAG experiment. Not all
        of these metrics are used to calculate the final score, but they
        are included in the evaluation results. When omitted, defaults
        are derived from the configured evaluators.

    evaluators : list[BaseEvaluator] | None, default=None
        Evaluator instances used to score RAG patterns during optimization.
        When ``None``, defaults to ``[UnitxtEvaluator()]``.  To enable
        LLM-as-a-Judge evaluation, pass both a ``UnitxtEvaluator`` and a
        ``LLMaJEvaluator`` configured with a judge model.

    n_mps_foundation_models : int, default=3
        Amount of foundation models to be further used in experiment post pre-selection.

    n_mps_embedding_models : int, default=2
        Amount of embedding models to be further used in experiment post pre-selection.

    inference_max_threads : int, default=10
        Defines the number of threads to use during generation model inference.

    Attributes
    ----------
    results : ExperimentResults
        Instance holding information about each iteration during the experiment.
        It consists of statuses, RAG pattern objects, scores and settings.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        documents: list[DoclingDocument],
        benchmark_data: pd.DataFrame,
        search_space: AI4RAGSearchSpace,
        vector_store_type: str,
        optimizer_settings: OptimizerSettings,
        event_handler: BaseEventHandler,
        client: OgxClient | Any = None,
        ogx_vector_io_provider_id: str | None = None,
        optimization_metric: RAGMetric | str = Metrics.OVERALL_SCORE,
        **kwargs,
    ):
        self.documents = documents
        self.benchmark_data = BenchmarkData(benchmark_data)
        self.search_space = search_space
        self.vector_store_type = vector_store_type
        self.ogx_vector_io_provider_id = ogx_vector_io_provider_id
        self.optimizer_settings = optimizer_settings
        self.event_handler = event_handler
        self.client = client
        self.optimization_metric = optimization_metric

        self.evaluators: list[BaseEvaluator] = kwargs.pop("evaluators", None)
        self.metrics: Sequence[RAGMetric | str] | None = kwargs.pop(
            "metrics", None
        )  # resolved in _resolve_metrics_and_validate
        self.n_mps_foundation_models = kwargs.pop(
            "n_mps_foundation_models", ModelsPreSelector.DEFAULT_N_FOUNDATION_MODELS
        )
        self.n_mps_embedding_models = kwargs.pop("n_mps_embedding_models", ModelsPreSelector.DEFAULT_N_EMBEDDING_MODELS)
        self.known_observations: list[dict] | None = kwargs.pop("known_observations", None)
        self.inference_max_threads: int = kwargs.pop("inference_max_threads", 10)

        self.results: ExperimentResults = ExperimentResults()
        self._exception_handler = ExperimentExceptionHandler(self.event_handler)

        if kwargs:
            logger.warning("Unknown parameters: %s", kwargs)

        self._resolve_metrics_and_validate()

    @property
    def documents(self) -> list[DoclingDocument]:
        """Get list of documents."""
        return self._documents

    @documents.setter
    def documents(self, docs: list[DoclingDocument] | None) -> None:
        """
        Validate and set documents value.
        All documents must be ``DoclingDocument`` instances.
        """
        proper_docs = []
        if docs:
            for idx, doc in enumerate(docs):
                if isinstance(doc, DoclingDocument):
                    if not doc.name:
                        logger.warning("Document at index %s has no name set.", idx)
                    proper_docs.append(doc)
                else:
                    raise ValueError(f"Incorrect type of document provided at index: {idx}. Expected DoclingDocument.")

        self._documents = proper_docs

    @property
    def optimization_metric(self) -> RAGMetric:
        """Get optimization metrics used for the experiment."""
        return self._optimization_metric

    @optimization_metric.setter
    def optimization_metric(self, val: RAGMetric | str) -> None:
        """Validate and set optimization metrics"""
        available_metrics = [m.name for m in Metrics]

        if isinstance(val, str):
            n_val = next((metric for metric in Metrics if metric.name == val), None)
            val_name = val
        elif isinstance(val, RAGMetric):
            n_val = val if val in Metrics else None
            val_name = val.name
        else:
            raise RAGExperimentError(
                f"Incorrect type for optimization metric: {val}. Expected ai4rag.evaluator.metric.RAGMetric or str."
            )

        if not n_val:
            raise RAGExperimentError(
                f"Provided optimization metric: '{val_name}' is not supported. Available metrics: {available_metrics}."
            )

        self._optimization_metric = n_val

    @property
    def benchmark_data(self) -> BenchmarkData:
        """Get benchmark data."""
        return self._benchmark_data

    @benchmark_data.setter
    def benchmark_data(self, val: BenchmarkData) -> None:
        """Check and set benchmark data based on the executed scenario."""
        self._benchmark_data = val

    @property
    def evaluators(self) -> list[BaseEvaluator]:
        """Get experiment evaluators."""
        return self._evaluators

    @evaluators.setter
    def evaluators(self, val: list[BaseEvaluator] | None) -> None:
        """Validate and set experiment evaluators."""
        if val is None:
            self._evaluators = [UnitxtEvaluator()]
            logger.info("No evaluators provided; defaulting to UnitxtEvaluator only.")
        else:
            if not all(isinstance(e, BaseEvaluator) for e in val):
                raise ValueError("All evaluators must be BaseEvaluator instances.")
            self._evaluators = list(val)

    @property
    def metrics(self) -> Sequence[RAGMetric]:
        """Get evaluation metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, val: Sequence[RAGMetric | str] | None) -> None:
        """Validate and set evaluation metrics.

        Accepts ``None`` (resolved later by ``_resolve_metrics_and_validate``),
        a sequence of ``RAGMetric`` instances, a sequence of metric name
        strings, or a mixed sequence of both.
        """
        if val is None:
            self._metrics = None
            return

        available = {m.name: m for m in Metrics}
        resolved: list[RAGMetric] = []

        for item in val:
            if isinstance(item, RAGMetric):
                if item not in Metrics:
                    raise ValueError(f"Unknown RAGMetric '{item.name}'. Available: {list(available.keys())}.")
                resolved.append(item)
            elif isinstance(item, str):
                metric = available.get(item)
                if metric is None:
                    raise ValueError(f"Unknown metric name '{item}'. Available: {list(available.keys())}.")
                resolved.append(metric)
            else:
                raise TypeError(f"Each metric must be a RAGMetric or str, got {type(item).__name__}.")

        if not resolved:
            raise ValueError("Metrics sequence must not be empty.")

        self._metrics = tuple(resolved)

    def _resolve_metrics_and_validate(self) -> None:
        """Derive default metrics from configured evaluators and validate coverage.

        Called once at the end of ``__init__``.  When no explicit metrics
        were provided, the default set is derived from the evaluators that
        are already configured at this point.  Regardless of how metrics
        were set, the optimization metric is checked against the available
        evaluator types.
        """
        if self.metrics is None:
            base: list[RAGMetric] = [
                Metrics.ANSWER_CORRECTNESS,
                Metrics.FAITHFULNESS,
                Metrics.CONTEXT_CORRECTNESS,
                Metrics.OVERALL_SCORE,
            ]
            evaluator_types = {e.EVALUATOR_TYPE for e in self._evaluators}
            if "judge" in evaluator_types:
                base.append(Metrics.JUDGE_ANSWER_RELEVANCE)
            self._metrics = tuple(base)
            logger.info("No metrics provided; defaulting to %s.", [m.name for m in self._metrics])

        evaluator_types = {e.EVALUATOR_TYPE for e in self._evaluators}
        opt = self.optimization_metric
        if opt.evaluator not in ("custom", *evaluator_types):
            raise ValueError(
                f"Optimization metric '{opt.name}' requires a '{opt.evaluator}' evaluator, "
                f"but only {evaluator_types} are configured. "
                f"Pass an evaluator with EVALUATOR_TYPE='{opt.evaluator}' in the evaluators list."
            )

    def run_pre_selection(
        self,
        foundation_models: list[BaseFoundationModel],
        embedding_models: list[BaseEmbeddingModel],
        n_records: int = 5,
        random_seed: int = 17,
    ) -> dict[str, list[BaseEmbeddingModel | BaseFoundationModel]]:
        """
        Run models pre-selection using ModelsPreSelector and sample
        of the data.

        Parameters
        ----------
        embedding_models : list[BaseEmbeddingModel]
            Embedding models to be considered during pre-selection process.

        foundation_models : list[BaseFoundationModel]
            Foundation models to be evaluated during pre-selection process.

        n_records : int, default=5
            Amount of records that should be used during models pre-selection.

        random_seed : int, default=17
            Random seed value used for sampling benchmark data records.

        Returns
        -------
        dict[str, list[BaseFoundationModel | EmbeddingModel]]
            Best embedding models and foundation models found in pre-selection.
        """
        _log_start_mps = (
            f"Starting foundation models pre-selection with following "
            f"foundation models: {[str(fm) for fm in foundation_models]} "
            f"and following embedding models: {[str(em) for em in embedding_models]}."
        )
        logger.info(_log_start_mps)
        self.event_handler.on_status_change(
            level=LogLevel.INFO,
            message=_log_start_mps,
            step=ExperimentStep.MODEL_SELECTION,
        )

        mps = ModelsPreSelector(
            benchmark_data=self.benchmark_data.get_random_sample(n_records=n_records, random_seed=random_seed),
            documents=self.documents.copy(),
            foundation_models=foundation_models,
            embedding_models=embedding_models,
            metric=Metrics.OVERALL_SCORE,
        )
        mps.evaluate_patterns()

        selected_models = mps.select_models(
            n_embedding_models=self.n_mps_embedding_models, n_foundation_models=self.n_mps_foundation_models
        )

        logger.info(
            "Models pre-selection has been finished. Selected foundation models: %s and selected embedding models: %s.",
            [str(model) for model in selected_models["foundation_models"]],
            [str(model) for model in selected_models["embedding_models"]],
        )

        return selected_models

    # pylint: disable=too-many-locals, too-many-statements
    def run_single_evaluation(self, rag_params: RAGParamsType) -> float:
        """
        Evaluate a single RAG configuration and return its score using provided documents.

        Parameters
        ----------
        rag_params : RAGParamsType
            A dictionary containing rag parameters as keys and their values.

        Returns
        -------
        float
            A single evaluation score obtained by the executed rag pattern.
        """
        start_time = time.time()

        chunking_params = get_chunking_params(rag_params)
        retrieval_params = get_retrieval_params(rag_params)

        foundation_model = rag_params.get(AI4RAGParamNames.FOUNDATION_MODEL)
        embedding_model = rag_params.get(AI4RAGParamNames.EMBEDDING_MODEL)

        embedding_params_dict = (
            asdict(embedding_model.params) if is_dataclass(embedding_model.params) else embedding_model.params
        )
        indexing_params = {
            "chunking": chunking_params,
            "embedding": {
                "model_id": embedding_model.model_id,
                "embedding_params": embedding_params_dict,
            },
        }

        logger.info("Using indexing params: %s", indexing_params)

        retrieval_method = retrieval_params[AI4RAGParamNames.RETRIEVAL_METHOD]
        number_of_chunks = retrieval_params[AI4RAGParamNames.NUMBER_OF_CHUNKS]

        search_mode = retrieval_params.get(AI4RAGParamNames.SEARCH_MODE, "vector")
        if search_mode != "vector" and self.vector_store_type == "chroma":
            raise RAGExperimentError(
                f"Search mode '{search_mode}' is not supported with chroma vector store. "
                "Only 'vector' mode is supported for chroma."
            )

        context_template_text = foundation_model.context_template_text
        system_message_text = foundation_model.system_message_text
        user_message_text = foundation_model.user_message_text

        rag_params = {
            "retrieval": retrieval_params,
            "generation": {
                "model_id": foundation_model.model_id,
                "temperature": foundation_model.params.temperature,
                "max_completion_tokens": foundation_model.params.max_completion_tokens,
                "context_template_text": context_template_text,
                "user_message_text": user_message_text,
                "system_message_text": system_message_text,
                "language": foundation_model.language.to_dict(),
            },
        }

        logger.info("Using retrieval and generation params: %s", rag_params)

        result_score = self.results.evaluation_explored_or_cached(
            indexing_params=indexing_params, rag_params=rag_params
        )
        if result_score is not None:
            return result_score

        pattern_name = self._create_pattern_name()
        logger.info("Using name '%s' for the currently evaluated pattern.", pattern_name)

        reuse_collection_name = self._get_reusable_collection_name(indexing_params=indexing_params)

        try:
            vector_store = get_vector_store(
                vs_type=self.vector_store_type,
                embedding_model=embedding_model,
                reuse_collection_name=reuse_collection_name,
                client=self.client,
                ogx_vector_io_provider_id=self.ogx_vector_io_provider_id,
            )
        except Exception as exc:
            raise VectorStoreInitializationError(
                exc,
                embedding_model_id=embedding_model.model_id,
                vector_store_provider_id=self.ogx_vector_io_provider_id or "local_chroma",
            ) from exc

        collection_name = vector_store.collection_name

        if not self._collection_exists(collection_name=collection_name):
            chunking_method = chunking_params.get(AI4RAGParamNames.CHUNKING_METHOD)
            chunk_size = chunking_params.get(AI4RAGParamNames.CHUNK_SIZE)
            chunk_overlap = chunking_params.get(AI4RAGParamNames.CHUNK_OVERLAP)

            if chunking_method == "hybrid":
                chunker = DoclingChunker(max_tokens=chunk_size)
            else:
                chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunked_documents = chunker.split_documents(self.documents)

            if self.event_handler:
                self.event_handler.on_status_change(
                    level=LogLevel.INFO,
                    message=(
                        f"Chunking documents using the {chunking_method} method, chunk_size: {chunk_size} "
                        f"and chunk_overlap: {chunk_overlap}."
                    ),
                    step=ExperimentStep.CHUNKING,
                )

            self.event_handler.on_status_change(
                level=LogLevel.INFO,
                message=(
                    f"Embedding chunks using the {embedding_model.model_id} model. "
                    f"Building index: {collection_name}."
                ),
                step=ExperimentStep.EMBEDDING,
            )

            try:
                vector_store.add_documents(chunked_documents)
            except Exception as exc:
                raise IndexingError(exc, collection_name, embedding_model.model_id) from exc

        else:
            self.event_handler.on_status_change(
                level=LogLevel.INFO,
                message=f"Using index {collection_name}.",
                step=ExperimentStep.EMBEDDING,
            )

        logger.info("Using retriever with parameters: %s", retrieval_params)

        retriever = Retriever(
            vector_store=vector_store,
            number_of_chunks=number_of_chunks,
            method=retrieval_method,
            search_mode=search_mode,
            ranker_strategy=retrieval_params.get(AI4RAGParamNames.RANKER_STRATEGY),
            ranker_k=retrieval_params.get(AI4RAGParamNames.RANKER_K),
            ranker_alpha=retrieval_params.get(AI4RAGParamNames.RANKER_ALPHA),
        )

        rag_pattern = SimpleRAG(
            foundation_model=foundation_model,
            retriever=retriever,
        )

        _rag_log = (
            f"Retrieval and generation using collection: '{collection_name}' and "
            f"foundation model: '{foundation_model.model_id}'."
        )
        logger.info(_rag_log)
        self.event_handler.on_status_change(
            level=LogLevel.INFO,
            message=_rag_log,
            step=ExperimentStep.GENERATION,
        )

        inference_response = query_rag(
            rag=rag_pattern, questions=list(self.benchmark_data.questions), max_threads=self.inference_max_threads
        )
        result_scores, evaluation_data = self._evaluate_response(
            inference_response=inference_response,
            pattern_name=pattern_name,
        )

        stop_time = time.time()
        execution_time = stop_time - start_time

        final_score = next(
            (r["scores"]["mean"] for r in result_scores["metrics"] if r["name"] == self.optimization_metric.name),
            None,
        )
        if final_score is None:
            raise RAGExperimentError(
                f"Optimization metric '{self.optimization_metric.name}' not found in evaluation results. "
                f"Available: {[m['name'] for m in result_scores['metrics']]}."
            )

        logger.info("Calculated optimization score for '%s': %s", pattern_name, final_score)

        evaluation_result = EvaluationResult(
            pattern_name=pattern_name,
            collection=collection_name,
            indexing_params=indexing_params,
            rag_params=rag_params,
            scores=result_scores,
            execution_time=execution_time,
            final_score=final_score,
            rag_pattern=rag_pattern,
        )

        evaluation_results_json = self.results.create_evaluation_results_json(
            evaluation_data=evaluation_data, evaluation_result=evaluation_result
        )

        logger.info(
            "Evaluation scores: %s",
            {el.get("question"): el.get("metrics") for el in evaluation_results_json if isinstance(el, dict)},
        )

        try:
            self._stream_finished_pattern(
                evaluation_result=evaluation_result,
                evaluation_results_json=evaluation_results_json,
            )
        except Exception as exc:
            raise AssetSaveError(exc) from exc

        self.results.add_evaluation(
            evaluation_data=evaluation_data,
            evaluation_result=evaluation_result,
        )

        return final_score

    def search(self, **kwargs) -> None:
        """
        Prepare and execute experiment to find the best RAG parameters.

        Result of the search() can be reviewed via self.results as this object
        stores results of each evaluation or via self.event_handler with custom
        implementation.
        """

        logger.info("Starting RAG optimization process...")

        def objective_function(space: RAGParamsType) -> float | None:
            """Function passed to the optimizer."""
            try:
                return self.run_single_evaluation(space)
            except AI4RAGError as err:
                msg = self._exception_handler.handle_exception(err)
                raise FailedIterationError(msg) from err

        # MPS - models pre-selection based on sample evaluation.
        # Run if there are more than 3 foundation models or more than 2 embedding models.
        foundation_models = list(self.search_space[AI4RAGParamNames.FOUNDATION_MODEL].values)
        embedding_models = list(self.search_space[AI4RAGParamNames.EMBEDDING_MODEL].values)

        if (
            len(embedding_models) > self.n_mps_embedding_models or len(foundation_models) > self.n_mps_foundation_models
        ) and not kwargs.get("skip_mps", False):
            selected_models = self.run_pre_selection(
                foundation_models=foundation_models, embedding_models=embedding_models
            )
            self.search_space[AI4RAGParamNames.FOUNDATION_MODEL] = Parameter(
                name=AI4RAGParamNames.FOUNDATION_MODEL, param_type="C", values=selected_models["foundation_models"]
            )
            self.search_space[AI4RAGParamNames.EMBEDDING_MODEL] = Parameter(
                name=AI4RAGParamNames.EMBEDDING_MODEL, param_type="C", values=selected_models["embedding_models"]
            )

        optimizer_class: type[BaseOptimizer] = kwargs.get("optimizer", GAMOptimizer)

        optimizer_kwargs = {}
        if self.known_observations is not None:
            optimizer_kwargs["known_observations"] = self.known_observations

        # In the search kwargs user may pass different optimizer class for testing purposes
        optimizer = optimizer_class(
            objective_function=objective_function,
            search_space=self.search_space,
            settings=self.optimizer_settings,
            **optimizer_kwargs,
        )
        logger.info(
            "Using optimizer: %s with optimizer settings: %s",
            optimizer_class.__name__,
            self.optimizer_settings.to_dict(),
        )

        try:
            _ = optimizer.search()
        except OptimizationError as err:
            final_error_msg = self._exception_handler.get_final_error_msg()
            raise RAGExperimentError(final_error_msg) from err

        self.event_handler.on_status_change(
            level=LogLevel.INFO,
            message="Experiment optimization process finished.",
        )

    def _stream_finished_pattern(
        self,
        evaluation_result: EvaluationResult,
        evaluation_results_json: list,
    ) -> None:
        """
        Stream finished pattern.

        Parameters
        ----------
        evaluation_result : EvaluationResult
            Data made of evaluation results.

        evaluation_results_json : list
            Prepared partial payload for the streamed content.
        """
        retrieval_payload = {
            "method": evaluation_result.rag_params["retrieval"][AI4RAGParamNames.RETRIEVAL_METHOD],
            "number_of_chunks": evaluation_result.rag_params["retrieval"][AI4RAGParamNames.NUMBER_OF_CHUNKS],
            "search_mode": evaluation_result.rag_params["retrieval"].get(AI4RAGParamNames.SEARCH_MODE, "vector"),
        }

        if evaluation_result.rag_params["retrieval"][AI4RAGParamNames.WINDOW_SIZE]:
            retrieval_payload["window_size"] = evaluation_result.rag_params["retrieval"][AI4RAGParamNames.WINDOW_SIZE]

        if retrieval_payload["search_mode"] == "hybrid":
            ranker_strategy = evaluation_result.rag_params["retrieval"].get(AI4RAGParamNames.RANKER_STRATEGY)
            retrieval_payload["ranker_strategy"] = ranker_strategy

            if ranker_strategy == "rrf":
                retrieval_payload["ranker_k"] = evaluation_result.rag_params["retrieval"].get(AI4RAGParamNames.RANKER_K)

            if ranker_strategy == "weighted":
                retrieval_payload["ranker_alpha"] = evaluation_result.rag_params["retrieval"].get(
                    AI4RAGParamNames.RANKER_ALPHA
                )

        vector_store_payload = {
            "provider_id": self.ogx_vector_io_provider_id or "local_chroma",
            "vector_store_id": evaluation_result.collection,
        }
        if self.vector_store_type == "ogx":
            try:
                provider = self.client.providers.retrieve(self.ogx_vector_io_provider_id)
                provider_type = getattr(provider, "provider_type", "unknown")
            except Exception as exc:
                provider_type = "unknown"
                logger.warning(
                    "Could not retrieve provider_type attribute of vector store in use...",
                    exc_info=exc,
                )
            vector_store_payload["provider_type"] = provider_type

        indexing_payload = {
            "chunking": {
                "method": evaluation_result.indexing_params["chunking"][AI4RAGParamNames.CHUNKING_METHOD],
                "chunk_size": evaluation_result.indexing_params["chunking"][AI4RAGParamNames.CHUNK_SIZE],
                "chunk_overlap": evaluation_result.indexing_params["chunking"][AI4RAGParamNames.CHUNK_OVERLAP],
            },
            "embedding": evaluation_result.indexing_params.get("embedding"),
        }

        generation_payload = evaluation_result.rag_params.get("generation")

        n_known = len(self.known_observations) if self.known_observations else 0

        metrics_payload = [
            {**m, "optimization_metric": True} if m["name"] == self.optimization_metric.name else m
            for m in evaluation_result.scores["metrics"]
        ]

        payload = {
            "name": evaluation_result.pattern_name,
            "max_combinations": self.search_space.max_combinations,
            "evaluation": {"metrics": metrics_payload},
            "duration_seconds": int(evaluation_result.execution_time),
            "settings": {
                "vector_store_binding": vector_store_payload,
                **indexing_payload,
                "retrieval": retrieval_payload,
                "generation": generation_payload,
            },
            "iteration": len(self.results) + n_known,
        }

        self.event_handler.on_pattern_creation(
            payload=payload,
            evaluation_results=evaluation_results_json,
        )

    def _evaluate_response(
        self,
        inference_response: list[dict[str, Any]],
        pattern_name: str,
    ) -> tuple[EvaluationMetricsResult, list[EvaluationData]]:
        """
        Evaluate response using all configured evaluators and merge results.

        Each evaluator receives only the metrics matching its
        ``EVALUATOR_TYPE``.  Results are merged into a single
        ``EvaluationMetricsResult`` before custom metrics (e.g.
        ``overall_score``) are computed on top.

        Parameters
        ----------
        inference_response : list[dict[str, Any]]
            List of model's responses containing question, answer and used
            reference documents for each record.

        pattern_name : str
            Name of the pattern for which evaluation is performed.

        Returns
        -------
        tuple[EvaluationMetricsResult, list[EvaluationData]]
            Combined evaluation scores and input evaluation data.
        """
        evaluator_names = [e.__class__.__name__ for e in self.evaluators]
        logger.info("Evaluating RAG Pattern '%s' using %s.", pattern_name, evaluator_names)
        self.event_handler.on_status_change(
            level=LogLevel.INFO,
            message=f"Evaluating RAG Pattern '{pattern_name}' using {evaluator_names}.",
            step="evaluation",
        )

        eval_data = build_evaluation_data(benchmark_data=self.benchmark_data, inference_response=inference_response)

        evaluator_map: dict[str, BaseEvaluator] = {e.EVALUATOR_TYPE: e for e in self.evaluators}

        metrics_by_type: dict[str, list[RAGMetric]] = {}
        for m in self.metrics:
            if m.evaluator != "custom":
                metrics_by_type.setdefault(m.evaluator, []).append(m)

        partial_results: list[EvaluationMetricsResult] = []
        for eval_type, type_metrics in metrics_by_type.items():
            evaluator = evaluator_map.get(eval_type)
            if evaluator is None:
                logger.debug(
                    "No evaluator registered for type '%s'; skipping metrics %s.",
                    eval_type,
                    [m.name for m in type_metrics],
                )
                continue
            partial_results.append(evaluator.evaluate_metrics(evaluation_data=eval_data, metrics=type_metrics))

        result = merge_evaluation_results(partial_results)
        apply_custom_metrics(scores=result, metrics=self.metrics)

        logger.info("Evaluation results for '%s': %s.", pattern_name, result)
        return result, eval_data

    def _collection_exists(self, collection_name: str) -> bool:
        """
        This method checks if a collection with a given name already exists.
        The trick comes with chromadb. We always need to assume that collection
        does not exist, as we create new instance of chroma in memory per each
        run.

        Parameters
        ----------
        collection_name : str
            Name of the collection to check if exists.

        Returns
        -------
        bool
            True if collection exist, otherwise False.
        """
        return collection_name in self.results.collection_names

    def _get_reusable_collection_name(self, indexing_params: dict[str, Any]) -> str | None:
        """
        This method returns name of the collection / vector_store_id (for OGX)
        if chosen indexing params have already been used to create an index / collection.

        Parameters
        ----------
        indexing_params : dict[str, Any]
            Dictionary containing keys and values that are compared with
            previously used ones, to establish if the newly created collection
            would be exactly the same.

        Returns
        -------
        str | None
            Collection name that is new or one of the previously created.
            None if there is no collection to reuse.
        """
        collection = self.results.get_existing_collection(indexing_params=indexing_params)
        if collection is not None:
            collection_name = collection
            logger.info("Reusing existing collection: '%s'", collection_name)
            return collection_name

        return None

    def _create_pattern_name(self) -> str:
        """
        Create pattern name based on the already existing patterns and length
        of results, as we iterate patterns from 1 to n.

        Returns
        -------
        str
            Pattern name.
            Example: "Pattern7"
        """
        n_known = len(self.known_observations) if self.known_observations else 0
        return f"Pattern{n_known + len(self.results) + 1}"
