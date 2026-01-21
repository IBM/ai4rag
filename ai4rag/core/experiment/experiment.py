# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import time
from datetime import datetime
from typing import Any, Sequence

import pandas as pd
from langchain_core.documents import Document
from llama_stack_client import LlamaStackClient

from ai4rag import logger
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.exception_handler import (
    AssetSaveError,
    AI4RAGError,
    ExperimentExceptionsHandler,
    IndexingError,
)
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.core.experiment.results import EvaluationResult, ExperimentResults
from ai4rag.core.experiment.utils import (
    RAGExperimentError,
    RAGParamsType,
    build_evaluation_data,
    get_chunking_params,
    get_retrieval_params,
    query_rag,
)
from ai4rag.core.hpo.base_optimiser import BaseOptimiser, OptimiserSettings, OptimisationError
from ai4rag.core.hpo.random_opt import FailedIterationError, RandomOptimiser
from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData, MetricType
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator
from ai4rag.rag.chunking import LangChainChunker
from ai4rag.rag.embedding.base_model import EmbeddingModel
from ai4rag.rag.foundation_models.base_model import FoundationModel
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.template.rag_template import LlamaStackRAG
from ai4rag.search_space.src.models import EmbeddingModels
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import (
    AI4RAGParamNames,
    EventsToReport,
    ExperimentStep,
)
from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel
from ai4rag.utils.experiment_monitor import ExperimentMonitor
from ai4rag.rag.vector_store.get_vector_store import get_vector_store


class AI4RAGExperiment:
    """
    Class responsible for conducting AutoRAG experiment, that consists of finding the best hyperparameters
    for several steps/stages. Based on client instance it should perform connections with instances like
    COS, VectorStore (external or internal) etc.

    Parameters
    ----------
    client : LlamaStackClient
        Instance of the llama stack client allowing to communicate
        with the llama stack server.

    documents : list[Document | tuple[str, str]]
        List of documents to embed in vector db and use as context in RAG.
        When given as list of langchain's Document instances, both content and document
        ids must be provided:
        Document(page_content=..., metadata={document_id: 'some_id'})
        When given as list of tuples it should be (content, document_id)

    benchmark_data : pd.DataFrame | BenchmarkData
        Structure with 3 columns: 'question', 'correct_answers' and - if applicable - 'correct_answer_document_ids'.

    vector_store_type : str
        Specific type of Vector Data Base that will be used during the experiment.

    optimiser_settings : OptimiserSettings
        Settings for the optimiser to be used during the experiment.

    event_handler : BaseEventHandler
        Instance satisfying BaseEventHandler's interface to stream information
        from the training.

    optimization_metric : str, default=MetricType.FAITHFULNESS
        Metrics that should be used for calculating final score value that will be minimized.
        This sequence should contain 1 value for first release.

    search_space : AI4RAGSearchSpace
        Grid of parameters used during hyperparameter optimisation.

    knowledge_base_references : KnowledgeBaseReferences
        Knowledge Base References (Vector Store or SQL Database) to conduct experiment on.
        It is used interchangeably with documents,
        if given chunking and embedding step is skipped as reference already has data.

    Other Parameters
    ----------------
    output_path : str
        Path to the directory where output files/artifacts should be stored

    embeddings_provider : Literal["watsonx"]
        Literal type of embeddings provider

    job_id : str
        Unique identifier for a job

    metrics : Sequence[str]
        Metrics that will be evaluated during AutoRAG experiment. Not all
        of these metrics will be used to calculate final score, but they will
        be included in the evaluation results.

    evaluator : BaseEvaluator
        An implementation of the BaseEvaluator class, that will be used by the AI4RAGExperiment
        To evaluate the RAG pattern performance and will be utilized during the optimization
        process.

    experiment_monitor_output_path : str, default=None
        Path to which json file with results form ExperimentMonitor instance is saved.
        If None, file is not saved

    n_mps_fm : int, default=3
        Amount of foundation models to be further used in experiment post pre-selection.

    n_mps_em : int, default=2
        Amount of embedding models to be further used in experiment post pre-selection.

    input_data_references : list[dict] | None
        Required for properly creating an AI service for _chroma_. It's an in-memory DB so the documents
        need to be inserted everytime the AI service code is executed.

    Attributes
    ----------
    documents : list[Document]
        Validated documents

    results : ExperimentResults
        Status of each run in the hyperparameter optimization.

    search_output : dict
        Data from the search() result.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes,too-many-lines
    def __init__(
        self,
        client: LlamaStackClient,
        event_handler: BaseEventHandler,
        optimiser_settings: OptimiserSettings,
        search_space: AI4RAGSearchSpace,
        benchmark_data: pd.DataFrame | BenchmarkData,
        vector_store_type: str,
        documents: list[Document] | None = None,
        optimization_metric: str = MetricType.FAITHFULNESS,
        **kwargs,
    ):
        self.client = client

        self.benchmark_data = BenchmarkData(benchmark_data)
        self.documents = documents
        self.vector_store_type = vector_store_type

        self.optimiser_settings = optimiser_settings
        self.search_space = search_space or AI4RAGSearchSpace()
        self.n_mps_fm = kwargs.pop("n_mps_fm", 3)
        self.n_mps_em = kwargs.pop("n_mps_em", 2)
        self.event_handler = event_handler
        self.search_output = None

        self.output_path: str | None = kwargs.pop("output_path", None)
        self.job_id = kwargs.pop("job_id", "a0b1c2d3-zxcv-asdf-qwer-poiulkjhmnbv").replace("-", "_")

        self.metrics: Sequence[str] = kwargs.pop(
            "metrics", (MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS, MetricType.CONTEXT_CORRECTNESS)
        )
        self.optimization_metric = optimization_metric

        self.evaluator: BaseEvaluator = kwargs.pop(
            "evaluator",
            UnitxtEvaluator(),
        )

        self.results: ExperimentResults = ExperimentResults()

        experiment_monitor_output_path = kwargs.pop("experiment_monitor_output_path", None)
        self.experiment_monitor = ExperimentMonitor(
            output_path=experiment_monitor_output_path,
        )

        self.exceptions_handler = ExperimentExceptionsHandler(self.event_handler)

        if kwargs:
            logger.warning("Unknown parameters: %s", kwargs)

    @property
    def documents(self) -> list[Document]:
        """Get list of documents"""
        return self._documents

    @documents.setter
    def documents(self, docs: list[Document | tuple[str, str]] | None) -> None:
        """
        Validate and set documents value.
        We need to make sure if we have needed content and document_ids
        provided and the documents. All documents need to be instances
        of langchain's Document class.
        """
        proper_docs = []
        if docs:
            for idx, doc in enumerate(docs):
                if isinstance(doc, Document):
                    if not doc.page_content:
                        which_doc = doc.metadata.get("document_id") or idx
                        logger.warning("Empty document: %s", which_doc)
                    if not doc.metadata.get("document_id", None):
                        logger.warning("document_id not provided for document at index: %s", idx)

                    proper_docs.append(doc)

                else:
                    raise ValueError(f"Incorrect type of document provided at index: {idx}.")

        self._documents = proper_docs

    @property
    def optimization_metric(self) -> str:
        """Get optimization metrics used for the experiment."""
        return self._optimization_metric

    @optimization_metric.setter
    def optimization_metric(self, val: str) -> None:
        """Validate and set optimization metrics"""
        if val not in MetricType:
            raise RAGExperimentError(
                f"Provided optimization metric: '{val}' is not supported. "
                f"Available metrics: ['answer_correctness', 'faithfulness', 'context_correctness']."
            )

        self._optimization_metric = val

    @property
    def benchmark_data(self) -> BenchmarkData:
        """Get benchmark data."""
        return self._benchmark_data

    @benchmark_data.setter
    def benchmark_data(self, val: BenchmarkData) -> None:
        """Check and set benchmark data based on the executed scenario."""
        self._benchmark_data = val

    def run_pre_selection(
        self,
        foundation_models: list[FoundationModel],
        embedding_models: list[EmbeddingModel],
        n_records: int = 5,
        random_seed: int = 17,
    ) -> dict[str, list[EmbeddingModel | FoundationModel]]:
        """
        Run models pre-selection using ModelsPreSelector and sample
        of the data.

        Parameters
        ----------
        embedding_models : list[EmbeddingModel]
            Embedding models to be considered during pre-selection process.

        foundation_models : list[FoundationModel]
            Foundation models to be evaluated during pre-selection process.

        n_records : int, default=10
            Amount of records that should be used during models pre-selection.

        random_seed : int, default=17
            Random seed value used for sampling benchmark data records.

        Returns
        -------
        dict[str, list[FoundationModel | EmbeddingModel]]
            Best embedding models and foundation models found in pre-selection.
        """
        _log_start_mps = (
            f"Starting foundation models pre-selection with following foundation models: {[str(fm) for fm in foundation_models]} "
            f"and following embedding models: {[str(em) for em in embedding_models]}."
        )
        logger.info(_log_start_mps)
        self.event_handler.on_status_change(
            level=LogLevel.INFO,
            message=_log_start_mps,
            step=ExperimentStep.MODEL_SELECTION,
        )

        # pylint: disable=protected-access
        mps = ModelsPreSelector(
            ls_client=self.client,
            benchmark_data=self.benchmark_data.get_random_sample(n_records=n_records, random_seed=random_seed),
            documents=self.documents.copy(),
            foundation_models=foundation_models,
            embedding_models=embedding_models,
            experiment_monitor=self.experiment_monitor,
            metric=self.optimization_metric,
            predict_number_of_questions=len(self.benchmark_data),
        )
        mps.evaluate_patterns()

        selected_models = mps.select_models(n_em=self.n_mps_em, n_fm=self.n_mps_fm)

        logger.info(
            "Models pre-selection has been finished. Selected foundation models: %s and selected embedding models: %s.",
            [str(model) for model in selected_models["foundation_models"]],
            [str(model) for model in selected_models["embedding_models"]],
        )

        return selected_models

    def run_single_evaluation(self, rag_params: RAGParamsType, **kwargs: Any) -> float:
        """
        Selects proper single run evaluation function based on provided experiment params.

        Parameters
        ----------
        rag_params : RAGParamsType
            A dictionary containing rag parameters as keys and their values.

        Returns
        -------
        float
            A single evaluation score obtained by the executed rag pattern.
        """
        return self.run_single_evaluation_using_documents(rag_params, **kwargs)

    def run_single_evaluation_using_documents(self, rag_params: RAGParamsType, **kwargs: Any) -> float:
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

        distance_metric = EmbeddingModels.get_distance_metric(embedding_model.model_id)

        indexing_params = {
            "chunking": chunking_params,
            "embedding": {
                "model_id": embedding_model.model_id,
                "distance_metric": distance_metric,
            },
        }

        logger.info("Using indexing params: %s", indexing_params)

        context_template_text = foundation_model.context_template_text
        system_message_text = foundation_model.system_message_text
        user_message_text = foundation_model.user_message_text

        rag_params = {
            "retrieval": retrieval_params,
            "generation": {
                "model_id": foundation_model.model_id,
                "context_template_text": context_template_text,
                "user_message_text": user_message_text,
                "system_message_text": system_message_text,
            },
        }

        logger.info("Using retrieval and generation params: %s", rag_params)

        result_score = self.results.evaluation_explored_or_cached(
            indexing_params=indexing_params, rag_params=rag_params
        )
        if result_score is not None:
            return result_score

        pattern_name = self._create_pattern_name()
        self.experiment_monitor.on_pattern_start()
        logger.info("Using name '%s' for the currently evaluated pattern.", pattern_name)

        collection_name = self._create_collection_name(indexing_params=indexing_params)

        vector_store = get_vector_store(
            vs_type=self.vector_store_type,
            embedding_model=embedding_model,
            collection_name=collection_name,
        )

        if not self._collection_exists(collection_name=collection_name):
            chunking_method = chunking_params.get(AI4RAGParamNames.CHUNKING_METHOD)
            chunk_size = chunking_params.get(AI4RAGParamNames.CHUNK_SIZE)
            chunk_overlap = chunking_params.get(AI4RAGParamNames.CHUNK_OVERLAP)

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

            self.experiment_monitor.on_start_event_info()

            self.event_handler.on_status_change(
                level=LogLevel.INFO,
                message=f"Embedding chunks using the {embedding_model.model_id} model. Building index: {collection_name}.",
                step=ExperimentStep.EMBEDDING,
            )

            try:
                vector_store.add_documents(chunked_documents)
            except Exception as exc:
                raise IndexingError(exc, collection_name, embedding_model.model_id) from exc

            self.experiment_monitor.on_finish_event_info(
                event=EventsToReport.EMBEDDING, step=ExperimentStep.OPTIMIZATION, model_id=embedding_model.model_id
            )
        else:
            self.event_handler.on_status_change(
                level=LogLevel.INFO,
                message=f"Using index {collection_name}.",
                step=ExperimentStep.EMBEDDING,
            )

        retrieval_method = retrieval_params[AI4RAGParamNames.RETRIEVAL_METHOD]
        number_of_chunks = retrieval_params[AI4RAGParamNames.NUMBER_OF_CHUNKS]

        logger.info("Using retriever with parameters: %s", retrieval_params)

        retriever = Retriever(
            vector_store=vector_store,
            method=retrieval_method,
            number_of_chunks=number_of_chunks,
        )

        rag = LlamaStackRAG(
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

        self.experiment_monitor.on_start_event_info()
        inference_response = query_rag(
            rag=rag,
            questions=list(self.benchmark_data.questions),
        )
        self.experiment_monitor.on_finish_event_info(
            event=EventsToReport.RETRIEVAL_GENERATION,
            step=ExperimentStep.OPTIMIZATION,
            model_id=foundation_model.model_id,
            retrieved_chunks=number_of_chunks,
        )

        result_scores, evaluation_data = self._evaluate_response(
            inference_response=inference_response,
            pattern_name=pattern_name,
        )

        stop_time = time.time()
        execution_time = stop_time - start_time

        result_score = result_scores["scores"][self.optimization_metric]["mean"]

        logger.info("Calculated optimization score for '%s': %s", pattern_name, result_score)

        evaluation_result = EvaluationResult(
            pattern_name=pattern_name,
            collection=collection_name,
            indexing_params=indexing_params,
            rag_params=rag_params,
            scores=result_scores,
            execution_time=execution_time,
            final_score=result_score,
        )

        evaluation_results_json = self.results.create_evaluation_results_json(
            evaluation_data=evaluation_data, evaluation_result=evaluation_result
        )

        logger.info(
            "Evaluation scores: %s",
            {el.get("question_id"): el.get("scores") for el in evaluation_results_json if isinstance(el, dict)},
        )

        try:
            self._stream_finished_pattern(
                evaluation_result=evaluation_result,
                evaluation_results_json=evaluation_results_json,
            )
        except Exception as exc:
            raise AssetSaveError(exc) from exc

        self.experiment_monitor.on_pattern_finish(pattern_name)

        self.results.add_evaluation(
            evaluation_data=evaluation_data,
            evaluation_result=evaluation_result,
        )

        return result_score

    def search(self, **kwargs) -> Sequence[EvaluationResult]:
        """
        Prepare and execute experiment to find the best RAG parameters.

        Returns
        -------
        dict
            Dictionary with search results. The very same dictionary is saved in the AI4RAGExperiment
            instance in its self.search_output attribute.
        """

        logger.info("Starting RAG optimization process...")

        # pylint: disable=inconsistent-return-statements
        def objective_function(space: dict) -> float | None:
            """Function passed to the optimiser."""
            try:
                return self.run_single_evaluation(space, **kwargs)
            except AI4RAGError as err:
                msg = self.exceptions_handler.handle_exception(err)
                raise FailedIterationError(msg) from err

        # MPS - models pre-selection based on sample evaluation. Run if there are more than 3 foundation models
        foundation_models = list(self.search_space[AI4RAGParamNames.FOUNDATION_MODEL].values)
        embedding_models_parameter = self.search_space[AI4RAGParamNames.EMBEDDING_MODEL]
        embedding_models = list(embedding_models_parameter.values)

        if (
            embedding_models and len(embedding_models) > self.n_mps_em or len(foundation_models) > self.n_mps_fm
        ) and not kwargs.get("skip_mps", False):
            selected_models = self.run_pre_selection(foundation_models, embedding_models=embedding_models)
            self.search_space.search_space[AI4RAGParamNames.FOUNDATION_MODEL] = Parameter(
                name=AI4RAGParamNames.FOUNDATION_MODEL, param_type="C", values=selected_models["foundation_models"]
            )
            if not self.use_knowledge_bases:
                self.search_space.search_space[AI4RAGParamNames.EMBEDDING_MODEL] = Parameter(
                    name=AI4RAGParamNames.EMBEDDING_MODEL, param_type="C", values=selected_models["embedding_models"]
                )
            # To clear cached_property calculating possible combinations
            if hasattr(self.search_space, "combinations"):
                del self.search_space.combinations  # deleting so that it can be rebuilt with new modelsg

        optimiser_class: type[BaseOptimiser] = kwargs.get("optimiser", RandomOptimiser)

        # This line is introduced to make AI4RAGExperiment testing easier
        optimiser = optimiser_class(
            objective_function=objective_function,
            search_space=self.search_space,
            settings=self.optimiser_settings,
        )
        logger.debug(
            "Using optimiser: %s with optimiser settings: %s",
            optimiser_class.__name__,
            self.optimiser_settings.to_dict(),
        )

        try:
            _ = optimiser.search()
        except OptimisationError as err:
            final_error_msg = self.exceptions_handler.get_final_error_msg()
            raise RAGExperimentError(final_error_msg) from err

        self.search_output = self.results.get_best_evaluations(k=1)

        self.event_handler.on_status_change(
            level=LogLevel.INFO,
            message="Experiment optimization process finished.",
        )
        self.experiment_monitor.close()

        return self.search_output

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
        metrics = []
        for metric in self.metrics:
            scores = evaluation_result.scores["scores"][metric]
            single_metric = {
                "metric_name": metric,
                "mean": scores["mean"],
                "ci_low": scores["ci_low"],
                "ci_high": scores["ci_high"],
            }
            metrics.append(single_metric)

        retrieval_payload = {
            "method": evaluation_result.rag_params["retrieval"][AI4RAGParamNames.RETRIEVAL_METHOD],
            "number_of_chunks": evaluation_result.rag_params["retrieval"][AI4RAGParamNames.NUMBER_OF_CHUNKS],
        }

        if evaluation_result.rag_params["retrieval"][AI4RAGParamNames.WINDOW_SIZE]:
            retrieval_payload["window_size"] = evaluation_result.rag_params["retrieval"][AI4RAGParamNames.WINDOW_SIZE]

        vector_store_payload = {
            "datasource_type": self.vector_store_type,
            "collection_name": evaluation_result.collection,
        }

        indexing_payload = {
            "chunking": {
                "method": evaluation_result.indexing_params["chunking"][AI4RAGParamNames.CHUNKING_METHOD],
                "chunk_size": evaluation_result.indexing_params["chunking"][AI4RAGParamNames.CHUNK_SIZE],
                "chunk_overlap": evaluation_result.indexing_params["chunking"][AI4RAGParamNames.CHUNK_OVERLAP],
            },
            "embedding": evaluation_result.indexing_params.get("embedding"),
        }

        generation_payload = evaluation_result.rag_params.get("generation")

        payload = {
            "metrics": {"test_data": metrics},
            "context": {
                "auto_rag": {
                    "rag_pattern": {
                        "composition_steps": [
                            "model_selection",
                            "chunking",
                            "embeddings",
                            "retrieval",
                            "generation",
                        ],
                        "duration_seconds": int(evaluation_result.execution_time),
                        "name": evaluation_result.pattern_name,
                        "settings": {
                            "vector_store": vector_store_payload,
                            **indexing_payload,
                            **retrieval_payload,
                            "generation": generation_payload,
                        },
                    },
                    "iteration": len(self.results),
                    "max_combinations": self.search_space.max_combinations,
                }
            },
        }

        self.event_handler.on_pattern_creation(
            payload=payload,
            evaluation_results=evaluation_results_json,
            output_path=self.output_path,
            pattern_name=evaluation_result.pattern_name,
        )

    def _evaluate_response(
        self,
        inference_response: list[dict[str, Any]],
        pattern_name: str,
    ) -> tuple[dict[str, dict], list[EvaluationData]]:
        """
        Evaluate response from the model based on the chosen context,
        real questions/answers/ids from the benchmark_data.

        Parameters
        ----------
        inference_response : list[dict[str, Any]]
            List of model's responses containing question, answer and used
            reference documents for each record.

        pattern_name : str
            Name of the pattern for which evaluation is performed.

        Returns
        -------
        tuple[dict[str, dict], list[EvaluationData]]
            Data from evaluation that is of following structure:
            data = {
                "scores": {"answer_correctness": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}, ...},
                "question_scores": {
                    "answer_correctness": {"q_id_0": 0.5, "q_id_1": 0.8, ...},
                    "context_correctness": {"q_id_0": 0.5, "q_id_1": 0.8, ...},
                },
            }
        """

        logger.info(
            "Evaluating the RAG Pattern '%s' response using %s.", pattern_name, self.evaluator.__class__.__name__
        )
        self.event_handler.on_status_change(
            level=LogLevel.INFO,
            message=f"Evaluating the RAG Pattern '{pattern_name}' response using {self.evaluator.__class__.__name__}.",
            step="evaluation",
        )

        eval_data = build_evaluation_data(benchmark_data=self.benchmark_data, inference_response=inference_response)
        result = self.evaluator.evaluate_metrics(evaluation_data=eval_data, metrics=self.metrics)

        logger.info("Response evaluation results for '%s': %s.", pattern_name, result)
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

    def _create_collection_name(self, indexing_params: dict[str, Any]) -> str:
        """
        This method is utilised in the process of creating collection name
        for Any vector database.
        The general idea is to reuse the same collection if embedding and
        chunking parameters remain the same as in other trials, so we do not
        duplicate effort and consume more memory.

        Parameters
        ----------
        indexing_params : dict[str, Any]
            Dictionary containing keys and values that are compared with
            previously used ones, to establish if the newly created collection
            would be exactly the same.

        Returns
        -------
            Collection name that is new or one of the previously created
        """
        collection = self.results.collection_exists(indexing_params=indexing_params)
        if collection is not None:
            collection_name = collection
            logger.info("Reusing existing collection: '%s'", collection_name)
            return collection_name

        collection_id = datetime.now().strftime("%Y%m%d%H%M%S")

        ret = f"ai4rag_{self.job_id[:8]}_{collection_id}"
        logger.info("Creating new collection with name '%s'", ret)

        return ret

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
        return f"Pattern{len(self.results) + 1}"
