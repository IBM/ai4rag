# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, TypedDict

from docling_core.types.doc import DoclingDocument

from ai4rag import logger
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.exception_handler import (
    EvaluationError,
    ExperimentExceptionHandler,
    GenerationError,
    IndexingError,
)
from ai4rag.core.experiment.utils import build_evaluation_data, query_rag
from ai4rag.evaluator import UnitxtEvaluator
from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationMetricsResult
from ai4rag.evaluator.custom_metrics import apply_custom_metrics
from ai4rag.evaluator.metric import Metrics, RAGMetric
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.template.simple_rag_template import SimpleRAG
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.chroma import ChromaVectorStore
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = ["PreSelectorError", "ModelsPreSelector"]


class PreSelectorError(Exception):
    """Exception to be raised when critical issue occurs in the MPS."""


class MPSEvaluationResultsTyped(TypedDict):
    """Typing helper for evaluation results coming from ModelsPreSelector."""

    foundation_model: BaseFoundationModel
    embedding_model: BaseEmbeddingModel
    evaluation: EvaluationMetricsResult


# pylint: disable=too-many-instance-attributes
class ModelsPreSelector:
    """
    Class responsible for performing foundation and embeddings models preselection.
    Using sample of benchmark_data and sample of grounding documents
    ModelsPreSelector is able to evaluate which top models
    should be selected as the best promising ones for the further experiment steps.

    ModelsPreSelector performs RAG pattern evaluation for each
    foundation model and embedding model pair with pre-configured settings
    using data sample. It provides the best performing pairs of generation and
    embedding models, that are considered further during HPO.

    Parameters
    ----------
    foundation_models : list[BaseFoundationModel]
        List of foundation models that should be considered in the selection.

    embedding_models : list[BaseEmbeddingModel]
        Embedding models to models pre-selection.

    documents : list[DoclingDocument]
        Grounding documents that will be sampled to perform pre-selection.

    benchmark_data : BenchmarkData
        Sample of benchmark data used for the pre-selection.

    metric : RAGMetric, default=Metrics.OVERALL_SCORE
        Metric used for ranking models during pre-selection.

    Attributes
    ----------
    evaluator : BaseEvaluator
        Instance responsible for RAG pattern's response evaluation.

    retrieval_params : dict
        Retrieval parameters for all MPS evaluations.

    chunking_params : dict
        Chunking parameters for all MPS evaluations.

    evaluation_results : list[MPSEvaluationResultsTyped]
        Dictionary holding results from evaluating each RAG Pattern.
        This may be overwritten by the user to avoid evaluation and
        pre-select models based on mean scores.

    DEFAULT_N_FOUNDATION_MODELS : int
        Number of foundation models to select in the process of MPS.

    DEFAULT_N_EMBEDDING_MODELS : int
        Number of embedding models to select in the process of MPS.
    """

    DEFAULT_N_FOUNDATION_MODELS = 3
    DEFAULT_N_EMBEDDING_MODELS = 2

    def __init__(
        self,
        foundation_models: list[BaseFoundationModel],
        embedding_models: list[BaseEmbeddingModel],
        documents: list[DoclingDocument],
        benchmark_data: BenchmarkData,
        metric: RAGMetric = Metrics.OVERALL_SCORE,
        **kwargs,
    ):
        self.benchmark_data = benchmark_data
        self.documents = documents
        self.foundation_models = foundation_models
        self.embedding_models = embedding_models
        self.metric = metric

        self.evaluator: BaseEvaluator = kwargs.pop("evaluator", UnitxtEvaluator())
        self.retrieval_params = {
            "number_of_chunks": kwargs.get(AI4RAGParamNames.NUMBER_OF_CHUNKS, 3),
            "method": kwargs.get(AI4RAGParamNames.RETRIEVAL_METHOD, "simple"),
            "search_mode": kwargs.get(AI4RAGParamNames.SEARCH_MODE, "vector"),
        }
        self.chunking_params = {
            "chunk_size": kwargs.get(AI4RAGParamNames.CHUNK_SIZE, 512),
            "method": kwargs.get(AI4RAGParamNames.CHUNKING_METHOD, "recursive"),
            "chunk_overlap": kwargs.get(AI4RAGParamNames.CHUNK_OVERLAP, 128),
        }
        self.evaluation_results: list[MPSEvaluationResultsTyped] = []
        self._exception_handler = ExperimentExceptionHandler()
        self.max_threads = kwargs.pop("max_threads", 10)
        self._unitxt_metrics = tuple(m for m in Metrics if m.evaluator == "unitxt")
        self.metrics = (*self._unitxt_metrics, Metrics.OVERALL_SCORE)

    def evaluate_patterns(self):
        """
        Evaluate RAG pattern per each foundation model provided. All settings
        of the patterns are configured and remain the same for each evaluation.

        If knowledge base references were provided, the retriever is created once and reused.
        Otherwise, a separate vector store is built for each embedding model, and the best-performing ones are selected.

        For evaluation only sample of the documents is used, embedded and added
        to chroma vector store.

        This method does not return anything, but in the end changes attributes
        of the instance: self.evaluation_results is a mapping holding results
        for each pattern.
        """
        logger.debug("MPS: Sampling documents")
        document_ids = []
        for element in self.benchmark_data.document_ids:
            document_ids.extend(element)

        documents = [document for document in self.documents if document.name in document_ids]
        chunked_documents = self._chunk_documents(documents)

        for i, embedding_model in enumerate(self.embedding_models):
            try:
                collection_name = f"mps_collection_{i}"
                try:
                    vector_store = self._create_vector_store(
                        embedding_model, chunked_documents, collection_name=collection_name
                    )
                except Exception as exc:
                    raise IndexingError(exc, collection_name, embedding_model.model_id) from exc

                retriever = Retriever(vector_store, **self.retrieval_params)
                self._evaluate_foundation_models(retriever=retriever, embedding_model=embedding_model)

            except IndexingError as exc:
                self._exception_handler.handle_exception(exc)
                logger.warning("Pre-evaluation of '%s' has failed.", embedding_model.model_id)
                continue

        if not self.evaluation_results:
            msg = self._exception_handler.get_final_error_msg()
            raise PreSelectorError(
                f"Foundation models pre-selection has failed. "
                f"None of the given models has been successfully evaluated. {msg}"
            )

    def _evaluate_foundation_models(self, retriever: Retriever, embedding_model: BaseEmbeddingModel):
        """
        Evaluates each embedding model with given retriever.

        Parameters
        ----------
        retriever : Retriever
            Instance to be used in retrieval phase.

        embedding_model : BaseEmbeddingModel
            Embedding model used for collection creation.
        """
        for foundation_model in self.foundation_models:
            try:
                logger.info(
                    "Starting pre-evaluation of foundation model: %s and embedding model: %s.",
                    foundation_model.model_id,
                    embedding_model.model_id,
                )

                result_scores = self._evaluate_single_pattern(foundation_model=foundation_model, retriever=retriever)

                self.evaluation_results.append(
                    {
                        "embedding_model": embedding_model,
                        "foundation_model": foundation_model,
                        "evaluation": result_scores,
                    }
                )

                logger.debug(
                    "Finished pre-evaluation of foundation model: %s and embedding model: %s",
                    foundation_model.model_id,
                    embedding_model.model_id,
                )
            except (GenerationError, EvaluationError) as exc:
                self._exception_handler.handle_exception(exc)
                logger.warning("Pre-evaluation of '%s' has failed.", foundation_model.model_id)
                continue

    @staticmethod
    def _create_vector_store(
        embedding_model: BaseEmbeddingModel,
        chunked_documents: list[AI4RAGChunk],
        collection_name: str,
    ) -> BaseVectorStore:
        """
        Create instance of vector store with given chunked documents and embedding model.

        Parameters
        ----------
        embedding_model : BaseEmbeddingModel
            Embedding model used for collection creation.

        chunked_documents : list[AI4RAGChunk]
            Chunked documents for the embedding process.

        collection_name : str
            Name of the collection in the chroma vector database.

        Returns
        -------
        VectorStore
            Instance for communication with properly created index in the
            vector database.

        Raises
        ------
        PreSelectorError
            When 2 attempts of embedding documents are failing
        """
        logger.info("Building index for pre-evaluation using embedding model: '%s'.", embedding_model.model_id)

        vector_store = ChromaVectorStore(
            embedding_model=embedding_model,
            collection_name=collection_name,
        )

        logger.debug("MPS: Embedding documents ...")
        try:
            vector_store.add_documents(chunked_documents)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to create in-memory vector index due to: %s.", repr(err), exc_info=True)
            try:
                vector_store.add_documents(chunked_documents)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                raise PreSelectorError(f"Failed to create in-memory vector index due to: {repr(exc)}.") from exc
        logger.debug("MPS: Embedding documents finished!")

        return vector_store

    def _evaluate_single_pattern(
        self, foundation_model: BaseFoundationModel, retriever: Retriever
    ) -> EvaluationMetricsResult:
        """
        Perform retrieval-augmented generation and evaluate generated response.

        Parameters
        ----------
        foundation_model : BaseFoundationModel
            Model to be used for RAG.

        retriever : Retriever
            Instance for retrieving documents from vector database.

        Returns
        -------
        EvaluationMetricsResult
            Aggregate metrics with confidence intervals and per-question scores.
        """

        rag = SimpleRAG(foundation_model=foundation_model, retriever=retriever)

        inference_response = query_rag(
            rag=rag, questions=list(self.benchmark_data.questions), max_threads=self.max_threads
        )

        result = self._evaluate_response(inference_response=inference_response)

        apply_custom_metrics(scores=result, metrics=self.metrics)

        return result

    def select_models(
        self,
        n_embedding_models: int = DEFAULT_N_EMBEDDING_MODELS,
        n_foundation_models: int = DEFAULT_N_FOUNDATION_MODELS,
    ) -> dict[str, list[BaseEmbeddingModel | BaseFoundationModel]]:
        """
        Select n models pairs based on evaluation scores.

        Parameters
        ----------
        n_embedding_models : int, default=2
            Amount of embedding models to be returned.

        n_foundation_models : int, default=3
            Amount of foundation models to be returned.

        Returns
        -------
        dict[str, list[str]]
            Pre-selected embedding and foundation models.
        """

        logger.info(
            "Search space contains %s foundation model(s) and %s embedding model(s). Starting models pre-selection...",
            len(self.foundation_models),
            len(self.embedding_models),
        )
        top_models_with_scores = self._mean_based_scoring()

        embedding_models = []
        foundation_models = []
        seen = set()
        for element in top_models_with_scores:
            fm = element.get("foundation_model")
            em = element.get("embedding_model")
            if em is not None and em not in seen:
                seen.add(em)
                embedding_models.append(em)
            if fm not in seen:
                seen.add(fm)
                foundation_models.append(fm)

        ret = {
            "foundation_models": foundation_models[:n_foundation_models],
            "embedding_models": embedding_models[:n_embedding_models],
        }

        logger.info(
            "Selected the best %s embedding model(s) and %s foundation model(s).",
            len(ret["embedding_models"]),
            len(ret["foundation_models"]),
        )

        return ret

    def _mean_based_scoring(self) -> list[dict]:
        """
        Scoring models based on mean metric value for all records used
        in the evaluation.

        Returns
        -------
        list[dict]
            Models and their corresponding mean scores in descending order.
        """
        logger.debug("MPS: Starting mean-based scoring...")

        _mean_scoring_results = []

        for result in self.evaluation_results:
            evaluation = result.get("evaluation", {})
            mean_score = next(
                (m["scores"]["mean"] for m in evaluation.get("metrics", []) if m["name"] == self.metric.name),
                None,
            )
            _mean_scoring_results.append(
                {
                    "embedding_model": result.get("embedding_model"),
                    "foundation_model": result.get("foundation_model"),
                    "score": mean_score,
                }
            )

        logger.debug("MPS: Finished mean-based scoring!")

        models_with_scores = sorted(
            _mean_scoring_results,
            key=lambda x: x["score"] if x["score"] is not None else float("-inf"),
            reverse=True,
        )

        return models_with_scores

    def _evaluate_response(self, inference_response: list[dict[str, Any]]) -> EvaluationMetricsResult:
        """
        Evaluate response from the model based on the chosen context,
        real questions/answers/ids from the benchmark_data.

        Parameters
        ----------
        inference_response : list[dict[str, Any]]
            List of model's responses containing question, answer and used
            reference documents for each record.

        Returns
        -------
        EvaluationMetricsResult
            Aggregate metrics with confidence intervals and per-question scores.
        """
        logger.debug("MPS: Evaluating responses...")

        eval_data = build_evaluation_data(benchmark_data=self.benchmark_data, inference_response=inference_response)
        evaluation_result = self.evaluator.evaluate_metrics(evaluation_data=eval_data, metrics=self._unitxt_metrics)

        logger.debug("MPS: Responses evaluation finished!")

        return evaluation_result

    def _chunk_documents(self, documents: list[DoclingDocument]) -> list[AI4RAGChunk]:
        """
        Chunk provided documents.

        Parameters
        ----------
        documents : list[DoclingDocument]
            Docling documents to chunk.

        Returns
        -------
        list[AI4RAGChunk]
            Chunked documents.
        """
        logger.debug("MPS: Chunking documents...")
        chunker = LangChainChunker(**self.chunking_params)

        chunked_documents = chunker.split_documents(documents)

        logger.debug("MPS: Chunking documents finished!")

        return chunked_documents
