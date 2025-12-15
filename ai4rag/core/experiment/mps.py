# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from langchain_core.documents import Document

from ibm_watsonx_ai.foundation_models.extensions.rag.chunker import LangChainChunker
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.extensions.rag import Retriever
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import VectorStore
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.adapters.milvus_adapter import MilvusVectorStore

from ai4rag.core.experiment.exception_handler import (
    ExperimentExceptionsHandler,
    IndexingError,
    GenerationError,
    EvaluationError,
)
from ai4rag.evaluator import UnitxtEvaluator
from ai4rag.evaluator.base_evaluator import BaseEvaluator
from ai4rag.search_space.src.models import EmbeddingModels
from ai4rag.search_space.prepare.input_payload_types import AI4RAGModel
from ai4rag.core.ai_service.rag_service import RAGService
from ai4rag.core.experiment.utils import (
    query_inference_service,
    build_evaluation_data,
)
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.rag.vector_store.get_vector_store import get_vector_store
from ai4rag.rag.embedding import get_embeddings
from ai4rag.utils.constants import (
    DEFAULT_WORD_TO_TOKEN_RATIO,
    AI4RAGParamNames,
    ExperimentStep,
    EventsToReport,
    HybridRankerConstants,
)
from ai4rag.utils.knowledge_base_references import (
    get_vector_stores_from_knowledge_base_references,
    VectorStoreKnowledgeBaseReference,
    DatabaseKnowledgeBaseReference,
)
from ai4rag import logger


__all__ = ["PreSelectorError", "ModelsPreSelector"]


class PreSelectorError(Exception):
    """Exception to be raised when critical issue occurs in the MPS."""


# pylint: disable=too-many-instance-attributes
class ModelsPreSelector:
    """
    Class responsible for performing foundation models preselection.
    Using sample of benchmark_data and sample of grounding documents
    ModelsPreSelector is able to evaluate which top models
    should be selected as the best promising ones.

    ModelsPreSelector performs RAG service evaluation for each
    foundation model and embedding model pair with pre-configured settings
    using data sample. It provided best performing pairs of generation and
    embedding models, that are considered further in the experiment.

    If knowledge base references are provided, only foundation models are selected,
    since indexing phase is skipped.

    Parameters
    ----------
    embedding_models : list[str]
        Embedding models to models pre-selection.

    benchmark_data : BenchmarkData
        Sample of benchmark data used for the pre-selection.

    documents : list[Document]
        Grounding documents that will be sampled to perform pre-selection.

    agent : str
        Name of the agent to be used for models selection.

    kb_vector_store_references: list[VectorStoreKnowledgeBaseReference]
        List of vector stores to be used to select foundation models (scenario with knowledge base references).

    kb_database_references: list[DatabaseKnowledgeBaseReference]
        List of databases to be used to select foundation models (scenario with knowledge base references).

    foundation_models : list[AI4RAGModel]
        List of foundation models that should be considered in the selection.

    metric : str
        Metric used in ranking the models.

    api_client : APIClient | None, default=None
        Instance of APIClient to perform evaluation of RAG Patterns.
        It is optional so that we can provide known evaluations and use
        only models selection.


    Other Parameters
    ----------------
    evaluator : BaseEvaluator, default=UnitxtEvaluator()
        Instance used in the evaluation of RAG Patterns.

    generation_params : dict
        Mapping constraining generation parameters during inference.


    Attributes
    ----------
    retrieval_params : dict
        Retrieval parameters for all MPS evaluations.

    chunking_params : dict
        Chunking parameters for all MPS evaluations.

    evaluation_results : dict
        Dictionary holding results from evaluating each RAG Pattern.
        This may be overwritten by the user to avoid evaluation and
        pre-select models based on mean scores or RFR.

    mean_scores : dict[str, float]
        Mapping with predicted scores for each model.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        metric: str,
        foundation_models: list[AI4RAGModel],
        benchmark_data: BenchmarkData,
        embedding_models: list[str],
        documents: list[Document] | None = None,
        api_client: APIClient | None = None,
        **kwargs,
    ):
        self.benchmark_data = benchmark_data
        self.documents = documents
        self.foundation_models = foundation_models
        self.embedding_models = embedding_models
        self.metric = metric
        self.api_client = api_client

        self.evaluator: BaseEvaluator = kwargs.get(
            "evaluator", UnitxtEvaluator()
        )

        self.retrieval_params = {
            "number_of_chunks": kwargs.get(AI4RAGParamNames.NUMBER_OF_RETRIEVED_CHUNKS, 3),
            "method": kwargs.get(AI4RAGParamNames.RETRIEVAL_METHOD, "simple"),
            "window_size": kwargs.get(AI4RAGParamNames.RETRIEVAL_WINDOW_SIZE, 0),
        }

        self.chunking_params = {
            "chunk_size": kwargs.get(AI4RAGParamNames.CHUNK_SIZE, 512),
            "method": kwargs.get(AI4RAGParamNames.CHUNKING_METHOD, "recursive"),
            "chunk_overlap": kwargs.get(AI4RAGParamNames.CHUNK_OVERLAP, 128),
        }

        self.evaluation_results = {}
        self.mean_scores = {}

        self.exceptions_handler = ExperimentExceptionsHandler()
        self.experiment_monitor = kwargs.get("experiment_monitor", None)

    # pylint: disable=R0914
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
        for element in self.benchmark_data.documents_ids:
            document_ids.extend(element)

        documents = [document for document in self.documents if document.metadata["document_id"] in document_ids]
        chunked_documents = self._chunk_documents(documents)

        for i, embedding_model in enumerate(self.embedding_models):
            try:
                collection_name = f"MPS_collection_{i}"
                try:
                    vector_store = self._create_vector_store(
                        embedding_model, chunked_documents, collection_name=collection_name
                    )
                except Exception as exc:
                    raise IndexingError(exc, collection_name, embedding_model) from exc

                retriever = Retriever(vector_store, **self.retrieval_params)
                self._evaluate_foundation_models(retrievers=[retriever], embedding_model=embedding_model)

            except IndexingError as exc:
                self.exceptions_handler.handle_exception(exc)
                logger.warning("Pre-evaluation of '%s' has failed.", embedding_model)
                continue

        if not self.evaluation_results:
            msg = self.exceptions_handler.get_final_error_msg()
            raise PreSelectorError(
                f"Foundation models pre-selection has failed. "
                f"None of the given models has been successfully evaluated. {msg}"
            )

    def _evaluate_foundation_models(
        self,
        retrievers: list[Retriever] | None = None,
        embedding_model: str | None = None,
        kbr: bool = False,
    ):
        """
        Evaluates each embedding model with given retriever.

        Parameters
        ----------
        retrievers : BaseRetriever
            Retriever to be used in retrieval phase.

        embedding_model: str | None
            Name of the embedding model used by the retriever,
            is set to None if knowledge base references were not provided.

        kbr : bool, default=False
            This flag is set to True when knowledge base references have been used.
        """
        for foundation_model in self.foundation_models:
            try:
                embedding_model_log = f"and embedding model: {embedding_model}" if embedding_model else ""
                logger.info(
                    "Starting pre-evaluation of foundation model: %s %s.", foundation_model, embedding_model_log
                )

                result_scores = self._evaluate_single_pattern(
                    foundation_model=foundation_model, retrievers=retrievers, databases=databases, kbr=kbr
                )

                self.evaluation_results[(embedding_model, foundation_model)] = result_scores

                logger.debug(
                    "Finished pre-evaluation of foundation model: %s%s",
                    foundation_model,
                    embedding_model_log,
                )
            except (GenerationError, EvaluationError) as exc:
                self.exceptions_handler.handle_exception(exc)
                logger.warning("Pre-evaluation of '%s' has failed.", foundation_model)
                continue

    def _create_vector_store(
        self, embedding_model: str, chunked_documents: list[Document], collection_name: str = "MPS_collection"
    ) -> VectorStore:
        """
        Create instance of vector store with given chunked documents and embedding model.

        Parameters
        ----------
        embedding_model : str
            ID of the watsonx embedding model used to create instance
            of Embeddings and VectorStore.

        chunked_documents : list[Document]
            Chunked documents fot the embedding process.

        collection_name : str, default="MPS_collection"
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
        logger.info("Building index for pre-evaluation using embedding model: '%s'.", embedding_model)
        embeddings = get_embeddings(model_name=embedding_model, api_client=self.api_client)

        distance_metric = EmbeddingModels.get_distance_metric(embedding_model)
        vector_store = get_vector_store(
            vs_type="chroma",
            embeddings=embeddings,
            distance_metric=distance_metric,
            index_name=collection_name,
        )

        logger.debug("MPS: Embedding documents ...")
        if self.experiment_monitor:
            self.experiment_monitor.on_start_event_info()
        try:
            vector_store.add_documents(chunked_documents)
        except WMLClientError as err:
            logger.warning("Failed to create in-memory vector index due to: %s.", repr(err), exc_info=True)
            try:
                vector_store.add_documents(chunked_documents)
            except WMLClientError as exc:
                raise PreSelectorError(f"Failed to create in-memory vector index due to: {repr(exc)}.") from exc
        if self.experiment_monitor:
            self.experiment_monitor.on_finish_event_info(
                event=EventsToReport.EMBEDDING, step=ExperimentStep.MODEL_SELECTION, model_id=embedding_model
            )
        logger.debug("MPS: Embedding documents finished!")

        return vector_store

    def _evaluate_single_pattern(
        self,
        foundation_model: AI4RAGModel,
        databases: list[DatabaseKnowledgeBaseReference] | None,
        retrievers: list[Retriever] | None,
        kbr: bool = False,
    ) -> dict[str, dict]:
        """
        Perform retrieval-augmented generation and evaluate generated response.

        Parameters
        ----------
        foundation_model : str
            ID of the foundation model to be used RAG.

        databases : list[DatabaseKnowledgeBaseReference]
            List of database references to use for evaluation.

        retrievers : Retriever
            Instances for retrieving documents from vector database.

        Returns
        -------
        dict[str, dict]
            Evaluation scores per model.
        """
        generation_params = foundation_model.parameters.to_dict()
        default_max_sequence_length = (
            generation_params.pop("max_sequence_length")
            if generation_params.get("max_sequence_length")
            else foundation_model.max_sequence_length
        )
        model_inference = foundation_model.model_inference(api_client=self.api_client, params=generation_params)

        rag_service = RAGService(
            agent=self.agent,
            api_client=self.api_client,
            model=model_inference,
            context_template_text=foundation_model.context_template_text,
            system_message_text=getattr(foundation_model.chat_template_messages, "system_message_text", None),
            user_message_text=getattr(foundation_model.chat_template_messages, "user_message_text", None),
            retrievers=retrievers,
            databases=databases,
            default_max_sequence_length=default_max_sequence_length,
            word_to_token_ratio=DEFAULT_WORD_TO_TOKEN_RATIO,
            ranker_config=self.ranker_config,
            multiindex_enabled=kbr,
        )

        # pylint: disable=duplicate-code
        if self.experiment_monitor:
            self.experiment_monitor.on_start_event_info()

        inference_response = query_inference_service(
            api_client=self.api_client,
            rag_service=rag_service,
            questions=list(self.benchmark_data.questions),
        )

        if self.experiment_monitor:
            self.experiment_monitor.on_finish_event_info(
                event=EventsToReport.RETRIEVAL_GENERATION,
                step=ExperimentStep.MODEL_SELECTION,
                model_id=str(foundation_model),
                retrieved_chunks=self.retrieval_params["number_of_chunks"],
            )
        # pylint: enable=duplicate-code

        result_scores = self._evaluate_response(inference_response=inference_response)

        return result_scores

    def select_models(self, n_em: int = 2, n_fm: int = 3) -> dict[str, list[str]]:
        """
        Select n models pairs based on evaluation scores.

        Parameters
        ----------
        n_em : int, default=2
            Amount of embedding models to be returned.

        n_fm : int, default=3
            Amount of foundation models to be returned.

        Returns
        -------
        dict[str, list[str]]
            Pre-selected embedding and foundation models.
        """

        logger.info("Selecting the best %s embedding models and %s foundation models.", n_em, n_fm)
        best_models_pairs = self._mean_based_scoring()

        embedding_models = []
        foundation_models = []
        seen = set()
        for em, fm in best_models_pairs:
            if em is not None and em not in seen:
                seen.add(em)
                embedding_models.append(em)
            if fm not in seen:
                seen.add(fm)
                foundation_models.append(fm)

        ret = {"foundation_models": foundation_models[:n_fm]}
        if embedding_models:
            ret["embedding_models"] = embedding_models[:n_em]

        return ret

    def _mean_based_scoring(self) -> list[tuple[str, str]]:
        """
        Scoring models based on mean metric value for all records used
        in the evaluation.

        Returns
        -------
        list[str]
            Models pairs with the best scores.
        """
        logger.debug("MPS: Starting mean-based scoring...")

        for models, results in self.evaluation_results.items():
            mean_score = results.get("scores", {}).get(self.metric, {}).get("mean", {})
            self.mean_scores[models] = mean_score

        logger.debug("MPS: Finished mean-based scoring!")

        models_with_scores = sorted(
            [{"models_names": models_names, "val": val} for models_names, val in self.mean_scores.items()],
            key=lambda x: x.get("val"),
            reverse=True,
        )
        top_models_pairs = [x["models_names"] for x in models_with_scores]

        return top_models_pairs

    def _evaluate_response(self, inference_response: list[dict[str, Any]]) -> dict[str, dict]:
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
        dict[str, dict]
            Data from evaluation that is of following structure:
            data = {
                "scores": {"answer_correctness": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}, ...},
                "question_scores": {
                    "answer_correctness": {"q_id_0": 0.5, "q_id_1": 0.8, ...},
                    "context_correctness": {"q_id_0": 0.5, "q_id_1": 0.8, ...},
                },
            }
        """
        logger.debug("MPS: Evaluating responses...")

        eval_data = build_evaluation_data(benchmark_data=self.benchmark_data, inference_response=inference_response)
        evaluation_result = self.evaluator.evaluate_metrics(evaluation_data=eval_data, metrics=[self.metric])

        logger.debug("MPS: Responses evaluation finished!")

        return evaluation_result

    def _chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk provided documents.

        Parameters
        ----------
        documents : list[Document]
            List of LangChain.Document instance that will be chunked.

        Returns
        -------
        list[Document]
            Chunked documents.
        """
        logger.debug("MPS: Chunking documents...")
        chunker = LangChainChunker(**self.chunking_params)

        chunked_documents = chunker.split_documents(documents)

        logger.debug("MPS: Chunking documents finished!")

        return chunked_documents
