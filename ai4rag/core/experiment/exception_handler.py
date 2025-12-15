from collections import Counter

from ai4rag.utils.event_handler import BaseEventHandler, LogLevel
from ai4rag import logger


class AI4RAGError(Exception):
    """
    Base class for all external exceptions which can occur in AutoRAG.
    """

    def __init__(self, exception: Exception, message: str | None = None) -> None:
        self.exception = exception
        self.message = message

    def __repr__(self) -> str:
        if self.message:
            return f"AI4RAGError: {self.message} due to {repr(self.exception)}"
        return f"{self.__class__.__name__}: {repr(self.exception)})"

    def __str__(self) -> str:
        return repr(self)


class IndexingError(AI4RAGError):
    """Exception representing error during indexing chunks to vector store."""

    def __init__(self, exception, collection_name, embedding_model):
        super().__init__(exception)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}: Unable to embed and upload documents to vector store for collection name "
            f"'{self.collection_name}' and embedding model '{self.embedding_model}' due to: {repr(self.exception)}"
        )


class GenerationError(AI4RAGError):
    """Exception representing error during retrieval or inference."""

    def __init__(self, exception: Exception, inference_model: str, deployment_id: str | None = None):
        super().__init__(exception)
        self.inference_model = inference_model
        self.deployment_id = deployment_id
        self._message_core = (
            f"Unable to retrieve chunks and generate answers for deployment with ID: '{deployment_id}'"
            if deployment_id
            else f"Unable to retrieve chunks and generate answers for foundation model: '{inference_model}'"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {self._message_core} due to: {repr(self.exception)}"


class EvaluationError(AI4RAGError):
    """Exception representing error during evaluating generated pattern"""

    def __init__(self, exception):
        super().__init__(exception)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: Unable to evaluate generated pattern due to: {repr(self.exception)}"


class AssetSaveError(AI4RAGError):
    """Exception representing error during writing finished pattern to COS instance."""

    def __init__(self, exception):
        super().__init__(exception)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: Unable to save assets for evaluated pattern due to: {repr(self.exception)}"


class ExperimentExceptionsHandler:
    """
    Class responsible for handling exceptions raised during the experiment.
    Exceptions are stored in dict containing information about iteration and raised error.
    """

    def __init__(self, event_handler: BaseEventHandler | None = None) -> None:
        self.errors = []
        self.event_handler = event_handler

    def handle_exception(self, exception: AI4RAGError):
        """
        Handles single exception raised during the experiment.

        Parameters
        ----------
        exception : AI4RAGError
            Handled exception.

        """
        self.errors.append(exception)
        logger.warning(exception, exc_info=True)
        msg = repr(exception)
        if self.event_handler:
            self.event_handler.on_status_change(level=LogLevel.WARNING, message=msg)

        return msg

    def get_final_error_msg(self) -> str:
        """
        Counts exceptions raised during the experiment, generates and returns a summary message.
        The summary message is generated based on the most frequently occurring exception.

        Returns
        -------
        str
            Summary message for all aggregated errors,
        """
        logger.error("Several errors occurred during the experiment: %s", self.errors)

        errors_counter = Counter(error.__class__ for error in self.errors)
        most_common_error_type_name = errors_counter.most_common()[0][0].__name__

        error_content = next((er for er in self.errors if most_common_error_type_name in er.__class__.__name__))

        return f"{error_content}. " f"To find more details please see generated logs file."
