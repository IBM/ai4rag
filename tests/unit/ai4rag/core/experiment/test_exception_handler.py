# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag.core.experiment.exception_handler import (
    AI4RAGError,
    AssetSaveError,
    EvaluationError,
    ExperimentExceptionHandler,
    GenerationError,
    IndexingError,
)
from ai4rag.utils.event_handler import LogLevel


class TestAI4RAGError:
    """Test suite for AI4RAGError base exception class."""

    def test_ai4rag_error_creation_with_exception(self):
        """Test creating AI4RAGError with exception."""
        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception)

        assert error.exception == base_exception
        assert error.message is None

    def test_ai4rag_error_creation_with_message(self):
        """Test creating AI4RAGError with exception and message."""
        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception, message="Custom message")

        assert error.exception == base_exception
        assert error.message == "Custom message"

    def test_ai4rag_error_repr_without_message(self):
        """Test __repr__ when message is None."""
        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception)

        repr_str = repr(error)
        assert "AI4RAGError" in repr_str
        assert "ValueError" in repr_str

    def test_ai4rag_error_repr_with_message(self):
        """Test __repr__ when message is provided."""
        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception, message="Custom message")

        repr_str = repr(error)
        assert "AI4RAGError" in repr_str
        assert "Custom message" in repr_str
        assert "ValueError" in repr_str

    def test_ai4rag_error_str(self):
        """Test __str__ returns same as __repr__."""
        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception, message="Custom message")

        assert str(error) == repr(error)


class TestIndexingError:
    """Test suite for IndexingError exception class."""

    def test_indexing_error_creation(self):
        """Test creating IndexingError."""
        base_exception = ConnectionError("Connection failed")
        error = IndexingError(base_exception, "collection_test", "model-123")

        assert error.exception == base_exception
        assert error.collection_name == "collection_test"
        assert error.embedding_model_id == "model-123"

    def test_indexing_error_repr(self):
        """Test IndexingError __repr__."""
        base_exception = ConnectionError("Connection failed")
        error = IndexingError(base_exception, "collection_test", "model-123")

        repr_str = repr(error)
        assert "IndexingError" in repr_str
        assert "collection_test" in repr_str
        assert "model-123" in repr_str
        assert "ConnectionError" in repr_str

    def test_indexing_error_is_ai4rag_error(self):
        """Test that IndexingError inherits from AI4RAGError."""
        assert issubclass(IndexingError, AI4RAGError)


class TestGenerationError:
    """Test suite for GenerationError exception class."""

    def test_generation_error_creation(self):
        """Test creating GenerationError."""
        base_exception = TimeoutError("Timeout")
        error = GenerationError(base_exception, "llama-model-id")

        assert error.exception == base_exception
        assert error.model_id == "llama-model-id"

    def test_generation_error_repr(self):
        """Test GenerationError __repr__."""
        base_exception = TimeoutError("Timeout")
        error = GenerationError(base_exception, "llama-model-id")

        repr_str = repr(error)
        assert "GenerationError" in repr_str
        assert "llama-model-id" in repr_str
        assert "TimeoutError" in repr_str
        assert "Unable to retrieve chunks and generate answers" in repr_str

    def test_generation_error_is_ai4rag_error(self):
        """Test that GenerationError inherits from AI4RAGError."""
        assert issubclass(GenerationError, AI4RAGError)


class TestEvaluationError:
    """Test suite for EvaluationError exception class."""

    def test_evaluation_error_creation(self):
        """Test creating EvaluationError."""
        base_exception = ValueError("Evaluation failed")
        error = EvaluationError(base_exception)

        assert error.exception == base_exception

    def test_evaluation_error_repr(self):
        """Test EvaluationError __repr__."""
        base_exception = ValueError("Evaluation failed")
        error = EvaluationError(base_exception)

        repr_str = repr(error)
        assert "EvaluationError" in repr_str
        assert "ValueError" in repr_str
        assert "Unable to evaluate generated pattern" in repr_str

    def test_evaluation_error_is_ai4rag_error(self):
        """Test that EvaluationError inherits from AI4RAGError."""
        assert issubclass(EvaluationError, AI4RAGError)


class TestAssetSaveError:
    """Test suite for AssetSaveError exception class."""

    def test_asset_save_error_creation(self):
        """Test creating AssetSaveError."""
        base_exception = IOError("Save failed")
        error = AssetSaveError(base_exception)

        assert error.exception == base_exception

    def test_asset_save_error_repr(self):
        """Test AssetSaveError __repr__."""
        base_exception = IOError("Save failed")
        error = AssetSaveError(base_exception)

        repr_str = repr(error)
        assert "AssetSaveError" in repr_str
        assert "OSError" in repr_str or "IOError" in repr_str  # IOError is OSError in Python 3
        assert "Unable to save assets" in repr_str

    def test_asset_save_error_is_ai4rag_error(self):
        """Test that AssetSaveError inherits from AI4RAGError."""
        assert issubclass(AssetSaveError, AI4RAGError)


class TestExperimentExceptionsHandlerInitialization:
    """Test suite for ExperimentExceptionsHandler initialization."""

    def test_init_without_event_handler(self):
        """Test initialization without event handler."""
        handler = ExperimentExceptionHandler()

        assert handler.errors == []
        assert handler.event_handler is None

    def test_init_with_event_handler(self, mocker):
        """Test initialization with event handler."""
        mock_event_handler = mocker.MagicMock()
        handler = ExperimentExceptionHandler(event_handler=mock_event_handler)

        assert handler.errors == []
        assert handler.event_handler == mock_event_handler


class TestExperimentExceptionsHandlerHandleException:
    """Test suite for ExperimentExceptionsHandler.handle_exception method."""

    def test_handle_exception_adds_to_errors_list(self, mocker):
        """Test that handle_exception adds exception to errors list."""
        mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception, message="Custom message")

        handler.handle_exception(error)

        assert len(handler.errors) == 1
        assert handler.errors[0] == error

    def test_handle_exception_logs_warning(self, mocker):
        """Test that handle_exception logs warning."""
        mock_logger = mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception, message="Custom message")

        handler.handle_exception(error)

        mock_logger.warning.assert_called_once()

    def test_handle_exception_returns_error_message(self, mocker):
        """Test that handle_exception returns error representation."""
        mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        base_exception = ValueError("Test error")
        error = IndexingError(base_exception, "collection_1", "model_1")

        result = handler.handle_exception(error)

        assert isinstance(result, str)
        assert "IndexingError" in result

    def test_handle_exception_calls_event_handler(self, mocker):
        """Test that handle_exception calls event_handler if provided."""
        mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        mock_event_handler = mocker.MagicMock()
        handler = ExperimentExceptionHandler(event_handler=mock_event_handler)

        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception)

        handler.handle_exception(error)

        mock_event_handler.on_status_change.assert_called_once()
        call_args = mock_event_handler.on_status_change.call_args
        assert call_args.kwargs["level"] == LogLevel.WARNING

    def test_handle_exception_does_not_call_event_handler_if_none(self, mocker):
        """Test that handle_exception doesn't call event_handler when None."""
        mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler(event_handler=None)

        base_exception = ValueError("Test error")
        error = AI4RAGError(base_exception)

        # Should not raise error
        handler.handle_exception(error)

        assert len(handler.errors) == 1

    def test_handle_exception_multiple_exceptions(self, mocker):
        """Test handling multiple exceptions."""
        mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        errors = [
            IndexingError(ValueError("Error 1"), "col1", "model1"),
            GenerationError(RuntimeError("Error 2"), "model2"),
            EvaluationError(TypeError("Error 3")),
        ]

        for error in errors:
            handler.handle_exception(error)

        assert len(handler.errors) == 3
        assert handler.errors == errors


class TestExperimentExceptionsHandlerGetFinalErrorMsg:
    """Test suite for ExperimentExceptionsHandler.get_final_error_msg method."""

    def test_get_final_error_msg_single_error(self, mocker):
        """Test get_final_error_msg with single error."""
        mock_logger = mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        error = IndexingError(ValueError("Test error"), "collection_1", "model_1")
        handler.handle_exception(error)

        # Reset logger mock after handle_exception
        mock_logger.reset_mock()

        msg = handler.get_final_error_msg()

        assert isinstance(msg, str)
        assert "IndexingError" in msg
        assert "please see generated logs file" in msg.lower()
        mock_logger.error.assert_called_once()

    def test_get_final_error_msg_multiple_errors_same_type(self, mocker):
        """Test get_final_error_msg with multiple errors of same type."""
        mock_logger = mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        errors = [
            IndexingError(ValueError("Error 1"), "col1", "model1"),
            IndexingError(ValueError("Error 2"), "col2", "model2"),
            IndexingError(ValueError("Error 3"), "col3", "model3"),
        ]

        for error in errors:
            handler.handle_exception(error)

        mock_logger.reset_mock()

        msg = handler.get_final_error_msg()

        assert "IndexingError" in msg
        mock_logger.error.assert_called_once()

    def test_get_final_error_msg_multiple_error_types(self, mocker):
        """Test get_final_error_msg with multiple error types returns most common."""
        mock_logger = mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        errors = [
            IndexingError(ValueError("Error 1"), "col1", "model1"),
            IndexingError(ValueError("Error 2"), "col2", "model2"),
            GenerationError(RuntimeError("Error 3"), "model3"),
        ]

        for error in errors:
            handler.handle_exception(error)

        mock_logger.reset_mock()

        msg = handler.get_final_error_msg()

        # Most common error type is IndexingError (2 occurrences)
        assert "IndexingError" in msg
        mock_logger.error.assert_called_once()

    def test_get_final_error_msg_logs_all_errors(self, mocker):
        """Test that get_final_error_msg logs all errors."""
        mock_logger = mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        handler = ExperimentExceptionHandler()

        errors = [
            IndexingError(ValueError("Error 1"), "col1", "model1"),
            GenerationError(RuntimeError("Error 2"), "model2"),
        ]

        for error in errors:
            handler.handle_exception(error)

        mock_logger.reset_mock()

        handler.get_final_error_msg()

        # Should log all errors
        assert mock_logger.error.called
        log_args = mock_logger.error.call_args[0]
        assert "Several errors occurred" in log_args[0]


class TestExperimentExceptionsHandlerIntegration:
    """Integration tests for ExperimentExceptionsHandler."""

    def test_full_workflow_with_event_handler(self, mocker):
        """Test complete workflow with event handler."""
        mock_logger = mocker.patch("ai4rag.core.experiment.exception_handler.logger")
        mock_event_handler = mocker.MagicMock()

        handler = ExperimentExceptionHandler(event_handler=mock_event_handler)

        # Handle multiple errors
        errors = [
            IndexingError(ValueError("Indexing failed"), "col1", "model1"),
            GenerationError(RuntimeError("Generation failed"), "model2"),
            IndexingError(ValueError("Another indexing error"), "col2", "model3"),
        ]

        for error in errors:
            handler.handle_exception(error)

        # Verify all errors were stored
        assert len(handler.errors) == 3

        # Verify event handler was called for each error
        assert mock_event_handler.on_status_change.call_count == 3

        # Get final error message
        mock_logger.reset_mock()
        msg = handler.get_final_error_msg()

        # Most common error is IndexingError
        assert "IndexingError" in msg
        mock_logger.error.assert_called_once()

    def test_full_workflow_without_event_handler(self, mocker):
        """Test complete workflow without event handler."""
        mock_logger = mocker.patch("ai4rag.core.experiment.exception_handler.logger")

        handler = ExperimentExceptionHandler(event_handler=None)

        # Handle errors
        errors = [
            GenerationError(RuntimeError("Error 1"), "model1"),
            GenerationError(RuntimeError("Error 2"), "model2"),
        ]

        for error in errors:
            handler.handle_exception(error)

        assert len(handler.errors) == 2

        mock_logger.reset_mock()
        msg = handler.get_final_error_msg()

        assert "GenerationError" in msg
