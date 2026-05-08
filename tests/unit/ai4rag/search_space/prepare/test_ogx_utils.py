# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock, Mock

import pytest

from ai4rag.rag.embedding.ogx import OGXEmbeddingModel, OGXEmbeddingParams
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.search_space.prepare.ogx_utils import (
    SearchSpaceValueError,
    _get_default_ogx_models,
    _validate_availability_and_create_models,
    _validate_embedding_model,
    _validate_foundation_model,
)


class TestValidateFoundationModel:
    """Test _validate_foundation_model function."""

    def test_returns_true_when_model_responds(self):
        """Test that validation returns True when model responds successfully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Mock(choices=[])

        model = OGXFoundationModel(model_id="test-model", client=mock_client)

        result = _validate_foundation_model(model)

        assert result is True
        mock_client.chat.completions.create.assert_called_once()

    def test_returns_false_when_model_fails(self):
        """Test that validation returns False when model raises exception."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Model error")

        model = OGXFoundationModel(model_id="test-model", client=mock_client)

        result = _validate_foundation_model(model)

        assert result is False


class TestValidateEmbeddingModel:
    """Test _validate_embedding_model function."""

    def test_returns_true_when_model_responds(self):
        """Test that validation returns True when model responds successfully."""
        mock_client = MagicMock()
        mock_data = Mock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response

        model = OGXEmbeddingModel(
            model_id="test-model",
            client=mock_client,
            params=OGXEmbeddingParams(embedding_dimension=768, context_length=512),
        )

        result = _validate_embedding_model(model)

        assert result is True

    def test_returns_false_when_model_fails(self):
        """Test that validation returns False when model raises exception."""
        mock_client = MagicMock()

        model = OGXEmbeddingModel(
            model_id="test-model",
            client=mock_client,
            params=OGXEmbeddingParams(embedding_dimension=768, context_length=512),
        )

        mock_client.embeddings.create.side_effect = Exception("Model error")

        result = _validate_embedding_model(model)

        assert result is False


class TestGetDefaultOGXModels:
    """Test _get_default_ogx_models function."""

    def _make_llm_mock(self, model_id: str) -> Mock:
        m = Mock()
        m.id = model_id
        m.custom_metadata = {"model_type": "llm"}
        return m

    def _make_embedding_mock(self, model_id: str, dim: int = 768, ctx: int = 512) -> Mock:
        m = Mock()
        m.id = model_id
        m.custom_metadata = {"model_type": "embedding", "embedding_dimension": dim, "context_length": ctx}
        return m

    def _client(self, *models) -> MagicMock:
        mock_client = MagicMock()
        mock_client.models.list.return_value.data = list(models)
        return mock_client

    def test_returns_registered_foundation_and_embedding_models(self):
        """Returns model objects split by type without validating them."""
        mock_llm = self._make_llm_mock("test-llm")
        mock_emb = self._make_embedding_mock("test-emb")
        client = self._client(mock_llm, mock_emb)

        result = _get_default_ogx_models(client)

        assert result["foundation_models"] == [mock_llm]
        assert result["embedding_models"] == [mock_emb]

    def test_does_not_validate_models(self, mocker):
        """_get_default_ogx_models must not call validation functions."""
        mock_llm = self._make_llm_mock("test-llm")
        mock_emb = self._make_embedding_mock("test-emb")
        client = self._client(mock_llm, mock_emb)

        validate_fm = mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_foundation_model")
        validate_em = mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_embedding_model")

        _get_default_ogx_models(client)

        validate_fm.assert_not_called()
        validate_em.assert_not_called()

    def test_raises_when_no_llm_registered(self):
        """Error when OGX has no models of type 'llm'."""
        mock_emb = self._make_embedding_mock("test-emb")
        client = self._client(mock_emb)

        with pytest.raises(SearchSpaceValueError, match="no registered models of type 'llm'"):
            _get_default_ogx_models(client)

    def test_raises_when_no_embedding_registered(self):
        """Error when OGX has no models of type 'embedding'."""
        mock_llm = self._make_llm_mock("test-llm")
        client = self._client(mock_llm)

        with pytest.raises(SearchSpaceValueError, match="no registered models of type 'embedding'"):
            _get_default_ogx_models(client)

    def test_filters_by_model_type(self):
        """Models with an unrecognised model_type are excluded from both lists."""
        mock_llm = self._make_llm_mock("test-llm")
        mock_emb = self._make_embedding_mock("test-emb")
        untyped = Mock()
        untyped.id = "unknown"
        untyped.custom_metadata = {"model_type": "unknown_type"}
        client = self._client(mock_llm, mock_emb, untyped)

        result = _get_default_ogx_models(client)

        assert len(result["foundation_models"]) == 1
        assert len(result["embedding_models"]) == 1


class TestValidateAvailabilityAndCreateModels:
    """Test _validate_availability_and_create_models function."""

    def _make_llm_registry_mock(self, model_id: str) -> Mock:
        m = Mock()
        m.id = model_id
        m.custom_metadata = {"model_type": "llm"}
        return m

    def _make_emb_registry_mock(self, model_id: str, dim: int = 768, ctx: int = 512) -> Mock:
        m = Mock()
        m.id = model_id
        m.custom_metadata = {"model_type": "embedding", "embedding_dimension": dim, "context_length": ctx}
        return m

    def test_returns_llm_instances_for_valid_models(self, mocker):
        """Returns OGXFoundationModel instances for each valid LLM."""
        mock_client = MagicMock()
        registered = [self._make_llm_registry_mock("llm-1"), self._make_llm_registry_mock("llm-2")]
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_foundation_model", return_value=True)

        result = _validate_availability_and_create_models(
            registered_models=registered,
            models_type="llm",
            client=mock_client,
            provided_models_ids=["llm-1", "llm-2"],
        )

        assert len(result) == 2
        assert all(isinstance(m, OGXFoundationModel) for m in result)
        assert {m.model_id for m in result} == {"llm-1", "llm-2"}

    def test_returns_embedding_instances_for_valid_models(self, mocker):
        """Returns OGXEmbeddingModel instances for each valid embedding model."""
        mock_client = MagicMock()
        registered = [self._make_emb_registry_mock("emb-1", dim=768, ctx=512)]
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_embedding_model", return_value=True)

        result = _validate_availability_and_create_models(
            registered_models=registered,
            models_type="embedding",
            client=mock_client,
            provided_models_ids=["emb-1"],
        )

        assert len(result) == 1
        assert isinstance(result[0], OGXEmbeddingModel)
        assert result[0].model_id == "emb-1"

    def test_raises_when_llm_not_registered(self, mocker):
        """Error when a requested LLM ID is not in the registered list."""
        mock_client = MagicMock()
        registered = [self._make_llm_registry_mock("llm-ok")]
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_foundation_model", return_value=True)

        with pytest.raises(SearchSpaceValueError, match=r"not registered in OGX.*llm-unknown"):
            _validate_availability_and_create_models(
                registered_models=registered,
                models_type="llm",
                client=mock_client,
                provided_models_ids=["llm-unknown"],
            )

    def test_raises_when_embedding_not_registered(self, mocker):
        """Error when a requested embedding ID is not in the registered list."""
        mock_client = MagicMock()
        registered = [self._make_emb_registry_mock("emb-ok")]
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_embedding_model", return_value=True)

        with pytest.raises(SearchSpaceValueError, match=r"not registered in OGX.*emb-unknown"):
            _validate_availability_and_create_models(
                registered_models=registered,
                models_type="embedding",
                client=mock_client,
                provided_models_ids=["emb-unknown"],
            )

    def test_raises_when_llm_does_not_respond(self, mocker):
        """Error when a registered LLM fails validation."""
        mock_client = MagicMock()
        registered = [self._make_llm_registry_mock("llm-bad")]
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*llm-bad"):
            _validate_availability_and_create_models(
                registered_models=registered,
                models_type="llm",
                client=mock_client,
                provided_models_ids=["llm-bad"],
            )

    def test_raises_when_embedding_does_not_respond(self, mocker):
        """Error when a registered embedding model fails validation."""
        mock_client = MagicMock()
        registered = [self._make_emb_registry_mock("emb-bad")]
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_embedding_model", return_value=False)

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*emb-bad"):
            _validate_availability_and_create_models(
                registered_models=registered,
                models_type="embedding",
                client=mock_client,
                provided_models_ids=["emb-bad"],
            )

    def test_raises_when_embedding_instantiation_fails(self, mocker):
        """Exception during OGXEmbeddingModel.__init__ is treated as non-responding."""
        mock_client = MagicMock()
        registered = [self._make_emb_registry_mock("emb-broken")]
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils.OGXEmbeddingModel",
            side_effect=RuntimeError("Cannot connect"),
        )

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*emb-broken"):
            _validate_availability_and_create_models(
                registered_models=registered,
                models_type="embedding",
                client=mock_client,
                provided_models_ids=["emb-broken"],
            )

    def test_error_message_combines_unregistered_and_not_responding(self, mocker):
        """A single exception lists both unregistered and non-responding model IDs."""
        mock_client = MagicMock()
        registered = [self._make_llm_registry_mock("llm-bad")]
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError) as exc_info:
            _validate_availability_and_create_models(
                registered_models=registered,
                models_type="llm",
                client=mock_client,
                provided_models_ids=["llm-bad", "llm-unknown"],
            )

        error_msg = str(exc_info.value)
        assert "llm-bad" in error_msg
        assert "llm-unknown" in error_msg
        assert "do not respond" in error_msg
        assert "not registered in OGX" in error_msg

    def test_only_validates_provided_models(self, mocker):
        """Models that are registered but not requested must not be validated."""
        mock_client = MagicMock()
        registered = [
            self._make_llm_registry_mock("llm-ok"),
            self._make_llm_registry_mock("llm-not-requested"),
        ]
        validate_mock = mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model", return_value=True
        )

        _validate_availability_and_create_models(
            registered_models=registered,
            models_type="llm",
            client=mock_client,
            provided_models_ids=["llm-ok"],
        )

        assert validate_mock.call_count == 1
        assert validate_mock.call_args[0][0].model_id == "llm-ok"
