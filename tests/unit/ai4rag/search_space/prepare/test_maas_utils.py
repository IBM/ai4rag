# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock, Mock

import pytest

from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel, OpenAIEmbeddingParams
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.prepare.maas_utils import (
    SearchSpaceValueError,
    _build_valid_and_invalid_models,
    _list_maas_models,
    _model_owned_by,
    _short_model_id,
    _validate_availability_and_create_models,
    _validate_embedding_model,
    _validate_foundation_model,
)


def _make_model_mock(short_id: str, namespace: str = "ai-eng-cracow") -> Mock:
    """Build a MaaS ``Model``-like mock with the fully-qualified id and owned_by shape."""
    m = Mock()
    m.id = f"publishers/{namespace}/models/{short_id}"
    m.owned_by = f"{namespace}/{short_id}"
    return m


def _capable_model_client(dim: int = 768) -> MagicMock:
    """A per-model client mock that supports foundation chat and embedding auto-detection.

    - ``embeddings.create(...).data`` yields a single vector of length ``dim`` so
      dimension auto-detection succeeds and context-length probing never raises.
    - ``chat.completions.create(...)`` returns empty choices so foundation
      construction and validation are inert.
    """
    client = MagicMock()
    emb_response = Mock()
    emb_response.data = [Mock(embedding=[0.0] * dim)]
    client.embeddings.create.return_value = emb_response
    client.chat.completions.create.return_value = Mock(choices=[])
    return client


class TestShortModelId:
    """Test _short_model_id function."""

    def test_returns_last_path_segment(self):
        """The usable short id is the final segment of the fully-qualified id."""
        assert _short_model_id("publishers/ai-eng-cracow/models/qwen3-8b-fp8-dynamic") == "qwen3-8b-fp8-dynamic"

    def test_returns_input_when_already_short(self):
        """An id without path separators is returned unchanged."""
        assert _short_model_id("bge-m3") == "bge-m3"


class TestModelOwnedBy:
    """Test _model_owned_by function."""

    def test_returns_owned_by_when_present(self):
        """Uses the model's ``owned_by`` attribute when populated."""
        model = _make_model_mock("qwen3", namespace="ns")
        assert _model_owned_by(model) == "ns/qwen3"

    def test_derives_from_id_when_owned_by_missing(self):
        """Falls back to ``<namespace>/<short-id>`` derived from the id."""
        model = Mock()
        model.owned_by = None
        model.id = "publishers/ai-eng-cracow/models/bge-m3"
        assert _model_owned_by(model) == "ai-eng-cracow/bge-m3"

    def test_returns_id_when_unparseable(self):
        """Returns the raw id when neither owned_by nor a parseable id is available."""
        model = Mock()
        model.owned_by = None
        model.id = "bge-m3"
        assert _model_owned_by(model) == "bge-m3"


class TestListMaasModels:
    """Test _list_maas_models function."""

    def test_returns_registry_keyed_by_short_id(self):
        """Returns every listed model keyed by its short id, without splitting by type."""
        mock_llm = _make_model_mock("test-llm")
        mock_emb = _make_model_mock("test-emb")
        client = MagicMock()
        client.models.list.return_value.data = [mock_llm, mock_emb]

        registry = _list_maas_models(client)

        assert registry == {"test-llm": mock_llm, "test-emb": mock_emb}

    def test_returns_empty_registry_when_no_models(self):
        """An empty MaaS deployment yields an empty registry."""
        client = MagicMock()
        client.models.list.return_value.data = []

        assert _list_maas_models(client) == {}


class TestValidateFoundationModel:
    """Test _validate_foundation_model function."""

    def test_returns_true_when_model_responds(self):
        """Test that validation returns True when model responds successfully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Mock(choices=[])
        mock_client.with_options.return_value = mock_client

        model = OpenAIFoundationModel(model_id="test-model", client=mock_client)

        result = _validate_foundation_model(model)

        assert result is True
        mock_client.chat.completions.create.assert_called_once()

    def test_returns_false_when_model_fails(self):
        """Test that validation returns False when model raises exception."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Model error")
        mock_client.with_options.return_value = mock_client

        model = OpenAIFoundationModel(model_id="test-model", client=mock_client)

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
        mock_client.with_options.return_value = mock_client

        model = OpenAIEmbeddingModel(
            model_id="test-model",
            client=mock_client,
            params=OpenAIEmbeddingParams(embedding_dimension=768, context_length=1024),
        )

        result = _validate_embedding_model(model)

        assert result is True

    def test_returns_false_when_model_fails(self):
        """Test that validation returns False when model raises exception."""
        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client

        model = OpenAIEmbeddingModel(
            model_id="test-model",
            client=mock_client,
            params=OpenAIEmbeddingParams(embedding_dimension=768, context_length=1024),
        )

        mock_client.embeddings.create.side_effect = Exception("Model error")

        result = _validate_embedding_model(model)

        assert result is False


class TestBuildValidAndInvalidModels:
    """Test _build_valid_and_invalid_models function."""

    def test_derives_per_model_url_and_reuses_api_key(self, mocker):
        """Each model gets its own client at ``{scheme}://{netloc}/{owned_by}/v1`` with the shared key."""
        general_client = MagicMock()
        general_client.base_url = "https://maas.example.com/maas-api/v1"
        general_client.api_key = "secret-key"
        registry = {"qwen3": _make_model_mock("qwen3", namespace="ai-eng-cracow")}
        create_client = mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_foundation_model", return_value=True)

        valid, invalid = _build_valid_and_invalid_models(
            candidate_ids=["qwen3"],
            registered_models_as_dict=registry,
            models_type="llm",
            client=general_client,
        )

        create_client.assert_called_once_with(
            base_url="https://maas.example.com/ai-eng-cracow/qwen3/v1",
            api_key="secret-key",
        )
        assert [m.model_id for m in valid] == ["qwen3"]
        assert invalid == []

    def test_splits_valid_and_invalid_by_validation(self, mocker):
        """Models failing validation land in the invalid bucket, keyed by short id."""
        general_client = MagicMock()
        registry = {"ok": _make_model_mock("ok"), "bad": _make_model_mock("bad")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils._validate_foundation_model",
            side_effect=lambda model: model.model_id == "ok",
        )

        valid, invalid = _build_valid_and_invalid_models(
            candidate_ids=["ok", "bad"],
            registered_models_as_dict=registry,
            models_type="llm",
            client=general_client,
        )

        assert [m.model_id for m in valid] == ["ok"]
        assert invalid == ["bad"]

    def test_embedding_params_are_auto_detected(self, mocker):
        """Embedding models carry no MaaS metadata, so dimension/context are auto-detected."""
        general_client = MagicMock()
        registry = {"emb-1": _make_model_mock("emb-1")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(dim=1024),
        )

        valid, invalid = _build_valid_and_invalid_models(
            candidate_ids=["emb-1"],
            registered_models_as_dict=registry,
            models_type="embedding",
            client=general_client,
        )

        assert invalid == []
        assert len(valid) == 1
        assert isinstance(valid[0], OpenAIEmbeddingModel)
        assert valid[0].params.embedding_dimension == 1024


class TestValidateAvailabilityAndCreateModels:
    """Test _validate_availability_and_create_models function."""

    def test_returns_llm_instances_for_valid_models(self, mocker):
        """Returns OpenAIFoundationModel instances for each valid LLM."""
        general_client = MagicMock()
        registry = {"llm-1": _make_model_mock("llm-1"), "llm-2": _make_model_mock("llm-2")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_foundation_model", return_value=True)

        result = _validate_availability_and_create_models(
            registered_models=registry,
            models_type="llm",
            client=general_client,
            provided_models_ids=["llm-1", "llm-2"],
        )

        assert len(result) == 2
        assert all(isinstance(m, OpenAIFoundationModel) for m in result)
        assert {m.model_id for m in result} == {"llm-1", "llm-2"}

    def test_returns_embedding_instances_for_valid_models(self, mocker):
        """Returns OpenAIEmbeddingModel instances for each valid embedding model."""
        general_client = MagicMock()
        registry = {"emb-1": _make_model_mock("emb-1")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_embedding_model", return_value=True)

        result = _validate_availability_and_create_models(
            registered_models=registry,
            models_type="embedding",
            client=general_client,
            provided_models_ids=["emb-1"],
        )

        assert len(result) == 1
        assert isinstance(result[0], OpenAIEmbeddingModel)
        assert result[0].model_id == "emb-1"

    def test_raises_when_llm_not_available(self, mocker):
        """Error when a requested LLM ID is not in the MaaS registry."""
        general_client = MagicMock()
        registry = {"llm-ok": _make_model_mock("llm-ok")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_foundation_model", return_value=True)

        with pytest.raises(SearchSpaceValueError, match=r"not available in MaaS.*llm-unknown"):
            _validate_availability_and_create_models(
                registered_models=registry,
                models_type="llm",
                client=general_client,
                provided_models_ids=["llm-unknown"],
            )

    def test_raises_when_embedding_not_available(self, mocker):
        """Error when a requested embedding ID is not in the MaaS registry."""
        general_client = MagicMock()
        registry = {"emb-ok": _make_model_mock("emb-ok")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_embedding_model", return_value=True)

        with pytest.raises(SearchSpaceValueError, match=r"not available in MaaS.*emb-unknown"):
            _validate_availability_and_create_models(
                registered_models=registry,
                models_type="embedding",
                client=general_client,
                provided_models_ids=["emb-unknown"],
            )

    def test_raises_when_llm_does_not_respond(self, mocker):
        """Error when an available LLM fails validation."""
        general_client = MagicMock()
        registry = {"llm-bad": _make_model_mock("llm-bad")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*llm-bad"):
            _validate_availability_and_create_models(
                registered_models=registry,
                models_type="llm",
                client=general_client,
                provided_models_ids=["llm-bad"],
            )

    def test_raises_when_embedding_does_not_respond(self, mocker):
        """Error when an available embedding model fails validation."""
        general_client = MagicMock()
        registry = {"emb-bad": _make_model_mock("emb-bad")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_embedding_model", return_value=False)

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*emb-bad"):
            _validate_availability_and_create_models(
                registered_models=registry,
                models_type="embedding",
                client=general_client,
                provided_models_ids=["emb-bad"],
            )

    def test_raises_when_embedding_auto_detection_fails(self, mocker):
        """A RuntimeError during OpenAIEmbeddingModel auto-detection is treated as non-responding."""
        general_client = MagicMock()
        registry = {"emb-broken": _make_model_mock("emb-broken")}
        broken_client = MagicMock()
        broken_client.embeddings.create.side_effect = Exception("Cannot connect")
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=broken_client,
        )

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*emb-broken"):
            _validate_availability_and_create_models(
                registered_models=registry,
                models_type="embedding",
                client=general_client,
                provided_models_ids=["emb-broken"],
            )

    def test_error_message_combines_unavailable_and_not_responding(self, mocker):
        """A single exception lists both unavailable and non-responding model IDs."""
        general_client = MagicMock()
        registry = {"llm-bad": _make_model_mock("llm-bad")}
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.search_space.prepare.maas_utils._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError) as exc_info:
            _validate_availability_and_create_models(
                registered_models=registry,
                models_type="llm",
                client=general_client,
                provided_models_ids=["llm-bad", "llm-unknown"],
            )

        error_msg = str(exc_info.value)
        assert "llm-bad" in error_msg
        assert "llm-unknown" in error_msg
        assert "do not respond" in error_msg
        assert "not available in MaaS" in error_msg

    def test_only_validates_provided_models(self, mocker):
        """Models that are available but not requested must not be validated."""
        general_client = MagicMock()
        registry = {
            "llm-ok": _make_model_mock("llm-ok"),
            "llm-not-requested": _make_model_mock("llm-not-requested"),
        }
        mocker.patch(
            "ai4rag.search_space.prepare.maas_utils.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        validate_mock = mocker.patch(
            "ai4rag.search_space.prepare.maas_utils._validate_foundation_model", return_value=True
        )

        _validate_availability_and_create_models(
            registered_models=registry,
            models_type="llm",
            client=general_client,
            provided_models_ids=["llm-ok"],
        )

        assert validate_mock.call_count == 1
        assert validate_mock.call_args[0][0].model_id == "llm-ok"
