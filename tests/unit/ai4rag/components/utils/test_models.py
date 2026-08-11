# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for ai4rag.components.utils.models.

Covers both operating modes of the shared helpers:

* **discovery** — instantiate models from bare ids by listing the serving
  registry, deriving per-model endpoints, and validating responsiveness;
* **restore** — rebuild models from serialized report specs, reusing every
  stored parameter and skipping the registry round-trip.
"""

from unittest.mock import MagicMock, Mock

import pytest
from openai import OpenAI

from ai4rag.components.utils.models import (
    _list_maas_models,
    _model_owned_by,
    _short_model_id,
    _validate_embedding_model,
    _validate_foundation_model,
    get_embedding_models,
    get_foundation_models,
)
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel, OpenAIEmbeddingParams
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


def _make_model_mock(short_id: str, namespace: str = "ai-eng-cracow") -> Mock:
    """Build a serving ``Model``-like mock with the fully-qualified id and owned_by shape."""
    m = Mock()
    m.id = f"publishers/{namespace}/models/{short_id}"
    m.owned_by = f"{namespace}/{short_id}"
    return m


def _capable_model_client(dim: int = 768) -> MagicMock:
    """A per-model client mock supporting foundation chat and embedding auto-detection.

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


def _general_client(registered_short_ids: list[str]) -> MagicMock:
    """Build a general serving client mock that passes the ``isinstance(client, OpenAI)`` gate."""
    client = MagicMock(spec=OpenAI)
    # spec=OpenAI does not expose instance attributes set in __init__, so set them explicitly.
    client.base_url = "https://maas.example.com/maas-api/v1"
    client.api_key = "secret-key"
    client.models.list.return_value.data = [_make_model_mock(sid) for sid in registered_short_ids]
    return client


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


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
        assert _model_owned_by(_make_model_mock("qwen3", namespace="ns")) == "ns/qwen3"

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
        client = MagicMock()
        mock_llm = _make_model_mock("test-llm")
        mock_emb = _make_model_mock("test-emb")
        client.models.list.return_value.data = [mock_llm, mock_emb]

        assert _list_maas_models(client) == {"test-llm": mock_llm, "test-emb": mock_emb}

    def test_returns_empty_registry_when_no_models(self):
        """An empty deployment yields an empty registry."""
        client = MagicMock()
        client.models.list.return_value.data = []
        assert _list_maas_models(client) == {}


class TestValidateFoundationModel:
    """Test _validate_foundation_model function."""

    def test_returns_true_when_model_responds(self):
        """Validation returns True when the model responds successfully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = Mock(choices=[])
        mock_client.with_options.return_value = mock_client

        assert _validate_foundation_model(OpenAIFoundationModel(model_id="m", client=mock_client)) is True
        mock_client.chat.completions.create.assert_called_once()

    def test_returns_false_when_model_fails(self):
        """Validation returns False when the model raises."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Model error")
        mock_client.with_options.return_value = mock_client

        assert _validate_foundation_model(OpenAIFoundationModel(model_id="m", client=mock_client)) is False


class TestValidateEmbeddingModel:
    """Test _validate_embedding_model function."""

    def test_returns_true_when_model_responds(self):
        """Validation returns True when the model responds successfully."""
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        mock_client.with_options.return_value = mock_client

        model = OpenAIEmbeddingModel(
            model_id="m", client=mock_client, params=OpenAIEmbeddingParams(embedding_dimension=768, context_length=1024)
        )
        assert _validate_embedding_model(model) is True

    def test_returns_false_when_model_fails(self):
        """Validation returns False when the model raises."""
        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        model = OpenAIEmbeddingModel(
            model_id="m", client=mock_client, params=OpenAIEmbeddingParams(embedding_dimension=768, context_length=1024)
        )
        mock_client.embeddings.create.side_effect = Exception("Model error")
        assert _validate_embedding_model(model) is False


# ---------------------------------------------------------------------------
# Discovery mode
# ---------------------------------------------------------------------------


class TestGetFoundationModelsDiscovery:
    """Instantiate foundation models from bare ids via the serving registry."""

    def test_derives_per_model_url_and_reuses_api_key(self, mocker):
        """Each model gets its own client at ``{scheme}://{netloc}/{owned_by}/v1`` with the shared key."""
        client = _general_client(["qwen3"])
        create_client = mocker.patch(
            "ai4rag.components.utils.models.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        mocker.patch("ai4rag.components.utils.models._validate_foundation_model", return_value=True)

        result = get_foundation_models(client, ["qwen3"])

        create_client.assert_called_once_with(
            base_url="https://maas.example.com/ai-eng-cracow/qwen3/v1",
            api_key="secret-key",
        )
        assert [m.model_id for m in result] == ["qwen3"]
        assert all(isinstance(m, OpenAIFoundationModel) for m in result)

    def test_raises_when_model_not_available(self, mocker):
        """A requested id absent from the registry raises before instantiation."""
        client = _general_client(["llm-ok"])
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=_capable_model_client())
        mocker.patch("ai4rag.components.utils.models._validate_foundation_model", return_value=True)

        with pytest.raises(SearchSpaceValueError, match=r"not available.*llm-unknown"):
            get_foundation_models(client, ["llm-unknown"])

    def test_raises_when_model_does_not_respond(self, mocker):
        """An available but non-responding model raises."""
        client = _general_client(["llm-bad"])
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=_capable_model_client())
        mocker.patch("ai4rag.components.utils.models._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*llm-bad"):
            get_foundation_models(client, ["llm-bad"])

    def test_error_message_combines_unavailable_and_not_responding(self, mocker):
        """A single exception lists both unavailable and non-responding model ids."""
        client = _general_client(["llm-bad"])
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=_capable_model_client())
        mocker.patch("ai4rag.components.utils.models._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError) as exc_info:
            get_foundation_models(client, ["llm-bad", "llm-unknown"])

        error_msg = str(exc_info.value)
        assert "llm-bad" in error_msg
        assert "llm-unknown" in error_msg
        assert "do not respond" in error_msg
        assert "not available on the serving endpoint" in error_msg

    def test_only_validates_provided_models(self, mocker):
        """Models that are registered but not requested must not be validated."""
        client = _general_client(["llm-ok", "llm-not-requested"])
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=_capable_model_client())
        validate_mock = mocker.patch("ai4rag.components.utils.models._validate_foundation_model", return_value=True)

        get_foundation_models(client, ["llm-ok"])

        assert validate_mock.call_count == 1
        assert validate_mock.call_args[0][0].model_id == "llm-ok"

    def test_validate_false_skips_responsiveness_check(self, mocker):
        """With validate=False no responsiveness probe runs, even in discovery mode."""
        client = _general_client(["llm-ok"])
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=_capable_model_client())
        validate_mock = mocker.patch("ai4rag.components.utils.models._validate_foundation_model", return_value=False)

        result = get_foundation_models(client, ["llm-ok"], validate=False)

        assert [m.model_id for m in result] == ["llm-ok"]
        validate_mock.assert_not_called()

    def test_non_openai_client_raises(self):
        """A client that is not an OpenAI instance raises a clear error."""
        with pytest.raises(SearchSpaceValueError, match="Unrecognized client type"):
            get_foundation_models(MagicMock(spec=object), ["llm-ok"])

    def test_empty_input_returns_empty_list_without_listing(self):
        """No entries means no work and no registry round-trip."""
        client = _general_client(["llm-ok"])
        assert get_foundation_models(client, []) == []
        client.models.list.assert_not_called()


class TestGetEmbeddingModelsDiscovery:
    """Instantiate embedding models from bare ids via the serving registry."""

    def test_returns_instances_with_auto_detected_params(self, mocker):
        """Embedding models carry no metadata, so dimension/context are auto-detected."""
        client = _general_client(["emb-1"])
        mocker.patch(
            "ai4rag.components.utils.models.create_maas_model_client",
            return_value=_capable_model_client(dim=1024),
        )

        result = get_embedding_models(client, ["emb-1"])

        assert len(result) == 1
        assert isinstance(result[0], OpenAIEmbeddingModel)
        assert result[0].params.embedding_dimension == 1024

    def test_auto_detection_failure_treated_as_not_responding(self, mocker):
        """A RuntimeError during auto-detection is surfaced as a non-responding model."""
        client = _general_client(["emb-broken"])
        broken_client = MagicMock()
        broken_client.embeddings.create.side_effect = Exception("Cannot connect")
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=broken_client)

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*emb-broken"):
            get_embedding_models(client, ["emb-broken"])


# ---------------------------------------------------------------------------
# Restore mode
# ---------------------------------------------------------------------------


class TestRestoreFromSpec:
    """Rebuild models from serialized report specs, reusing all stored settings."""

    def test_foundation_spec_reuses_url_params_language_and_prompts(self, mocker):
        """A foundation spec restores its endpoint, params, language and prompts verbatim."""
        client = _general_client([])
        create_client = mocker.patch(
            "ai4rag.components.utils.models.create_maas_model_client",
            return_value=_capable_model_client(),
        )
        spec = {
            "model_id": "qwen3",
            "type": "generation",
            "params": {"max_completion_tokens": 123, "temperature": 0.4},
            "base_url": "https://maas.example.com/ai-eng-cracow/qwen3/v1",
            "language": {"code": "de", "name": "German"},
            "system_message_text": "sys {reference_documents} {question}",
            "user_message_text": "usr {reference_documents} {question}",
            "context_template_text": "ctx {document}",
        }

        result = get_foundation_models(client, [spec], validate=False)

        # Restore uses the stored base_url and only the client's api key; the
        # registry is never listed.
        client.models.list.assert_not_called()
        create_client.assert_called_once_with(
            base_url="https://maas.example.com/ai-eng-cracow/qwen3/v1",
            api_key="secret-key",
        )
        (model,) = result
        assert model.model_id == "qwen3"
        assert model.params.max_completion_tokens == 123
        assert model.params.temperature == 0.4
        assert model.language.code == "de"
        assert model.language.name == "German"
        assert model.system_message_text == "sys {reference_documents} {question}"
        assert model.context_template_text == "ctx {document}"

    def test_embedding_spec_reuses_params_without_detection(self, mocker):
        """A restored embedding spec reuses stored params and never probes for them."""
        client = _general_client([])
        per_model_client = MagicMock()
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=per_model_client)
        spec = {
            "model_id": "bge-m3",
            "type": "embedding",
            "params": {"embedding_dimension": 1024, "context_length": 8192},
            "base_url": "https://maas.example.com/ai-eng-cracow/bge-m3/v1",
        }

        (model,) = get_embedding_models(client, [spec], validate=False)

        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.params.embedding_dimension == 1024
        assert model.params.context_length == 8192
        # No auto-detection round-trips against the per-model endpoint.
        per_model_client.embeddings.create.assert_not_called()
        client.models.list.assert_not_called()

    def test_mixed_ids_and_specs_lists_registry_once(self, mocker):
        """A mix of bare ids and full specs still resolves; the registry is listed for the id only."""
        client = _general_client(["llm-new"])
        mocker.patch("ai4rag.components.utils.models.create_maas_model_client", return_value=_capable_model_client())
        mocker.patch("ai4rag.components.utils.models._validate_foundation_model", return_value=True)
        spec = {"model_id": "llm-old", "base_url": "https://maas.example.com/ns/llm-old/v1"}

        result = get_foundation_models(client, [spec, "llm-new"])

        assert {m.model_id for m in result} == {"llm-old", "llm-new"}
        client.models.list.assert_called_once()

    def test_spec_without_model_id_raises(self):
        """A spec missing model_id is rejected."""
        client = _general_client([])
        with pytest.raises(SearchSpaceValueError, match="must include a non-empty 'model_id'"):
            get_foundation_models(client, [{"base_url": "https://x/v1"}], validate=False)
