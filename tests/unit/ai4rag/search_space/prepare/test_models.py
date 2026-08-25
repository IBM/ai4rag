# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for ai4rag.search_space.prepare.models.

Covers both operating modes of the shared helpers:

* **discovery** — instantiate models from bare ids by checking availability
  against the serving registry and validating responsiveness;
* **restore** — rebuild models from serialized report specs, reusing every
  stored parameter and skipping the registry round-trip.

A single OpenAI client now backs everything, so one mock
(:func:`_serving_client`) lists models *and* serves chat/embeddings for every
instantiated wrapper. Model ids are used verbatim, including their ``/``
characters.
"""

from unittest.mock import MagicMock, Mock

import pytest
from openai import OpenAI

from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel, OpenAIEmbeddingParams
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.prepare.models import (
    _list_maas_model_ids,
    _validate_embedding_model,
    _validate_foundation_model,
    get_embedding_models,
    get_foundation_models,
    serialize_model,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError

# Full model ids exactly as MaaS reports them, including the '/' path segments
# that are now used verbatim when calling chat/embeddings.
_FM_ID = "publishers/ai-eng-cracow/models/qwen3-8b-fp8-dynamic"
_EM_ID = "publishers/ai-eng-cracow/models/bge-m3"


def _make_model_mock(model_id: str) -> Mock:
    """Build a serving ``Model``-like mock exposing its full id verbatim."""
    m = Mock()
    m.id = model_id
    return m


def _serving_client(registered_ids: list[str], *, dim: int = 768) -> MagicMock:
    """Build a single MaaS client mock: lists models and serves chat/embeddings.

    A single client now backs everything, so the same mock must:

    - pass the ``isinstance(client, OpenAI)`` gate (via ``spec=OpenAI``);
    - report ``registered_ids`` verbatim from ``models.list().data``;
    - answer ``chat.completions.create`` with empty choices (inert foundation
      construction and validation); and
    - answer ``embeddings.create`` with a single vector of length ``dim`` so
      embedding dimension auto-detection succeeds.
    """
    client = MagicMock(spec=OpenAI)
    # spec=OpenAI does not expose instance attributes set in __init__, so set them explicitly.
    client.base_url = "https://maas.example.com/maas-api/v1"
    client.api_key = "secret-key"
    client.models.list.return_value.data = [_make_model_mock(mid) for mid in registered_ids]
    emb_response = Mock()
    emb_response.data = [Mock(embedding=[0.0] * dim)]
    client.embeddings.create.return_value = emb_response
    client.chat.completions.create.return_value = Mock(choices=[])
    return client


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


class TestListMaasModelIds:
    """Test _list_maas_model_ids function."""

    def test_returns_full_ids_verbatim(self):
        """Returns the verbatim id of every listed model, '/' characters intact."""
        client = MagicMock()
        client.models.list.return_value.data = [_make_model_mock(_FM_ID), _make_model_mock(_EM_ID)]

        assert _list_maas_model_ids(client) == {_FM_ID, _EM_ID}

    def test_returns_empty_set_when_no_models(self):
        """An empty deployment yields an empty set."""
        client = MagicMock()
        client.models.list.return_value.data = []
        assert _list_maas_model_ids(client) == set()


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

    def test_instantiates_available_id_on_shared_client(self, mocker):
        """A registered id is instantiated on the shared client, keeping its full id verbatim."""
        client = _serving_client([_FM_ID])
        mocker.patch("ai4rag.search_space.prepare.models._validate_foundation_model", return_value=True)

        result = get_foundation_models(client, [_FM_ID])

        assert [m.model_id for m in result] == [_FM_ID]
        assert all(isinstance(m, OpenAIFoundationModel) for m in result)
        assert all(m.client is client for m in result)

    def test_raises_when_model_not_available(self, mocker):
        """A requested id absent from the registry raises before instantiation."""
        client = _serving_client([_FM_ID])
        mocker.patch("ai4rag.search_space.prepare.models._validate_foundation_model", return_value=True)

        with pytest.raises(SearchSpaceValueError, match=r"not available.*llm-unknown"):
            get_foundation_models(client, ["publishers/ns/models/llm-unknown"])

    def test_raises_when_model_does_not_respond(self, mocker):
        """An available but non-responding model raises."""
        client = _serving_client([_FM_ID])
        mocker.patch("ai4rag.search_space.prepare.models._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError, match=r"do not respond"):
            get_foundation_models(client, [_FM_ID])

    def test_error_message_combines_unavailable_and_not_responding(self, mocker):
        """A single exception lists both unavailable and non-responding model ids."""
        available_but_broken = _FM_ID
        unknown = "publishers/ns/models/llm-unknown"
        client = _serving_client([available_but_broken])
        mocker.patch("ai4rag.search_space.prepare.models._validate_foundation_model", return_value=False)

        with pytest.raises(SearchSpaceValueError) as exc_info:
            get_foundation_models(client, [available_but_broken, unknown])

        error_msg = str(exc_info.value)
        assert available_but_broken in error_msg
        assert unknown in error_msg
        assert "do not respond" in error_msg
        assert "not available on the serving endpoint" in error_msg

    def test_only_validates_provided_models(self, mocker):
        """Models that are registered but not requested must not be validated."""
        other = "publishers/ns/models/llm-not-requested"
        client = _serving_client([_FM_ID, other])
        validate_mock = mocker.patch("ai4rag.search_space.prepare.models._validate_foundation_model", return_value=True)

        get_foundation_models(client, [_FM_ID])

        assert validate_mock.call_count == 1
        assert validate_mock.call_args[0][0].model_id == _FM_ID

    def test_validate_false_skips_responsiveness_check(self, mocker):
        """With validate=False no responsiveness probe runs, even in discovery mode."""
        client = _serving_client([_FM_ID])
        validate_mock = mocker.patch(
            "ai4rag.search_space.prepare.models._validate_foundation_model", return_value=False
        )

        result = get_foundation_models(client, [_FM_ID], validate=False)

        assert [m.model_id for m in result] == [_FM_ID]
        validate_mock.assert_not_called()

    def test_non_openai_client_raises(self):
        """A client that is not an OpenAI instance raises a clear error."""
        with pytest.raises(SearchSpaceValueError, match="Unrecognized client type"):
            get_foundation_models(MagicMock(spec=object), [_FM_ID])

    def test_empty_input_returns_empty_list_without_listing(self):
        """No entries means no work and no registry round-trip."""
        client = _serving_client([_FM_ID])
        assert get_foundation_models(client, []) == []
        client.models.list.assert_not_called()


class TestGetEmbeddingModelsDiscovery:
    """Instantiate embedding models from bare ids via the serving registry."""

    def test_returns_instances_with_auto_detected_params(self):
        """Embedding models carry no metadata, so dimension/context are auto-detected."""
        client = _serving_client([_EM_ID], dim=1024)

        result = get_embedding_models(client, [_EM_ID])

        assert len(result) == 1
        assert isinstance(result[0], OpenAIEmbeddingModel)
        assert result[0].client is client
        assert result[0].params.embedding_dimension == 1024

    def test_auto_detection_failure_treated_as_not_responding(self):
        """A failure during auto-detection is surfaced as a non-responding model."""
        client = _serving_client([_EM_ID])
        client.embeddings.create.side_effect = Exception("Cannot connect")

        with pytest.raises(SearchSpaceValueError, match=r"do not respond"):
            get_embedding_models(client, [_EM_ID])


# ---------------------------------------------------------------------------
# Restore mode
# ---------------------------------------------------------------------------


class TestRestoreFromSpec:
    """Rebuild models from serialized report specs, reusing all stored settings."""

    def test_foundation_spec_reuses_params_language_and_prompts(self):
        """A foundation spec restores its params, language and prompts verbatim."""
        client = _serving_client([])
        spec = {
            "model_id": _FM_ID,
            "type": "generation",
            "params": {"max_completion_tokens": 123, "temperature": 0.4},
            "language": {"code": "de", "name": "German"},
            "system_message_text": "sys {reference_documents} {question}",
            "user_message_text": "usr {reference_documents} {question}",
            "context_template_text": "ctx {document}",
        }

        result = get_foundation_models(client, [spec], validate=False)

        # Restore binds to the shared client and never lists the registry.
        client.models.list.assert_not_called()
        (model,) = result
        assert model.client is client
        assert model.model_id == _FM_ID
        assert model.params.max_completion_tokens == 123
        assert model.params.temperature == 0.4
        assert model.language.code == "de"
        assert model.language.name == "German"
        assert model.system_message_text == "sys {reference_documents} {question}"
        assert model.context_template_text == "ctx {document}"

    def test_embedding_spec_reuses_params_without_detection(self):
        """A restored embedding spec reuses stored params and never probes for them."""
        client = _serving_client([])
        spec = {
            "model_id": _EM_ID,
            "type": "embedding",
            "params": {"embedding_dimension": 1024, "context_length": 8192},
        }

        (model,) = get_embedding_models(client, [spec], validate=False)

        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.client is client
        assert model.params.embedding_dimension == 1024
        assert model.params.context_length == 8192
        # No auto-detection round-trips and no registry listing.
        client.embeddings.create.assert_not_called()
        client.models.list.assert_not_called()

    def test_mixed_ids_and_specs_lists_registry_once(self, mocker):
        """A mix of bare ids and full specs resolves; the registry is listed for the id only."""
        client = _serving_client([_FM_ID])
        mocker.patch("ai4rag.search_space.prepare.models._validate_foundation_model", return_value=True)
        spec = {"model_id": "publishers/ns/models/llm-old", "params": {"temperature": 0.2}}

        result = get_foundation_models(client, [spec, _FM_ID])

        assert {m.model_id for m in result} == {"publishers/ns/models/llm-old", _FM_ID}
        client.models.list.assert_called_once()

    def test_spec_without_model_id_raises(self):
        """A spec missing model_id is rejected."""
        client = _serving_client([])
        with pytest.raises(SearchSpaceValueError, match="must include a non-empty 'model_id'"):
            get_foundation_models(client, [{"params": {"temperature": 0.1}}], validate=False)


# ---------------------------------------------------------------------------
# Serialization (the write mirror of restore)
# ---------------------------------------------------------------------------


class TestSerializeModel:
    """serialize_model must produce specs that restore back to equivalent models."""

    def test_foundation_model_serializes_and_round_trips(self):
        """A foundation model serializes to a spec that restores to the same model."""
        client = _serving_client([])
        spec = {
            "model_id": _FM_ID,
            "type": "generation",
            "params": {"temperature": 0.4},
            "language": {"code": "de", "name": "German"},
            "system_message_text": "sys {reference_documents} {question}",
            "user_message_text": "usr {reference_documents} {question}",
            "context_template_text": "ctx {document}",
        }
        (model,) = get_foundation_models(client, [spec], validate=False)

        out = serialize_model(model)

        assert out["model_id"] == _FM_ID
        assert out["type"] == "generation"
        assert out["params"]["temperature"] == 0.4
        assert out["language"]["code"] == "de"
        assert out["language"]["name"] == "German"
        assert out["system_message_text"] == spec["system_message_text"]
        assert out["context_template_text"] == spec["context_template_text"]

        # Restoring the serialized spec yields an equivalent model.
        (restored,) = get_foundation_models(client, [out], validate=False)
        assert restored.model_id == model.model_id
        assert restored.params.temperature == model.params.temperature
        assert restored.language.code == model.language.code

    def test_embedding_model_serializes_without_language_or_prompts(self):
        """An embedding spec carries params only — no language or prompt keys."""
        client = _serving_client([])
        spec = {
            "model_id": _EM_ID,
            "type": "embedding",
            "params": {"embedding_dimension": 1024, "context_length": 8192},
        }
        (model,) = get_embedding_models(client, [spec], validate=False)

        out = serialize_model(model)

        assert out["model_id"] == _EM_ID
        assert out["type"] == "embedding"
        assert out["params"]["embedding_dimension"] == 1024
        assert out["params"]["context_length"] == 8192
        assert "language" not in out
        assert "system_message_text" not in out
        assert "context_template_text" not in out
