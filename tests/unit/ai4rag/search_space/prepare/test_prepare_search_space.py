# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for prepare_search_space module."""

import json
from copy import copy
from unittest.mock import MagicMock, Mock

import httpx
import pandas as pd
import pytest
from openai import AuthenticationError, BadRequestError, OpenAI
from pydantic import ValidationError

from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.prepare import (
    build_search_space_report,
    get_embedding_models,
    prepare_search_space_with_maas,
    prepare_search_space_with_models,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


def _make_model(model_id: str) -> Mock:
    """Build a MaaS ``Model``-like mock reporting its id verbatim.

    Ids are used exactly as the registry reports them, so the availability check
    matches the payload's ``model_id`` directly (no path stripping).
    """
    m = Mock()
    m.id = model_id
    return m


def _setup_client(mocker, registered_ids, *, foundation_valid=True, embedding_valid=True, dim: int = 768) -> MagicMock:
    """Build the single MaaS client mock and patch the validation helpers.

    One client now backs everything, so the same mock lists models *and* serves
    chat/embeddings — mirroring the real single-endpoint MaaS deployment.

    Parameters
    ----------
    mocker
        pytest-mock fixture.
    registered_ids : list[str]
        Ids of the models the MaaS registry should report, verbatim.
    foundation_valid, embedding_valid : bool | callable
        Return value (or ``side_effect``) for the corresponding validation function.
    dim : int, default=768
        Length of the embedding vector returned during auto-detection.
    """
    client = MagicMock(spec=OpenAI)
    # spec=OpenAI does not expose instance attributes set in __init__, so set them explicitly.
    client.base_url = "https://maas.example.com/maas-api/v1"
    client.api_key = "secret-key"
    client.timeout = httpx.Timeout(600)
    client.with_options.return_value = client
    client.models.list.return_value.data = [_make_model(mid) for mid in registered_ids]
    emb_response = Mock()
    emb_response.data = [Mock(index=0, embedding=[0.0] * dim)]
    client.embeddings.create.return_value = emb_response
    client.chat.completions.create.return_value = Mock(choices=[])

    fm_kwarg = {"side_effect": foundation_valid} if callable(foundation_valid) else {"return_value": foundation_valid}
    em_kwarg = {"side_effect": embedding_valid} if callable(embedding_valid) else {"return_value": embedding_valid}
    mocker.patch("ai4rag.search_space.prepare.models._validate_foundation_model", **fm_kwarg)
    mocker.patch("ai4rag.search_space.prepare.models._validate_embedding_model", **em_kwarg)
    return client


def _payload(foundation_ids=("default-llm",), embedding_ids=("default-embedding",), **extra):
    """Build a MaaS payload declaring foundation and embedding model ids per type."""
    payload: dict = {}
    if foundation_ids is not None:
        payload["foundation_models"] = [{"model_id": mid} for mid in foundation_ids]
    if embedding_ids is not None:
        payload["embedding_models"] = [{"model_id": mid} for mid in embedding_ids]
    payload.update(extra)
    return payload


class TestPrepareSearchSpaceWithMaas:
    """Test prepare_search_space_with_maas function."""

    def test_valid_payload_creates_search_space(self, mocker):
        """A payload declaring both model types yields a search space with both dimensions."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(), client)

        assert result is not None
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

    def test_custom_foundation_model_included(self, mocker):
        """The requested foundation model is instantiated and included."""
        client = _setup_client(mocker, ["custom-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(foundation_ids=["custom-llm"]), client)

        foundation_param = result["foundation_model"]
        assert len(foundation_param.values) == 1
        assert foundation_param.values[0].model_id == "custom-llm"

    def test_custom_embedding_model_included(self, mocker):
        """The requested embedding model is instantiated and included."""
        client = _setup_client(mocker, ["default-llm", "custom-embedding"])

        result = prepare_search_space_with_maas(_payload(embedding_ids=["custom-embedding"]), client)

        embedding_param = result["embedding_model"]
        assert len(embedding_param.values) == 1
        assert embedding_param.values[0].model_id == "custom-embedding"

    def test_invalid_payload_raises_validation_error(self):
        """An unknown payload key raises a validation error before any I/O."""
        with pytest.raises(ValidationError, match="Unknown validation error|invalid_parameter"):
            prepare_search_space_with_maas({"invalid_parameter": "value"}, MagicMock())

    def test_non_openai_client_raises_error(self):
        """A client that is not an OpenAI instance raises a clear error."""
        mock_client = MagicMock(spec=object)  # Not an OpenAI client

        with pytest.raises(SearchSpaceValueError, match="Unrecognized client type"):
            prepare_search_space_with_maas(_payload(), mock_client)

    def test_missing_foundation_models_raises_error(self):
        """Omitting foundation_models is rejected: MaaS cannot infer model type."""
        client = MagicMock(spec=OpenAI)

        with pytest.raises(SearchSpaceValueError, match="Provide both 'foundation_models' and 'embedding_models'"):
            prepare_search_space_with_maas(_payload(foundation_ids=None), client)

    def test_missing_embedding_models_raises_error(self):
        """Omitting embedding_models is rejected: MaaS cannot infer model type."""
        client = MagicMock(spec=OpenAI)

        with pytest.raises(SearchSpaceValueError, match="Provide both 'foundation_models' and 'embedding_models'"):
            prepare_search_space_with_maas(_payload(embedding_ids=None), client)

    def test_search_space_includes_hybrid_params_by_default(self, mocker):
        """All supported vector store backends support hybrid search, so it is always included."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(), client)

        param_names = [p.name for p in result.params]
        assert "search_mode" in param_names
        assert "ranker_strategy" in param_names
        assert "ranker_k" in param_names
        assert "ranker_alpha" in param_names

        search_mode_param = result["search_mode"]
        assert "vector" in search_mode_param.values
        assert "hybrid" in search_mode_param.values

    def test_neo4j_search_space_is_graph_only(self, mocker):
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(), client, vector_store_type="neo4j")

        assert result["search_mode"].values == ("graph",)
        assert "ranker_strategy" not in {param.name for param in result.params}

    def test_user_specifies_not_responding_foundation_model(self, mocker):
        """Error when a requested foundation model is available but does not respond."""
        client = _setup_client(
            mocker,
            ["llm-ok", "llm-bad", "default-embedding"],
            foundation_valid=lambda m, **_: m.model_id != "llm-bad",
        )

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*llm-bad"):
            prepare_search_space_with_maas(_payload(foundation_ids=["llm-bad"]), client)

    def test_user_specifies_not_responding_embedding_model(self, mocker):
        """Error when a requested embedding model is available but does not respond."""
        client = _setup_client(
            mocker,
            ["default-llm", "emb-ok", "emb-bad"],
            embedding_valid=lambda m, **_: m.model_id != "emb-bad",
        )

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*emb-bad"):
            prepare_search_space_with_maas(_payload(embedding_ids=["emb-bad"]), client)

    def test_user_specifies_unavailable_foundation_model(self, mocker):
        """Error when a requested foundation model is not available on the serving endpoint."""
        client = _setup_client(mocker, ["llm-ok", "default-embedding"])

        with pytest.raises(SearchSpaceValueError, match=r"not available.*llm-unknown"):
            prepare_search_space_with_maas(_payload(foundation_ids=["llm-unknown"]), client)

    def test_user_picks_available_model_while_others_fail(self, mocker):
        """A responding model succeeds even when other registered models do not respond."""
        client = _setup_client(
            mocker,
            ["llm-ok", "llm-bad", "default-embedding"],
            foundation_valid=lambda m, **_: m.model_id != "llm-bad",
        )

        result = prepare_search_space_with_maas(_payload(foundation_ids=["llm-ok"]), client)

        fm_ids = [m.model_id for m in result["foundation_model"].values]
        assert fm_ids == ["llm-ok"]

    def test_no_chunking_params_returns_default_chunking_dimensions(self, mocker):
        """Without chunking overrides the result includes default chunking_method and chunk_size dimensions."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(), client)

        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names
        assert "chunking_method" in param_names
        assert "chunk_size" in param_names

    def test_custom_chunking_methods_override_defaults(self, mocker):
        """chunking_methods in payload overrides the default chunking_method dimension."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(chunking_methods=["recursive"]), client)

        assert result["chunking_method"].values == ("recursive",)

    def test_custom_chunk_sizes_override_defaults(self, mocker):
        """chunk_sizes in payload overrides the default chunk_size dimension."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(chunk_sizes=[256, 512]), client)

        assert set(result["chunk_size"].values) == {256, 512}

    def test_both_chunking_params_applied_together(self, mocker):
        """Both chunking_methods and chunk_sizes in payload can be set simultaneously."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(chunking_methods=["hybrid"], chunk_sizes=[1024]), client)

        assert result["chunking_method"].values == ("hybrid",)
        assert result["chunk_size"].values == (1024,)

    def test_all_chunking_params_produce_non_empty_search_space(self, mocker):
        """chunking_methods, chunk_sizes, and chunk_overlaps together produce a non-empty search space."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(
            _payload(chunking_methods=["recursive"], chunk_sizes=[128, 256], chunk_overlaps=[32, 64]),
            client,
        )

        assert result["chunking_method"].values == ("recursive",)
        assert set(result["chunk_size"].values) == {128, 256}
        assert set(result["chunk_overlap"].values) == {32, 64}
        assert len(result.params) > 0

    def test_custom_chunk_overlaps_override_defaults(self, mocker):
        """chunk_overlaps in payload overrides the default chunk_overlap dimension."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(chunk_overlaps=[0, 128]), client)

        assert set(result["chunk_overlap"].values) == {0, 128}

    def test_unsupported_chunking_method_raises_error(self):
        """An unsupported chunking method raises ValidationError before any I/O."""
        with pytest.raises(ValidationError, match="Unsupported chunking methods"):
            prepare_search_space_with_maas({"chunking_methods": ["semantic"]}, MagicMock())

    def test_chunk_size_below_min_raises_error(self):
        """A chunk size below MIN_CHUNK_SIZE raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_maas({"chunk_sizes": [1]}, MagicMock())

    def test_chunk_size_above_max_raises_error(self):
        """A chunk size above MAX_CHUNK_SIZE raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_maas({"chunk_sizes": [99999]}, MagicMock())

    def test_chunk_overlap_below_min_raises_error(self):
        """A chunk overlap below MIN_CHUNK_OVERLAP raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_maas({"chunk_overlaps": [-1]}, MagicMock())

    def test_chunk_overlap_above_max_raises_error(self):
        """A chunk overlap above MAX_CHUNK_OVERLAP raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_maas({"chunk_overlaps": [99999]}, MagicMock())


@pytest.mark.parametrize(
    "provider,templates,expected",
    [
        ("milvus", None, ("agentic_rag",)),
        ("neo4j", None, ("agentic_graph_rag",)),
        ("milvus_lite", ["agentic_rag", "sequential_rag"], ("agentic_rag", "sequential_rag")),
        ("pgvector", ["sequential_rag"], ("sequential_rag",)),
    ],
)
def test_maas_template_dimension(mocker, provider, templates, expected):
    client = _setup_client(mocker, ["default-llm", "default-embedding"])
    result = prepare_search_space_with_maas(_payload(template_ids=templates), client, vector_store_type=provider)
    assert result["template_id"].values == expected


def test_maas_rejects_incompatible_templates_in_search_space(mocker):
    client = _setup_client(mocker, ["default-llm", "default-embedding"])
    with pytest.raises(SearchSpaceValueError, match="incompatible"):
        prepare_search_space_with_maas(_payload(template_ids=["sequential_rag"]), client, vector_store_type="neo4j")


def _direct_embedding_client(dimension: int) -> MagicMock:
    client = MagicMock(spec=OpenAI)
    client.base_url = "https://embedding.example.com/v1"
    client.timeout = httpx.Timeout(600)
    client.with_options.return_value = client
    client.embeddings.create.return_value = Mock(data=[Mock(index=0, embedding=[0.0] * dimension)])
    return client


@pytest.fixture
def direct_models():
    foundation_client = MagicMock(spec=OpenAI)
    foundation_client.api_key = "foundation-secret"
    foundation_client.base_url = "https://chat.example.com/gateway/v1"
    foundation_client.chat.completions.create.return_value = Mock(choices=[])
    embedding_client = _direct_embedding_client(768)
    embedding_client.api_key = "embedding-secret"
    return {
        "foundation_models": [OpenAIFoundationModel(foundation_client, "provider/chat", params={"temperature": 0.1})],
        "embedding_models": [
            OpenAIEmbeddingModel(
                embedding_client,
                "provider/embed",
                params={"embedding_dimension": 768, "context_length": 2048},
            )
        ],
    }


def test_direct_search_space_preserves_model_objects_and_overrides(direct_models):
    result = prepare_search_space_with_models(
        {
            "template_ids": ["sequential_rag"],
            "chunking_methods": ["recursive"],
            "chunk_sizes": [256],
            "chunk_overlaps": [64],
        },
        **direct_models,
    )

    for role, (model,) in direct_models.items():
        values = result[role.removesuffix("s")].values
        assert len(values) == 1
        assert values[0] is model
        model.client.models.list.assert_not_called()
    (foundation,) = direct_models["foundation_models"]
    (embedding,) = direct_models["embedding_models"]
    foundation_call = foundation.client.chat.completions.create
    embedding_call = embedding.client.embeddings.create
    foundation_call.assert_called_once()
    assert foundation_call.call_args.kwargs["model"] == foundation.model_id
    assert foundation_call.call_args.kwargs["messages"] == [{"role": "user", "content": "Hi"}]
    embedding_call.assert_called_once_with(
        model=embedding.model_id, input=["test"], extra_body={"truncate_prompt_tokens": -1}
    )
    assert result["template_id"].values == ("sequential_rag",)
    assert result["chunking_method"].values == ("recursive",)
    assert result["chunk_size"].values == (256,)
    assert result["chunk_overlap"].values == (64,)


def test_direct_language_detection_and_report(direct_models, tmp_path):
    (foundation,) = direct_models["foundation_models"]
    (embedding,) = direct_models["embedding_models"]
    prompt_before_detection = foundation.user_message_text
    completion = Mock()
    completion.message.content = '{"code": "pl"}'
    foundation.client.chat.completions.create.return_value = Mock(choices=[completion])

    result = prepare_search_space_with_models(
        {}, benchmark_data=pd.DataFrame({"question": ["Gdzie jest Warszawa?"]}), **direct_models
    )

    foundation.client.models.list.assert_not_called()
    probe, detection = foundation.client.chat.completions.create.call_args_list
    assert probe.kwargs["messages"] == [{"role": "user", "content": "Hi"}]
    assert detection.kwargs["model"] == foundation.model_id
    assert "response_format" in detection.kwargs
    embedding.client.embeddings.create.assert_called_once()
    assert foundation.language.code == "pl"
    assert foundation.user_message_text != prompt_before_detection

    report_path = tmp_path / "search-space.json"
    build_search_space_report(result).save_json(report_path)
    report_json = report_path.read_text(encoding="utf-8")
    for model in (foundation, embedding):
        assert model.client.api_key not in report_json
        assert model.client.base_url not in report_json
    report = json.loads(report_json)
    assert report["foundation_model"] == [
        {
            "model_id": "provider/chat",
            "type": "generation",
            "params": foundation.params.model_dump(),
            "language": {"code": "pl", "name": "Polish"},
            "system_message_text": foundation.system_message_text,
            "user_message_text": foundation.user_message_text,
            "context_template_text": foundation.context_template_text,
        }
    ]
    assert report["embedding_model"] == [
        {
            "model_id": "provider/embed",
            "type": "embedding",
            "params": {"embedding_dimension": 768, "context_length": 2048},
        }
    ]
    embedding.client.embeddings.create.reset_mock()
    (restored,) = get_embedding_models(embedding.client, report["embedding_model"], validate=False)
    embedding.client.embeddings.create.assert_not_called()
    assert restored.embed_query("restored query") == [0.0] * 768
    embedding.client.embeddings.create.assert_called_once_with(
        model=embedding.model_id, input=["restored query"], extra_body={"truncate_prompt_tokens": -1}
    )


def test_direct_language_detection_sanitizes_endpoint_errors(direct_models, caplog):
    (foundation,) = direct_models["foundation_models"]
    question = "Private benchmark question"
    error = BadRequestError(
        f"Could not process {question} with {foundation.client.api_key}",
        response=httpx.Response(400, request=httpx.Request("POST", foundation.client.base_url)),
        body=None,
    )
    foundation.client.chat.completions.create.side_effect = [Mock(choices=[]), error]
    original_language = foundation.language
    original_prompt = foundation.user_message_text

    prepare_search_space_with_models({}, benchmark_data=pd.DataFrame({"question": [question]}), **direct_models)

    assert foundation.client.chat.completions.create.call_count == 2
    assert "LLM language detection failed: BadRequestError, status_code=400" in caplog.text
    assert question not in caplog.text
    assert foundation.client.api_key not in caplog.text
    assert foundation.language is original_language
    assert foundation.user_message_text == original_prompt


def test_direct_search_space_explores_multiple_models_with_separate_clients(direct_models):
    foundation_models = direct_models["foundation_models"]
    embedding_models = direct_models["embedding_models"]
    local_chat_client = MagicMock(spec=OpenAI)
    local_chat_client.base_url = httpx.URL("http://localhost:8080/v1")
    cluster_embedding_client = _direct_embedding_client(1024)
    cluster_embedding_client.base_url = httpx.URL("http://embedding.models.svc.cluster.local:8080/v1")
    foundation_models.append(OpenAIFoundationModel(local_chat_client, "provider/multitask"))
    embedding_models.append(
        OpenAIEmbeddingModel(
            cluster_embedding_client,
            "provider/multitask",
            params={"embedding_dimension": 1024, "context_length": 4096},
        )
    )
    for model, code in zip(foundation_models, ("en", "pl"), strict=True):
        completion = Mock()
        completion.message.content = json.dumps({"code": code})
        model.client.chat.completions.create.return_value = Mock(choices=[completion])

    result = prepare_search_space_with_models(
        {}, benchmark_data=pd.DataFrame({"question": ["Where is Warsaw?"]}), **direct_models
    )

    for role, models in direct_models.items():
        for candidate, model in zip(result[role.removesuffix("s")].values, models, strict=True):
            assert candidate is model
    assert {(combo["foundation_model"], combo["embedding_model"]) for combo in result.combinations} == {
        (foundation, embedding) for foundation in foundation_models for embedding in embedding_models
    }
    for model, code in zip(foundation_models, ("en", "pl"), strict=True):
        model.client.models.list.assert_not_called()
        probe, detection = model.client.chat.completions.create.call_args_list
        assert probe.kwargs["model"] == detection.kwargs["model"] == model.model_id
        assert model.language.code == code
    for model in embedding_models:
        model.client.models.list.assert_not_called()
        model.client.embeddings.create.assert_called_once()
        assert model.client.embeddings.create.call_args.kwargs["model"] == model.model_id


def test_direct_search_space_reports_failed_probes_before_language_detection(direct_models, caplog):
    (foundation,) = direct_models["foundation_models"]
    (embedding,) = direct_models["embedding_models"]
    foundation.client.chat.completions.create.side_effect = AuthenticationError(
        f"Invalid credential {foundation.client.api_key}",
        response=httpx.Response(401, request=httpx.Request("POST", foundation.client.base_url)),
        body=None,
    )
    embedding.client.embeddings.create.side_effect = RuntimeError(f"Invalid credential {embedding.client.api_key}")

    with pytest.raises(SearchSpaceValueError) as caught:
        prepare_search_space_with_models(
            {}, benchmark_data=pd.DataFrame({"question": ["Where is Warsaw?"]}), **direct_models
        )

    error_message = str(caught.value)
    assert error_message.splitlines() == [
        f"Provided models of type 'foundation' do not respond: {[foundation.model_id]!r}.",
        f"Provided models of type 'embedding' do not respond: {[embedding.model_id]!r}.",
    ]
    foundation.client.chat.completions.create.assert_called_once()
    embedding.client.embeddings.create.assert_called_once()
    assert "status_code=401" in caplog.text
    for model in (foundation, embedding):
        assert model.client.api_key not in error_message + caplog.text


@pytest.mark.parametrize(
    "payload,field",
    [
        ({"foundation_models": [{"model_id": "other-chat"}]}, "foundation_models"),
        ({"embedding_models": None}, "embedding_models"),
        ({"chunking_methods": ["semantic"]}, "chunking_methods"),
        ({"chunk_sizes": [1]}, "chunk_sizes"),
    ],
)
def test_direct_search_space_validates_constraints(payload, field, direct_models):
    with pytest.raises(ValidationError, match=field):
        prepare_search_space_with_models(payload, **direct_models)
    assert all(not model.client.mock_calls for (model,) in direct_models.values())


@pytest.mark.parametrize("with_maas", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        {"foundation_models": [{"model_id": "model", "api_key": "private-key"}]},
        {"template_ids": ["private-key"]},
        {"chunking_methods": ["private-key"]},
        {"chunk_sizes": ["private-key"]},
    ],
)
def test_search_space_does_not_echo_rejected_constraint_values(with_maas, payload, direct_models, caplog):
    with caplog.at_level("INFO", logger="ai4rag"):
        with pytest.raises(ValidationError) as caught:
            if with_maas:
                prepare_search_space_with_maas(payload, MagicMock(spec=OpenAI))
            else:
                prepare_search_space_with_models(payload, **direct_models)

    assert "private-key" not in caplog.text
    assert "private-key" not in str(caught.value)


@pytest.mark.parametrize("role", ["foundation_models", "embedding_models"])
def test_direct_search_space_rejects_external_http_before_any_probe(role, direct_models, caplog):
    (model,) = direct_models[role]
    model.client.base_url = httpx.URL("http://models.example.com/v1?token=private-key")

    with pytest.raises(SearchSpaceValueError, match=rf"{role}.*HTTPS") as caught:
        prepare_search_space_with_models({}, **direct_models)

    assert "private-key" not in str(caught.value) + caplog.text
    assert all(not model.client.mock_calls for (model,) in direct_models.values())


@pytest.mark.parametrize("role", ["foundation_models", "embedding_models"])
@pytest.mark.parametrize("models", [[], ["model-id"], object()])
def test_direct_search_space_requires_nonempty_sequences_of_model_wrappers(role, models, direct_models):
    direct_models[role] = models
    with pytest.raises(SearchSpaceValueError, match=rf"{role}.*non-empty sequence"):
        prepare_search_space_with_models({}, **direct_models)


@pytest.mark.parametrize("role", ["foundation_models", "embedding_models"])
def test_direct_search_space_rejects_duplicate_ids_with_separate_clients(role, direct_models):
    duplicate = copy(direct_models[role][0])
    duplicate.client = MagicMock(spec=OpenAI)
    direct_models[role].append(duplicate)

    with pytest.raises(SearchSpaceValueError, match=rf"{role}.*unique model IDs"):
        prepare_search_space_with_models({}, **direct_models)


def test_direct_search_space_applies_neo4j_defaults(direct_models):
    result = prepare_search_space_with_models({}, vector_store_type="neo4j", **direct_models)

    assert result["template_id"].values == ("agentic_graph_rag",)
    assert result["search_mode"].values == ("graph",)
    assert "ranker_strategy" not in {param.name for param in result.params}


def test_direct_incompatible_template_leaves_model_unchanged(direct_models):
    (foundation,) = direct_models["foundation_models"]
    original_language = foundation.language
    original_prompt = "Use these sources: {reference_documents}\nQuestion: {question}"
    foundation.user_message_text = original_prompt
    completion = Mock()
    completion.message.content = '{"code": "pl"}'
    foundation.client.chat.completions.create.return_value = Mock(choices=[completion])

    with pytest.raises(SearchSpaceValueError, match="incompatible"):
        prepare_search_space_with_models(
            {"template_ids": ["sequential_rag"]},
            benchmark_data=pd.DataFrame({"question": ["Gdzie jest Warszawa?"]}),
            vector_store_type="neo4j",
            **direct_models,
        )

    assert all(not model.client.mock_calls for (model,) in direct_models.values())
    assert foundation.language is original_language
    assert foundation.user_message_text == original_prompt
