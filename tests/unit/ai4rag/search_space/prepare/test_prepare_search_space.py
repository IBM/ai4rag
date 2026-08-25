# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for prepare_search_space module."""

from unittest.mock import MagicMock, Mock

import pytest
from openai import OpenAI
from pydantic import ValidationError

from ai4rag.search_space.prepare import prepare_search_space_with_maas
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
    client.models.list.return_value.data = [_make_model(mid) for mid in registered_ids]
    emb_response = Mock()
    emb_response.data = [Mock(embedding=[0.0] * dim)]
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

    def test_chroma_vector_store_excludes_hybrid_params(self, mocker):
        """The chroma vector store type excludes hybrid search parameters."""
        client = _setup_client(mocker, ["default-llm", "default-embedding"])

        result = prepare_search_space_with_maas(_payload(), client, vector_store_type="chroma")

        param_names = [p.name for p in result.params]
        assert "search_mode" in param_names
        assert "ranker_strategy" not in param_names
        assert "ranker_k" not in param_names
        assert "ranker_alpha" not in param_names

        search_mode_param = result["search_mode"]
        assert search_mode_param.values == ("vector",)

    def test_non_chroma_vector_store_includes_hybrid_params_by_default(self, mocker):
        """Non-chroma vector stores (default milvus) include hybrid search parameters."""
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

    def test_user_specifies_not_responding_foundation_model(self, mocker):
        """Error when a requested foundation model is available but does not respond."""
        client = _setup_client(
            mocker,
            ["llm-ok", "llm-bad", "default-embedding"],
            foundation_valid=lambda m: m.model_id != "llm-bad",
        )

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*llm-bad"):
            prepare_search_space_with_maas(_payload(foundation_ids=["llm-bad"]), client)

    def test_user_specifies_not_responding_embedding_model(self, mocker):
        """Error when a requested embedding model is available but does not respond."""
        client = _setup_client(
            mocker,
            ["default-llm", "emb-ok", "emb-bad"],
            embedding_valid=lambda m: m.model_id != "emb-bad",
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
            foundation_valid=lambda m: m.model_id != "llm-bad",
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
