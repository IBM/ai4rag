# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for prepare_search_space module."""

from unittest.mock import MagicMock, Mock

import pytest
from ogx_client import OgxClient
from pydantic import ValidationError

from ai4rag.search_space.prepare import prepare_search_space_with_ogx
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


class TestPrepareSearchSpaceWithOgx:
    """Test prepare_search_space_with_ogx function."""

    def test_basic_payload_with_defaults(self, mocker):
        """Test preparation with empty payload using defaults."""
        mock_client = MagicMock(spec=OgxClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        # Empty payload should use defaults
        payload = {}

        result = prepare_search_space_with_ogx(payload, mock_client)

        # Should create search space with default models
        # Note: AI4RAGSearchSpace merges user params with defaults, so we get more than 2 params
        assert result is not None
        assert len(result.params) > 0
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

    def test_payload_with_custom_foundation_models(self, mocker):
        """Test preparation with custom foundation models."""
        mock_client = MagicMock(spec=OgxClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "custom-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "custom-llm"}]}

        result = prepare_search_space_with_ogx(payload, mock_client)

        assert result is not None
        # Verify parameter names are correct
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

        # Verify the custom foundation model is included
        foundation_param = result["foundation_model"]
        assert len(foundation_param.values) == 1
        assert foundation_param.values[0].model_id == "custom-llm"

    def test_payload_with_custom_embedding_models(self, mocker):
        """Test preparation with custom embedding models."""
        mock_client = MagicMock(spec=OgxClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "custom-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 1024}

        mock_client.models.list.return_value.data = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"embedding_models": [{"model_id": "custom-embedding"}]}

        result = prepare_search_space_with_ogx(payload, mock_client)

        assert result is not None
        # Verify parameter names are correct
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

        # Verify the custom embedding model is included
        embedding_param = result["embedding_model"]
        assert len(embedding_param.values) == 1
        assert embedding_param.values[0].model_id == "custom-embedding"

    def test_invalid_payload_raises_validation_error(self):
        """Test that invalid payload raises validation error."""
        with pytest.raises(ValidationError, match="Unknown validation error|invalid_parameter"):
            prepare_search_space_with_ogx({"invalid_parameter": "value"}, MagicMock())

    def test_non_ogx_client_raises_error(self):
        """Test that non-OgxClient raises error."""
        mock_client = MagicMock(spec=object)  # Not an OgxClient

        payload = {}

        with pytest.raises(SearchSpaceValueError, match="Unrecognized client type"):
            prepare_search_space_with_ogx(payload, mock_client)

    def test_chroma_vector_store_excludes_hybrid_params(self, mocker):
        """Test that chroma vector store type excludes hybrid search parameters."""
        mock_client = MagicMock(spec=OgxClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        result = prepare_search_space_with_ogx({}, mock_client, vector_store_type="chroma")

        param_names = [p.name for p in result.params]
        assert "search_mode" in param_names
        assert "ranker_strategy" not in param_names
        assert "ranker_k" not in param_names
        assert "ranker_alpha" not in param_names

        search_mode_param = result["search_mode"]
        assert search_mode_param.values == ("vector",)

    def test_ogx_vector_store_includes_hybrid_params_by_default(self, mocker):
        """Test that ogx vector store type includes hybrid search parameters by default."""
        mock_client = MagicMock(spec=OgxClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        result = prepare_search_space_with_ogx({}, mock_client, vector_store_type="ogx")

        param_names = [p.name for p in result.params]
        assert "search_mode" in param_names
        assert "ranker_strategy" in param_names
        assert "ranker_k" in param_names
        assert "ranker_alpha" in param_names

        search_mode_param = result["search_mode"]
        assert "vector" in search_mode_param.values
        assert "hybrid" in search_mode_param.values

    def test_user_specifies_not_responding_foundation_model(self, mocker):
        """Error when user requests a foundation model that is registered but not responding."""
        mock_client = MagicMock(spec=OgxClient)

        mock_llm_ok = Mock()
        mock_llm_ok.id = "llm-ok"
        mock_llm_ok.custom_metadata = {"model_type": "llm"}

        mock_llm_bad = Mock()
        mock_llm_bad.id = "llm-bad"
        mock_llm_bad.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm_ok, mock_llm_bad, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            side_effect=lambda m: m.model_id != "llm-bad",
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "llm-bad"}]}

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*llm-bad"):
            prepare_search_space_with_ogx(payload, mock_client)

    def test_user_specifies_not_responding_embedding_model(self, mocker):
        """Error when user requests an embedding model that is registered but not responding."""
        mock_client = MagicMock(spec=OgxClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_emb_ok = Mock()
        mock_emb_ok.id = "emb-ok"
        mock_emb_ok.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 1024}
        mock_emb_ok.metadata = {}

        mock_emb_bad = Mock()
        mock_emb_bad.id = "emb-bad"
        mock_emb_bad.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 1024}
        mock_emb_bad.metadata = {}

        mock_client.models.list.return_value.data = [mock_llm, mock_emb_ok, mock_emb_bad]

        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            side_effect=lambda m: m.model_id != "emb-bad",
        )

        payload = {"embedding_models": [{"model_id": "emb-bad"}]}

        with pytest.raises(SearchSpaceValueError, match=r"do not respond.*emb-bad"):
            prepare_search_space_with_ogx(payload, mock_client)

    def test_user_specifies_unregistered_foundation_model(self, mocker):
        """Error when user requests a foundation model not registered in OGX."""
        mock_client = MagicMock(spec=OgxClient)

        mock_llm = Mock()
        mock_llm.id = "llm-ok"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "llm-unknown"}]}

        with pytest.raises(SearchSpaceValueError, match=r"not registered in OGX.*llm-unknown"):
            prepare_search_space_with_ogx(payload, mock_client)

    def test_user_picks_available_model_while_others_fail(self, mocker):
        """User selects a responding model — succeeds even when other registered models fail."""
        mock_client = MagicMock(spec=OgxClient)

        mock_llm_ok = Mock()
        mock_llm_ok.id = "llm-ok"
        mock_llm_ok.custom_metadata = {"model_type": "llm"}

        mock_llm_bad = Mock()
        mock_llm_bad.id = "llm-bad"
        mock_llm_bad.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm_ok, mock_llm_bad, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_foundation_model",
            side_effect=lambda m: m.model_id != "llm-bad",
        )
        mocker.patch(
            "ai4rag.search_space.prepare.ogx_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "llm-ok"}]}
        result = prepare_search_space_with_ogx(payload, mock_client)

        fm_ids = [m.model_id for m in result["foundation_model"].values]
        assert fm_ids == ["llm-ok"]

    def _setup_mock_client(self, mocker):
        mock_client = MagicMock(spec=OgxClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value.data = [mock_llm, mock_embedding]

        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_foundation_model", return_value=True)
        mocker.patch("ai4rag.search_space.prepare.ogx_utils._validate_embedding_model", return_value=True)

        return mock_client

    def test_no_chunking_params_returns_default_chunking_dimensions(self, mocker):
        """Without chunking overrides the result includes default chunking_method and chunk_size dimensions."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx({}, mock_client)

        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names
        assert "chunking_method" in param_names
        assert "chunk_size" in param_names

    def test_custom_chunking_methods_override_defaults(self, mocker):
        """chunking_methods in payload overrides the default chunking_method dimension."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx({"chunking_methods": ["recursive"]}, mock_client)

        assert result["chunking_method"].values == ("recursive",)

    def test_custom_chunk_sizes_override_defaults(self, mocker):
        """chunk_sizes in payload overrides the default chunk_size dimension."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx({"chunk_sizes": [256, 512]}, mock_client)

        assert set(result["chunk_size"].values) == {256, 512}

    def test_both_chunking_params_applied_together(self, mocker):
        """Both chunking_methods and chunk_sizes in payload can be set simultaneously."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx(
            {"chunking_methods": ["hybrid"], "chunk_sizes": [1024]},
            mock_client,
        )

        assert result["chunking_method"].values == ("hybrid",)
        assert result["chunk_size"].values == (1024,)

    def test_all_chunking_params_produce_non_empty_search_space(self, mocker):
        """chunking_methods, chunk_sizes, and chunk_overlaps together produce a non-empty search space."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx(
            {"chunking_methods": ["recursive"], "chunk_sizes": [128, 256], "chunk_overlaps": [32, 64]},
            mock_client,
        )

        assert result["chunking_method"].values == ("recursive",)
        assert set(result["chunk_size"].values) == {128, 256}
        assert set(result["chunk_overlap"].values) == {32, 64}
        assert len(result.params) > 0

    def test_duplicate_chunking_methods_are_deduplicated(self, mocker):
        """Duplicate chunking_methods are silently deduplicated, preserving order."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx({"chunking_methods": ["recursive", "recursive"]}, mock_client)

        assert result["chunking_method"].values == ("recursive",)

    def test_duplicate_chunk_sizes_are_deduplicated(self, mocker):
        """Duplicate chunk_sizes are silently deduplicated, preserving order."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx({"chunk_sizes": [128, 129, 128]}, mock_client)

        assert result["chunk_size"].values == (128, 129)

    def test_unsupported_chunking_method_raises_error(self):
        """An unsupported chunking method raises ValidationError before any I/O."""
        with pytest.raises(ValidationError, match="Unsupported chunking methods"):
            prepare_search_space_with_ogx({"chunking_methods": ["semantic"]}, MagicMock())

    def test_chunk_size_below_min_raises_error(self):
        """A chunk size below MIN_CHUNK_SIZE raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_ogx({"chunk_sizes": [1]}, MagicMock())

    def test_chunk_size_above_max_raises_error(self):
        """A chunk size above MAX_CHUNK_SIZE raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_ogx({"chunk_sizes": [99999]}, MagicMock())

    def test_bool_chunk_size_raises_error(self):
        """A bool value in chunk_sizes raises ValidationError before any I/O."""
        with pytest.raises(ValidationError, match="must be a positive integer"):
            prepare_search_space_with_ogx({"chunk_sizes": [True]}, MagicMock())

    def test_custom_chunk_overlaps_override_defaults(self, mocker):
        """chunk_overlaps in payload overrides the default chunk_overlap dimension."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx({"chunk_overlaps": [0, 128]}, mock_client)

        assert set(result["chunk_overlap"].values) == {0, 128}

    def test_duplicate_chunk_overlaps_are_deduplicated(self, mocker):
        """Duplicate chunk_overlaps are silently deduplicated, preserving order."""
        mock_client = self._setup_mock_client(mocker)

        result = prepare_search_space_with_ogx({"chunk_overlaps": [0, 64, 0]}, mock_client)

        assert result["chunk_overlap"].values == (0, 64)

    def test_chunk_overlap_below_min_raises_error(self):
        """A chunk overlap below MIN_CHUNK_OVERLAP raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_ogx({"chunk_overlaps": [-1]}, MagicMock())

    def test_chunk_overlap_above_max_raises_error(self):
        """A chunk overlap above MAX_CHUNK_OVERLAP raises ValidationError before any I/O."""
        with pytest.raises(ValidationError):
            prepare_search_space_with_ogx({"chunk_overlaps": [99999]}, MagicMock())

    def test_bool_chunk_overlap_raises_error(self):
        """A bool value in chunk_overlaps raises ValidationError before any I/O."""
        with pytest.raises(ValidationError, match="must be a non-negative integer"):
            prepare_search_space_with_ogx({"chunk_overlaps": [True]}, MagicMock())
