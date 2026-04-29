# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for prepare_search_space module."""

from unittest.mock import MagicMock, Mock

import pytest
from llama_stack_client import LlamaStackClient
from pydantic import ValidationError

from ai4rag.search_space.prepare import prepare_search_space_with_llama_stack
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


class TestPrepareSearchSpaceWithLlamaStack:
    """Test prepare_search_space_with_llama_stack function."""

    def test_basic_payload_with_defaults(self, mocker):
        """Test preparation with empty payload using defaults."""
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        # Empty payload should use defaults
        payload = {}

        result = prepare_search_space_with_llama_stack(payload, mock_client)

        # Should create search space with default models
        # Note: AI4RAGSearchSpace merges user params with defaults, so we get more than 2 params
        assert result is not None
        assert len(result.params) > 0
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

    def test_payload_with_custom_foundation_models(self, mocker):
        """Test preparation with custom foundation models."""
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "custom-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "custom-llm"}]}

        result = prepare_search_space_with_llama_stack(payload, mock_client)

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
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list response
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "custom-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 1024}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"embedding_models": [{"model_id": "custom-embedding"}]}

        result = prepare_search_space_with_llama_stack(payload, mock_client)

        assert result is not None
        # Verify parameter names are correct
        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

        # Verify the custom embedding model is included
        embedding_param = result["embedding_model"]
        assert len(embedding_param.values) == 1
        assert embedding_param.values[0].model_id == "custom-embedding"

    def test_invalid_payload_raises_validation_error(self, mocker):
        """Test that invalid payload raises validation error."""
        mock_client = MagicMock(spec=LlamaStackClient)

        # Mock model list
        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        # Mock validation functions to always return True
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        # Invalid payload with unrecognized parameter
        payload = {"invalid_parameter": "value"}

        with pytest.raises(ValidationError, match="Unknown validation error|invalid_parameter"):
            prepare_search_space_with_llama_stack(payload, mock_client)

    def test_non_llama_stack_client_raises_error(self):
        """Test that non-LlamaStackClient raises error."""
        mock_client = MagicMock(spec=object)  # Not a LlamaStackClient

        payload = {}

        with pytest.raises(SearchSpaceValueError, match="Unrecognized client type"):
            prepare_search_space_with_llama_stack(payload, mock_client)

    def test_chroma_vector_store_excludes_hybrid_params(self, mocker):
        """Test that chroma vector store type excludes hybrid search parameters."""
        mock_client = MagicMock(spec=LlamaStackClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        result = prepare_search_space_with_llama_stack({}, mock_client, vector_store_type="chroma")

        param_names = [p.name for p in result.params]
        assert "search_mode" in param_names
        assert "ranker_strategy" not in param_names
        assert "ranker_k" not in param_names
        assert "ranker_alpha" not in param_names

        search_mode_param = result["search_mode"]
        assert search_mode_param.values == ("vector",)

    def test_ls_milvus_vector_store_includes_hybrid_params_by_default(self, mocker):
        """Test that ls_milvus vector store type includes hybrid search parameters by default."""
        mock_client = MagicMock(spec=LlamaStackClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        result = prepare_search_space_with_llama_stack({}, mock_client, vector_store_type="ls_milvus")

        param_names = [p.name for p in result.params]
        assert "search_mode" in param_names
        assert "ranker_strategy" in param_names
        assert "ranker_k" in param_names
        assert "ranker_alpha" in param_names

        search_mode_param = result["search_mode"]
        assert "vector" in search_mode_param.values
        assert "hybrid" in search_mode_param.values

    def test_default_vector_store_type_is_ls_milvus(self, mocker):
        """Test that default vector_store_type is ls_milvus (includes hybrid params by default)."""
        mock_client = MagicMock(spec=LlamaStackClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        result = prepare_search_space_with_llama_stack({}, mock_client)

        param_names = [p.name for p in result.params]
        assert "search_mode" in param_names
        assert "ranker_strategy" in param_names
        assert "ranker_k" in param_names
        assert "ranker_alpha" in param_names

    def test_user_specifies_not_responding_foundation_model(self, mocker):
        """Error when user requests a foundation model that is registered but not responding."""
        mock_client = MagicMock(spec=LlamaStackClient)

        mock_llm_ok = Mock()
        mock_llm_ok.id = "llm-ok"
        mock_llm_ok.custom_metadata = {"model_type": "llm"}

        mock_llm_bad = Mock()
        mock_llm_bad.id = "llm-bad"
        mock_llm_bad.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm_ok, mock_llm_bad, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            side_effect=lambda m: m.model_id != "llm-bad",
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "llm-bad"}]}

        with pytest.raises(SearchSpaceValueError, match="llm-bad.*registered but do not respond"):
            prepare_search_space_with_llama_stack(payload, mock_client)

    def test_user_specifies_not_responding_embedding_model(self, mocker):
        """Error when user requests an embedding model that is registered but not responding."""
        mock_client = MagicMock(spec=LlamaStackClient)

        mock_llm = Mock()
        mock_llm.id = "default-llm"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_emb_ok = Mock()
        mock_emb_ok.id = "emb-ok"
        mock_emb_ok.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 512}
        mock_emb_ok.metadata = {}

        mock_emb_bad = Mock()
        mock_emb_bad.id = "emb-bad"
        mock_emb_bad.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768, "context_length": 512}
        mock_emb_bad.metadata = {}

        mock_client.models.list.return_value = [mock_llm, mock_emb_ok, mock_emb_bad]

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            side_effect=lambda m: m.model_id != "emb-bad",
        )

        payload = {"embedding_models": [{"model_id": "emb-bad"}]}

        with pytest.raises(SearchSpaceValueError, match="emb-bad.*registered but do not respond"):
            prepare_search_space_with_llama_stack(payload, mock_client)

    def test_user_specifies_unregistered_foundation_model(self, mocker):
        """Error when user requests a foundation model not registered in llama-stack."""
        mock_client = MagicMock(spec=LlamaStackClient)

        mock_llm = Mock()
        mock_llm.id = "llm-ok"
        mock_llm.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            return_value=True,
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "llm-unknown"}]}

        with pytest.raises(SearchSpaceValueError, match="llm-unknown.*not registered within llama-stack"):
            prepare_search_space_with_llama_stack(payload, mock_client)

    def test_user_picks_available_model_while_others_fail(self, mocker):
        """User selects a responding model — succeeds even when other registered models fail."""
        mock_client = MagicMock(spec=LlamaStackClient)

        mock_llm_ok = Mock()
        mock_llm_ok.id = "llm-ok"
        mock_llm_ok.custom_metadata = {"model_type": "llm"}

        mock_llm_bad = Mock()
        mock_llm_bad.id = "llm-bad"
        mock_llm_bad.custom_metadata = {"model_type": "llm"}

        mock_embedding = Mock()
        mock_embedding.id = "default-embedding"
        mock_embedding.custom_metadata = {"model_type": "embedding", "embedding_dimension": 768}

        mock_client.models.list.return_value = [mock_llm_ok, mock_llm_bad, mock_embedding]

        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_foundation_model",
            side_effect=lambda m: m.model_id != "llm-bad",
        )
        mocker.patch(
            "ai4rag.search_space.prepare.llama_stack_utils._validate_embedding_model",
            return_value=True,
        )

        payload = {"foundation_models": [{"model_id": "llm-ok"}]}
        result = prepare_search_space_with_llama_stack(payload, mock_client)

        fm_ids = [m.model_id for m in result["foundation_model"].values]
        assert fm_ids == ["llm-ok"]
