# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional tests for prepare_search_space_with_maas against a live MaaS deployment."""

import os

import pytest
from dotenv import find_dotenv, load_dotenv

from ai4rag.search_space.prepare import prepare_search_space_with_maas
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from dev_utils.utils import create_dev_maas_client

load_dotenv(find_dotenv(".env.local"))


pytestmark = pytest.mark.skipif(
    os.environ.get("MAAS_BASE") is None,
    reason="MAAS_BASE environment variable not set",
)

#: MaaS carries no metadata to distinguish model types, so the caller must
#: declare foundation and embedding model ids explicitly.
FOUNDATION_MODEL_ID = os.environ.get("AI4RAG_TEST_FOUNDATION_MODEL", "qwen3-8b-fp8-dynamic")
EMBEDDING_MODEL_ID = os.environ.get("AI4RAG_TEST_EMBEDDING_MODEL", "bge-m3")


@pytest.fixture(scope="module")
def client():
    return create_dev_maas_client()


@pytest.fixture(scope="module")
def valid_payload():
    """A payload declaring one foundation and one embedding model by type."""
    return {
        "foundation_models": [{"model_id": FOUNDATION_MODEL_ID}],
        "embedding_models": [{"model_id": EMBEDDING_MODEL_ID}],
    }


class TestPrepareSearchSpaceWithMaas:
    """Prepare search space against a live MaaS deployment."""

    def test_payload_builds_complete_search_space(self, client, valid_payload):
        """A payload declaring both model types validates them and builds a complete search space."""
        result = prepare_search_space_with_maas(valid_payload, client)

        param_names = [p.name for p in result.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

        assert len(result["foundation_model"].values) >= 1
        assert len(result["embedding_model"].values) >= 1

        assert "chunk_size" in param_names
        assert "chunking_method" in param_names
        assert "retrieval_method" in param_names

    def test_requested_models_appear_in_search_space(self, client, valid_payload):
        """The specific models requested per type appear in the resulting search space."""
        result = prepare_search_space_with_maas(valid_payload, client)

        fm_ids = [m.model_id for m in result["foundation_model"].values]
        em_ids = [m.model_id for m in result["embedding_model"].values]
        assert fm_ids == [FOUNDATION_MODEL_ID]
        assert em_ids == [EMBEDDING_MODEL_ID]

    def test_missing_model_type_raises_error(self, client):
        """Omitting a model-type list is rejected: MaaS cannot infer model type."""
        payload = {"foundation_models": [{"model_id": FOUNDATION_MODEL_ID}]}

        with pytest.raises(SearchSpaceValueError, match="Provide both 'foundation_models' and 'embedding_models'"):
            prepare_search_space_with_maas(payload, client)

    def test_unavailable_model_raises_error(self, client):
        """Requesting a model that is not available in MaaS raises a clear error."""
        payload = {
            "foundation_models": [{"model_id": "nonexistent-model-xyz"}],
            "embedding_models": [{"model_id": EMBEDDING_MODEL_ID}],
        }

        with pytest.raises(SearchSpaceValueError, match="nonexistent-model-xyz"):
            prepare_search_space_with_maas(payload, client)
