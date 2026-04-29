# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional tests for prepare_search_space_with_llama_stack against a live Llama Stack server."""

import os

import pytest
from dotenv import find_dotenv, load_dotenv
from llama_stack_client import LlamaStackClient

from ai4rag.search_space.prepare import prepare_search_space_with_llama_stack
from ai4rag.search_space.src.exceptions import SearchSpaceValueError

load_dotenv(find_dotenv())


pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMA_STACK_CLIENT_BASE_URL") is None,
    reason="LLAMA_STACK_CLIENT_BASE_URL environment variable not set",
)


@pytest.fixture(scope="module")
def client():
    return LlamaStackClient(
        base_url=os.environ["LLAMA_STACK_CLIENT_BASE_URL"],
        api_key=os.environ.get("LLAMA_STACK_CLIENT_API_KEY", ""),
    )


@pytest.fixture(scope="module")
def discovered_search_space(client):
    """Discover models once and reuse across tests that need valid model IDs."""
    return prepare_search_space_with_llama_stack({}, client)


class TestPrepareSearchSpaceWithLlamaStack:
    """Prepare search space against a live Llama Stack server."""

    def test_empty_payload_builds_search_space_with_discovered_models(self, discovered_search_space):
        """Empty payload discovers and validates all server models, builds a complete search space."""
        param_names = [p.name for p in discovered_search_space.params]
        assert "foundation_model" in param_names
        assert "embedding_model" in param_names

        assert len(discovered_search_space["foundation_model"].values) >= 1
        assert len(discovered_search_space["embedding_model"].values) >= 1

        assert "chunk_size" in param_names
        assert "chunking_method" in param_names
        assert "retrieval_method" in param_names

    def test_user_payload_with_specific_models(self, client, discovered_search_space):
        """User-specified models that exist on the server appear in the resulting search space."""
        fm_id = discovered_search_space["foundation_model"].values[0].model_id
        em_id = discovered_search_space["embedding_model"].values[0].model_id

        payload = {
            "foundation_models": [{"model_id": fm_id}],
            "embedding_models": [{"model_id": em_id}],
        }

        result = prepare_search_space_with_llama_stack(payload, client)

        fm_ids = [m.model_id for m in result["foundation_model"].values]
        em_ids = [m.model_id for m in result["embedding_model"].values]
        assert fm_ids == [fm_id]
        assert em_ids == [em_id]

    def test_unregistered_model_raises_error(self, client):
        """Requesting a model that does not exist on the server raises a clear error."""
        payload = {"foundation_models": [{"model_id": "nonexistent-model-xyz"}]}

        with pytest.raises(SearchSpaceValueError, match="nonexistent-model-xyz"):
            prepare_search_space_with_llama_stack(payload, client)
