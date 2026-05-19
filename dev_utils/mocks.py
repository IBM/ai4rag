# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from random import random, seed
from typing import Any

from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

seed(42)


class MockedFoundationModel(BaseFoundationModel[None, dict[str, Any] | None]):
    def __init__(
        self,
        model_id: str,
        params: dict[str, Any] | None = None,
        client: None = None,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        context_template_text: str | None = None,
    ):
        super().__init__(
            client=client,
            model_id=model_id,
            params=params,
            system_message_text=system_message_text,
            user_message_text=user_message_text,
            context_template_text=context_template_text,
        )

    def create_response(self, user_message: str, vector_store_id: str | None = None) -> str:
        """
        Utilise Responses API (agent loop) to interact with the model.

        Parameters
        ----------
        user_message : str
            User message for the model to answer.

        vector_store_id : str | None
            If provided then references the vector store to search against.

        Returns
        -------
        str
            Response text from the model.
        """
        return "I cannot answer this question, because I am just a mocked model."


class MockedEmbeddingModel(BaseEmbeddingModel[None, dict[str, Any]]):
    def __init__(self, model_id: str, params: dict[str, Any], client: None = None):
        super().__init__(client=client, model_id=model_id, params=params)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        n = []
        for _ in texts:
            n.append([random() for _ in range(self.params["embedding_dimension"])])

        return n

    def embed_query(self, query: str) -> list[float]:
        return [random() for _ in range(self.params["embedding_dimension"])]


class MockedOGXClient:
    """Mock OGX client for testing without real OGX server."""

    class MockedProviders:
        """Mock providers interface."""

        def retrieve(self, provider_id: str):
            """Return mock provider with provider_type attribute."""
            class MockProvider:
                provider_type = "mock_provider"
            return MockProvider()

    def __init__(self):
        self.providers = self.MockedProviders()
