from random import random, seed
from typing import Any

from ai4rag.rag.embedding.base_model import EmbeddingModel
from ai4rag.rag.foundation_models.base_model import FoundationModel


seed(42)


class MockedFoundationModel(FoundationModel):
    def __init__(
        self,
        model_id: str,
        model_params: dict[str, Any] | None = None,
        client: None = None,
    ):
        super().__init__(client, model_id, model_params)

    def chat(self, system_message: str, user_message: str) -> str:
        return "I cannot answer this question, because I am just a mocked model."


class MockedEmbeddingModel(EmbeddingModel):
    def __init__(self, model_id: str, params: dict[str, Any], client: None = None):
        super().__init__(client, model_id, params)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        n = []
        for _ in texts:
            n.append([random() for _ in range(self.params["embedding_dimension"])])

        return n

    def embed_query(self, query: str) -> list[float]:
        return [random() for _ in range(self.params["embedding_dimension"])]
