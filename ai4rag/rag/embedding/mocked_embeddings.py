from .base_model import BaseEmbeddingModel


class MockedEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dim = 512

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        n = []
        for _ in texts:
            n.append([0.1 for _ in range(self._dim)])

        return n

    def embed_query(self, query: str) -> list[float]:
        return [0.5 for _ in range(self._dim)]
