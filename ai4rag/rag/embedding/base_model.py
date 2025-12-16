from abc import ABC, abstractmethod
from typing import Any


class BaseEmbeddingModel(ABC):
    def __init__(self, model_id: str, params: dict[str, Any]):
        self.model_id = model_id
        self.params = params

    @abstractmethod
    def embed_documents(self):
        raise NotImplementedError()

    @abstractmethod
    def embed_query(self):
        raise NotImplementedError()
