from typing import Any


class LlamaStackEmbeddingModel:
    def __init__(self, model_id: str, params: dict[str, Any]):
        self.model_id = model_id
        self.params = params

    def embed_documents(self):
        raise NotImplementedError()

    def embed_query(self):
        raise NotImplementedError()
