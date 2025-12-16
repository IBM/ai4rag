from typing import Any


class LlamaStackEmbeddingModel:
    def __init__(self, model_id: str, params: dict[str, Any]):
        self.model_id = model_id
        self.params = params

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents.

        Parameters
        ----------
        texts : list[str]
            List of text-like chunks.

        Returns
        -------
        list[list[float]]
            Embeddings made from the list of texts.
        """
        raise NotImplementedError()

    def embed_query(self, query: str) -> list[float]:
        """Embed query text

        Parameters
        ----------
        query : str
            User's query as text.

        Returns
        -------
        list[float]
            Single embeddings vector made from the query.
        """
        raise NotImplementedError
