# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import Generic, TypeVar

ClientT = TypeVar("ClientT")
EmbeddingParamsT = TypeVar("EmbeddingParamsT")


class BaseEmbeddingModel(ABC, Generic[ClientT, EmbeddingParamsT]):
    """Define the provider-independent embedding interface.

    Parameters
    ----------
    client : ClientT
        Client used by the concrete provider implementation.
    model_id : str
        Model identifier used for display, equality, ordering, and hashing.
    params : EmbeddingParamsT | None, default=None
        Provider-specific parameters. Assignment invokes any parameter setter
        defined by the concrete model.
    """

    def __init__(self, client: ClientT, model_id: str, params: EmbeddingParamsT | None = None) -> None:
        """Store the client, model identifier, and provider-specific parameters."""
        self.client: ClientT = client
        self.model_id = model_id
        self.params: EmbeddingParamsT | None = params

    def __str__(self) -> str:
        """Return the model identifier as its display string."""
        return self.model_id

    def __repr__(self) -> str:
        """Return the same model identifier used by str(self)."""
        return str(self)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        """Compare embedding models by identifier alone.

        Parameters
        ----------
        other : object
            Object to compare with this model.

        Returns
        -------
        bool | NotImplementedType
            Identifier equality for another embedding model, or NotImplemented
            to defer comparison of an unrelated object to Python.
        """
        if not isinstance(other, BaseEmbeddingModel):
            return NotImplemented

        return self.model_id == other.model_id

    def __hash__(self) -> int:
        """Return the identifier's hash, consistent with identifier-only equality."""
        return hash(self.model_id)

    def __lt__(self, other: object) -> bool | NotImplementedType:
        """Order embedding models lexicographically by identifier.

        Parameters
        ----------
        other : object
            Object to compare with this model.

        Returns
        -------
        bool | NotImplementedType
            Identifier ordering for another embedding model, or NotImplemented
            to defer comparison of an unrelated object to Python.
        """
        if not isinstance(other, BaseEmbeddingModel):
            return NotImplemented
        return self.model_id < other.model_id

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in input order.

        Parameters
        ----------
        texts : list[str]
            Texts to embed. Batching and recovery are provider-specific.

        Returns
        -------
        list[list[float]]
            One vector per input text, preserving input order.
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed one query string.

        Parameters
        ----------
        query : str
            Query text to embed.

        Returns
        -------
        list[float]
            The query's embedding vector.
        """
