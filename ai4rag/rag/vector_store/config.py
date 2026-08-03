# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import os
from abc import ABC
from dataclasses import dataclass

__all__ = ["BaseVectorStoreConfig", "MilvusConfig", "PGVectorConfig", "ChromaConfig"]


@dataclass(frozen=True, kw_only=True)
class BaseVectorStoreConfig(ABC):
    """Base config shared by every vector store backend.

    Attributes
    ----------
    provider : str
        Backend discriminator (e.g. ``"chroma"``, ``"milvus"``, ``"pgvector"``)
        used by :func:`ai4rag.rag.vector_store.get_vector_store.get_vector_store`
        to select the concrete store class.
    """

    provider: str


@dataclass(frozen=True, kw_only=True)
class ChromaConfig(BaseVectorStoreConfig):
    """Connection parameters for a Chroma instance.

    The running mode is inferred from which fields are set, so the same config
    class drives all three Chroma deployment styles:

    * **Ephemeral (default)** — fully in-memory, nothing persisted, when both
      ``persist_directory`` and ``host`` are ``None``.
    * **Persistent** — local on-disk storage when ``persist_directory`` is set.
    * **Client/server** — connect to a remote Chroma server when ``host`` is
      set (``host`` takes precedence over ``persist_directory``).

    Parameters
    ----------
    persist_directory : str | None, default=None
        Filesystem path backing a local persistent client. ``None`` selects an
        ephemeral in-memory client.
    host : str | None, default=None
        Hostname of a remote Chroma server. ``None`` keeps operation local
        (ephemeral or persistent).
    port : int, default=8000
        Port of the remote Chroma server. Used only when ``host`` is set.
    provider : str, default="chroma"
        Name of the provider used in the system.
    """

    persist_directory: str | None = None
    host: str | None = None
    port: int = 8000
    provider: str = "chroma"

    @classmethod
    def from_env(cls) -> "ChromaConfig":
        """Build config from ``CHROMA_*`` environment variables.

        Reads ``CHROMA_PERSIST_DIR``, ``CHROMA_HOST`` and ``CHROMA_PORT``.
        Unset variables fall back to the ephemeral in-memory defaults.

        Returns
        -------
        ChromaConfig
            Config populated from the ``CHROMA_*`` environment variables.
        """
        return cls(
            persist_directory=os.environ.get("CHROMA_PERSIST_DIR"),
            host=os.environ.get("CHROMA_HOST"),
            port=int(os.environ.get("CHROMA_PORT", "8000")),
        )


@dataclass(frozen=True, kw_only=True)
class MilvusConfig(BaseVectorStoreConfig):
    """Connection parameters for a Milvus instance.

    TLS is driven entirely by the ``uri`` scheme, matching the ``MilvusClient``
    contract: an ``https://`` URI opens a secure gRPC channel, an ``http://`` URI
    stays plaintext. When the endpoint presents a certificate signed by a
    self-signed or private CA, pass the CA/server certificate as PEM text via
    ``server_cert``; :class:`~ai4rag.rag.vector_store.milvus.MilvusVectorStore`
    materializes it to a temporary file for pymilvus to verify against. Endpoints
    with publicly trusted certificates need no ``server_cert``.

    Parameters
    ----------
    uri : str
        Milvus server URI. Use ``https://host:port`` for TLS,
        ``http://host:port`` for plaintext.
    token : str | None
        Authentication token (``"user:password"``). ``None`` for unauthenticated.
    server_cert : str | None
        PEM-encoded server/CA certificate used to verify a TLS connection.
        Required only for self-signed or private-CA endpoints; leave ``None``
        when the server uses a publicly trusted certificate.
    provider : str, default="milvus"
        Name of the provider used in the system.
    """

    uri: str
    token: str | None = None
    server_cert: str | None = None
    provider: str = "milvus"

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        """Build config from ``MILVUS_*`` environment variables.

        Reads ``MILVUS_URI`` (required), plus the optional ``MILVUS_TOKEN`` and
        ``MILVUS_SERVER_CERT``. ``MILVUS_SERVER_CERT`` holds the PEM certificate
        text itself, not a filesystem path.

        Returns
        -------
        MilvusConfig
            Config populated from the ``MILVUS_*`` environment variables.

        Raises
        ------
        KeyError
            If the required ``MILVUS_URI`` variable is not set.
        """
        return cls(
            uri=os.environ["MILVUS_URI"],
            token=os.environ.get("MILVUS_TOKEN"),
            server_cert=os.environ.get("MILVUS_SERVER_CERT"),
        )


@dataclass(frozen=True, kw_only=True)
class PGVectorConfig(BaseVectorStoreConfig):
    """Connection parameters for a PostgreSQL + pgvector instance.

    Parameters
    ----------
    host : str
        PostgreSQL host address.
    port : int
        PostgreSQL port.
    dbname : str
        Database name.
    user : str
        Database user.
    password : str | None
        Database password. ``None`` for trust/peer auth.
    provider : str, default="pgvector"
        Name of the provider used in the system.
    """

    host: str = "localhost"
    port: int = 5432
    dbname: str = "postgres"
    user: str = "postgres"
    password: str | None = None
    provider: str = "pgvector"

    @classmethod
    def from_env(cls) -> "PGVectorConfig":
        """Build config from ``PGVECTOR_*`` environment variables.

        Reads ``PGVECTOR_HOST``, ``PGVECTOR_PORT``, ``PGVECTOR_DB``,
        ``PGVECTOR_USER`` and ``PGVECTOR_PASSWORD``. Unset variables fall back to
        the local-PostgreSQL defaults; ``PGVECTOR_PASSWORD`` defaults to ``None``
        for trust/peer authentication.

        Returns
        -------
        PGVectorConfig
            Config populated from the ``PGVECTOR_*`` environment variables.
        """
        return cls(
            host=os.environ.get("PGVECTOR_HOST", "localhost"),
            port=int(os.environ.get("PGVECTOR_PORT", "5432")),
            dbname=os.environ.get("PGVECTOR_DB", "postgres"),
            user=os.environ.get("PGVECTOR_USER", "postgres"),
            password=os.environ.get("PGVECTOR_PASSWORD"),
        )
