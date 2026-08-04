# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import os
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "BaseVectorStoreConfig",
    "MilvusConfig",
    "PGVectorConfig",
    "ChromaConfig",
    "get_vector_store_config",
    "get_vector_store_env_vars",
]


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

    Attributes
    ----------
    env_vars : ClassVar[tuple[tuple[str, str], ...]]
        ``(name, description)`` pairs for the environment variables consulted by
        :meth:`from_env`. Exposed for documentation and notebook generation.
    """

    env_vars: ClassVar[tuple[tuple[str, str], ...]] = (
        ("CHROMA_HOST", "Hostname of a remote Chroma server. Leave unset to run locally."),
        ("CHROMA_PORT", "Port of the remote Chroma server (used with CHROMA_HOST; default 8000)."),
        (
            "CHROMA_PERSIST_DIR",
            "Filesystem path for a local persistent store. Unset uses an ephemeral in-memory store.",
        ),
    )

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

    Attributes
    ----------
    env_vars : ClassVar[tuple[tuple[str, str], ...]]
        ``(name, description)`` pairs for the environment variables consulted by
        :meth:`from_env`. Exposed for documentation and notebook generation.
    """

    env_vars: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "MILVUS_URI",
            "Milvus server URI. Use https://host:port for TLS or http://host:port for plaintext. (required)",
        ),
        ("MILVUS_TOKEN", "Authentication token in 'user:password' form. (optional)"),
        ("MILVUS_SERVER_CERT", "PEM-encoded CA/server certificate for self-signed TLS endpoints. (optional)"),
    )

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

    Attributes
    ----------
    env_vars : ClassVar[tuple[tuple[str, str], ...]]
        ``(name, description)`` pairs for the environment variables consulted by
        :meth:`from_env`. Exposed for documentation and notebook generation.
    """

    env_vars: ClassVar[tuple[tuple[str, str], ...]] = (
        ("PGVECTOR_HOST", "PostgreSQL host (default localhost)."),
        ("PGVECTOR_PORT", "PostgreSQL port (default 5432)."),
        ("PGVECTOR_DB", "Database name (default postgres)."),
        ("PGVECTOR_USER", "Database user (default postgres)."),
        ("PGVECTOR_PASSWORD", "Database password. Unset uses trust/peer authentication."),
    )

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


# Registry mapping a provider discriminator to its config class. Built from each
# class's ``provider`` default so the provider string has a single source of truth.
_CONFIG_BY_PROVIDER: dict[str, type[BaseVectorStoreConfig]] = {
    config_cls.provider: config_cls for config_cls in (ChromaConfig, MilvusConfig, PGVectorConfig)
}


def _resolve_config_cls(provider: str) -> type[BaseVectorStoreConfig]:
    """Return the config class registered for *provider*.

    Raises
    ------
    ValueError
        If *provider* does not name a supported backend.
    """
    try:
        return _CONFIG_BY_PROVIDER[provider]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONFIG_BY_PROVIDER))
        raise ValueError(f"Vector store provider '{provider}' is not supported. Choose one of: {supported}.") from exc


def get_vector_store_config(provider: str) -> BaseVectorStoreConfig:
    """Build a vector store config for *provider* from environment variables.

    Companion to :func:`ai4rag.rag.vector_store.get_vector_store.get_vector_store`:
    given only a provider discriminator, it selects the matching config class and
    populates it from that backend's ``*_ENV`` variables via ``from_env``. Keeping
    connection details in the environment means secrets never have to be embedded
    in generated artefacts (e.g. pattern notebooks).

    Parameters
    ----------
    provider : str
        Backend discriminator, one of ``"chroma"``, ``"milvus"`` or ``"pgvector"``.

    Returns
    -------
    BaseVectorStoreConfig
        A config instance of the class matching *provider*, populated from the
        environment.

    Raises
    ------
    ValueError
        If *provider* names an unsupported backend.
    KeyError
        If a variable required by the selected backend's ``from_env`` is unset
        (e.g. ``MILVUS_URI`` for Milvus).

    Examples
    --------
    >>> config = get_vector_store_config("milvus")  # reads MILVUS_URI, ...
    >>> store = get_vector_store(embedding_model, config, collection_name="ai4rag_docs")
    """
    return _resolve_config_cls(provider).from_env()


def get_vector_store_env_vars(provider: str) -> tuple[tuple[str, str], ...]:
    """Return the environment variables consulted by *provider*'s ``from_env``.

    Parameters
    ----------
    provider : str
        Backend discriminator, one of ``"chroma"``, ``"milvus"`` or ``"pgvector"``.

    Returns
    -------
    tuple[tuple[str, str], ...]
        ``(name, description)`` pairs, in the order they should be presented to
        a user. Descriptions note whether each variable is required or optional.

    Raises
    ------
    ValueError
        If *provider* names an unsupported backend.
    """
    return _resolve_config_cls(provider).env_vars
