# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

__all__ = [
    "SUPPORTED_PROVIDERS",
    "BaseVectorStoreConfig",
    "MilvusConfig",
    "MilvusLiteConfig",
    "PGVectorConfig",
    "get_vector_store_config",
    "get_vector_store_env_vars",
]

#: Default on-disk location for an embedded Milvus Lite database when the caller
#: does not specify one. A relative path lands in the current working directory.
DEFAULT_MILVUS_LITE_DB_PATH = "./ai4rag_milvus_lite.db"


def _is_server_url(value: object) -> bool:
    """Return whether *value* is a Milvus server URL (``http://``/``https://``).

    Shared by :meth:`MilvusConfig.__post_init__` and
    :meth:`MilvusLiteConfig.__post_init__` so the two mirror-image guards can
    never drift apart on which schemes count as a "server" endpoint.
    """
    return isinstance(value, str) and value.startswith(("http://", "https://"))


@dataclass(frozen=True, kw_only=True)
class BaseVectorStoreConfig(ABC):
    """Base config shared by every vector store backend.

    Attributes
    ----------
    provider : str
        Backend discriminator (``"milvus"``, ``"milvus_lite"``, or ``"pgvector"``)
        used by :func:`ai4rag.rag.vector_store.get_vector_store.get_vector_store`
        to select the concrete store class.
    """

    provider: str
    env_vars = None

    @classmethod
    @abstractmethod
    def from_env(cls) -> "BaseVectorStoreConfig":
        """Create config from environment variables."""


@dataclass(frozen=True, kw_only=True)
class MilvusConfig(BaseVectorStoreConfig):
    """Connection parameters for a **remote** Milvus server.

    This config targets a running Milvus (or Zilliz Cloud) instance reached over
    gRPC. For an embedded, local, zero-server database use
    :class:`MilvusLiteConfig` instead — the two are deliberately separate so that
    a mistyped or unreachable server ``uri`` fails loudly rather than silently
    spinning up a throwaway local database (a dangerous surprise in production).

    To enforce that, ``uri`` **must** be an ``http://`` or ``https://`` URL;
    anything else (a bare host, a file path, an empty string) is rejected at
    construction. TLS is driven by the scheme: ``https://`` opens a secure gRPC
    channel, ``http://`` stays plaintext. When a remote endpoint presents a
    certificate signed by a self-signed or private CA, pass the CA/server
    certificate as PEM text via ``server_cert``;
    :class:`~ai4rag.rag.vector_store.milvus.MilvusVectorStore` materializes it to
    a temporary file for pymilvus to verify against. Endpoints with publicly
    trusted certificates need no ``server_cert``.

    Parameters
    ----------
    uri : str
        Milvus server endpoint. Must start with ``http://`` (plaintext) or
        ``https://`` (TLS), e.g. ``https://host:19530``.
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

    Raises
    ------
    ValueError
        If ``uri`` is not an ``http://`` or ``https://`` URL.
    """

    env_vars: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "MILVUS_URI",
            "Milvus server endpoint URL: https://host:port (TLS) or http://host:port (plaintext). "
            "For a local embedded database, use the milvus_lite provider instead. (required)",
        ),
        ("MILVUS_TOKEN", "Authentication token in 'user:password' form. (optional)"),
        ("MILVUS_SERVER_CERT", "PEM-encoded CA/server certificate for self-signed TLS endpoints. (optional)"),
    )

    uri: str
    token: str | None = None
    server_cert: str | None = None
    provider: str = "milvus"

    def __post_init__(self) -> None:
        """Reject any ``uri`` that is not an explicit Milvus server URL.

        Guards against the footgun where an incorrect ``uri`` (a typo, a bare
        hostname, or a stray path) is silently interpreted by ``MilvusClient`` as
        a local Milvus Lite database file, creating a throwaway store instead of
        connecting to the intended server. Local, embedded use must go through
        :class:`MilvusLiteConfig`.
        """
        if not _is_server_url(self.uri):
            raise ValueError(
                f"MilvusConfig.uri must be a Milvus server URL starting with 'http://' or 'https://', "
                f"got {self.uri!r}. For a local, embedded database use MilvusLiteConfig(db_path=...) "
                "(provider 'milvus_lite') instead."
            )

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
        ValueError
            If ``MILVUS_URI`` is not an ``http://``/``https://`` URL.
        """
        return cls(
            uri=os.environ["MILVUS_URI"],
            token=os.environ.get("MILVUS_TOKEN"),
            server_cert=os.environ.get("MILVUS_SERVER_CERT"),
        )


@dataclass(frozen=True, kw_only=True)
class MilvusLiteConfig(BaseVectorStoreConfig):
    """Connection parameters for an **embedded, local** Milvus Lite database.

    Milvus Lite is the zero-server Milvus engine bundled with
    ``pymilvus[milvus-lite]``; it stores everything in a single local file and is
    the recommended lightweight option for local development, tests, and
    small-scale workloads (prototyping, up to roughly one million vectors) — not
    production serving. For a remote server use :class:`MilvusConfig`.

    Choosing the embedded engine is explicit: it happens only when this config is
    used (provider ``"milvus_lite"``), never as a silent fallback from a
    misconfigured :class:`MilvusConfig`.

    Parameters
    ----------
    db_path : str, default=:data:`DEFAULT_MILVUS_LITE_DB_PATH`
        Local filesystem path to the Milvus Lite database file. Created on first
        use; a relative path resolves against the current working directory.
    provider : str, default="milvus_lite"
        Name of the provider used in the system.

    Attributes
    ----------
    env_vars : ClassVar[tuple[tuple[str, str], ...]]
        ``(name, description)`` pairs for the environment variables consulted by
        :meth:`from_env`. Exposed for documentation and notebook generation.

    Raises
    ------
    ValueError
        If ``db_path`` is empty/blank, or looks like a server URL
        (``http://``/``https://``).
    """

    env_vars: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "MILVUS_LITE_DB_PATH",
            f"Local file path for the embedded Milvus Lite database "
            f"(default {DEFAULT_MILVUS_LITE_DB_PATH}). (optional)",
        ),
    )

    db_path: str = DEFAULT_MILVUS_LITE_DB_PATH
    provider: str = "milvus_lite"

    def __post_init__(self) -> None:
        """Reject a ``db_path`` that is blank or is actually a server URL.

        The symmetric guard to :meth:`MilvusConfig.__post_init__`: a value like
        ``https://host:19530`` is a server endpoint, not a local database file,
        and belongs in :class:`MilvusConfig`. An empty or whitespace-only path
        is rejected here too, rather than being handed to ``MilvusClient`` where
        it would surface as an opaque, hard-to-trace pymilvus error.
        """
        if not isinstance(self.db_path, str) or not self.db_path.strip():
            raise ValueError(
                f"MilvusLiteConfig.db_path must be a non-empty local filesystem path, got {self.db_path!r}."
            )
        if _is_server_url(self.db_path):
            raise ValueError(
                f"MilvusLiteConfig.db_path must be a local filesystem path, not a server URL, "
                f"got {self.db_path!r}. For a remote Milvus server use MilvusConfig(uri=...) "
                "(provider 'milvus') instead."
            )

    @classmethod
    def from_env(cls) -> "MilvusLiteConfig":
        """Build config from the ``MILVUS_LITE_DB_PATH`` environment variable.

        An unset variable falls back to :data:`DEFAULT_MILVUS_LITE_DB_PATH`.

        Returns
        -------
        MilvusLiteConfig
            Config populated from ``MILVUS_LITE_DB_PATH`` (or the default path).
        """
        return cls(db_path=os.environ.get("MILVUS_LITE_DB_PATH", DEFAULT_MILVUS_LITE_DB_PATH))


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
    pool_max_size : int, default=10
        Maximum number of concurrent connections the store's connection pool
        will open. The pool starts lean and grows lazily on demand, so this is
        a ceiling, not an eagerly-held count; it should be set to at least the
        maximum number of concurrent ``search()``/``add_documents()`` calls the
        caller will issue against this store, or those calls will queue for a
        slot and can eventually time out.
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
    pool_max_size: int = 10
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
# Wrapped in a read-only view so importers cannot mutate the shared mapping.
_CONFIG_BY_PROVIDER: MappingProxyType[str, type[BaseVectorStoreConfig]] = MappingProxyType(
    {config_cls.provider: config_cls for config_cls in (MilvusConfig, MilvusLiteConfig, PGVectorConfig)}
)

#: Provider discriminators accepted by :func:`get_vector_store_config` and
#: :func:`ai4rag.rag.vector_store.get_vector_store.get_vector_store`, derived from
#: the registry above so callers outside this package (e.g. the search space
#: defaults) never have to hand-maintain a second copy of this list.
SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(sorted(_CONFIG_BY_PROVIDER))


def _resolve_config_cls(provider: str) -> type[BaseVectorStoreConfig]:
    """Return the config class registered for *provider*.

    The single source of truth for which config class a provider discriminator
    maps to; used both to build a config from the environment (below) and by
    :func:`ai4rag.rag.vector_store.get_vector_store.get_vector_store` to check
    that a caller-supplied config matches its declared ``provider``. Not part of
    the package's public API (see ``ai4rag/rag/vector_store/__init__.py``) —
    both call sites live inside this package, so a leading-underscore, directly
    imported helper is enough without widening the public surface.

    Raises
    ------
    ValueError
        If *provider* does not name a supported backend.
    """
    try:
        return _CONFIG_BY_PROVIDER[provider]
    except KeyError as exc:
        raise ValueError(
            f"Vector store provider '{provider}' is not supported. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        ) from exc


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
        Backend discriminator, one of ``"milvus"``, ``"milvus_lite"`` or
        ``"pgvector"``.

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
        Backend discriminator, one of ``"milvus"``, ``"milvus_lite"`` or
        ``"pgvector"``.

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
