# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import os
from dataclasses import FrozenInstanceError

import pytest

from ai4rag.rag.vector_store.config import (
    ChromaConfig,
    MilvusConfig,
    PGVectorConfig,
    get_vector_store_config,
    get_vector_store_env_vars,
)


class TestChromaConfig:
    """Tests for ChromaConfig dataclass."""

    def test_defaults_are_ephemeral(self):
        cfg = ChromaConfig()
        assert cfg.persist_directory is None
        assert cfg.host is None
        assert cfg.port == 8000
        assert cfg.provider == "chroma"

    def test_custom_values(self):
        cfg = ChromaConfig(persist_directory="/data/chroma", host="chroma.local", port=9000)
        assert cfg.persist_directory == "/data/chroma"
        assert cfg.host == "chroma.local"
        assert cfg.port == 9000

    def test_frozen(self):
        cfg = ChromaConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.host = "other"

    def test_from_env_defaults(self, monkeypatch):
        for var in ("CHROMA_PERSIST_DIR", "CHROMA_HOST", "CHROMA_PORT"):
            monkeypatch.delenv(var, raising=False)
        cfg = ChromaConfig.from_env()
        assert cfg.persist_directory is None
        assert cfg.host is None
        assert cfg.port == 8000

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/chroma")
        monkeypatch.setenv("CHROMA_HOST", "chroma-server")
        monkeypatch.setenv("CHROMA_PORT", "9001")
        cfg = ChromaConfig.from_env()
        assert cfg.persist_directory == "/tmp/chroma"
        assert cfg.host == "chroma-server"
        assert cfg.port == 9001


class TestMilvusConfig:
    """Tests for MilvusConfig dataclass."""

    def test_constructor_uri_only(self):
        cfg = MilvusConfig(uri="https://milvus:19530")
        assert cfg.uri == "https://milvus:19530"
        assert cfg.token is None
        assert cfg.server_cert is None

    def test_constructor_uri_and_token(self):
        cfg = MilvusConfig(uri="https://milvus:19530", token="root:Milvus")
        assert cfg.uri == "https://milvus:19530"
        assert cfg.token == "root:Milvus"

    def test_constructor_with_server_cert(self):
        cert_pem = "-----BEGIN CERTIFICATE-----\nMIICert\n-----END CERTIFICATE-----\n"
        cfg = MilvusConfig(uri="https://milvus:19530", server_cert=cert_pem)
        assert cfg.server_cert == cert_pem

    def test_frozen(self):
        cfg = MilvusConfig(uri="http://localhost:19530")
        with pytest.raises(FrozenInstanceError):
            cfg.uri = "new"

    def test_from_env_uri_only(self, monkeypatch):
        monkeypatch.setenv("MILVUS_URI", "http://host:19530")
        monkeypatch.delenv("MILVUS_TOKEN", raising=False)
        monkeypatch.delenv("MILVUS_SERVER_CERT", raising=False)
        cfg = MilvusConfig.from_env()
        assert cfg.uri == "http://host:19530"
        assert cfg.token is None
        assert cfg.server_cert is None

    def test_from_env_uri_and_token(self, monkeypatch):
        monkeypatch.setenv("MILVUS_URI", "http://host:19530")
        monkeypatch.setenv("MILVUS_TOKEN", "user:pass")
        cfg = MilvusConfig.from_env()
        assert cfg.uri == "http://host:19530"
        assert cfg.token == "user:pass"

    def test_from_env_with_server_cert(self, monkeypatch):
        cert_pem = "-----BEGIN CERTIFICATE-----\nMIICert\n-----END CERTIFICATE-----\n"
        monkeypatch.setenv("MILVUS_URI", "https://host:19530")
        monkeypatch.setenv("MILVUS_SERVER_CERT", cert_pem)
        cfg = MilvusConfig.from_env()
        assert cfg.server_cert == cert_pem

    def test_from_env_missing_uri_raises(self, monkeypatch):
        monkeypatch.delenv("MILVUS_URI", raising=False)
        with pytest.raises(KeyError):
            MilvusConfig.from_env()


class TestPGVectorConfig:
    """Tests for PGVectorConfig dataclass."""

    def test_defaults(self):
        cfg = PGVectorConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.dbname == "postgres"
        assert cfg.user == "postgres"
        assert cfg.password is None

    def test_custom_values(self):
        cfg = PGVectorConfig(host="db.local", port=5433, dbname="mydb", user="admin", password="secret")
        assert cfg.host == "db.local"
        assert cfg.port == 5433
        assert cfg.dbname == "mydb"
        assert cfg.user == "admin"
        assert cfg.password == "secret"

    def test_frozen(self):
        cfg = PGVectorConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.host = "other"

    def test_from_env_defaults(self, monkeypatch):
        for var in ("PGVECTOR_HOST", "PGVECTOR_PORT", "PGVECTOR_DB", "PGVECTOR_USER", "PGVECTOR_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        cfg = PGVectorConfig.from_env()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.dbname == "postgres"
        assert cfg.user == "postgres"
        assert cfg.password is None

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("PGVECTOR_HOST", "pghost")
        monkeypatch.setenv("PGVECTOR_PORT", "5433")
        monkeypatch.setenv("PGVECTOR_DB", "testdb")
        monkeypatch.setenv("PGVECTOR_USER", "testuser")
        monkeypatch.setenv("PGVECTOR_PASSWORD", "testpass")
        cfg = PGVectorConfig.from_env()
        assert cfg.host == "pghost"
        assert cfg.port == 5433
        assert cfg.dbname == "testdb"
        assert cfg.user == "testuser"
        assert cfg.password == "testpass"


class TestGetVectorStoreConfig:
    """Tests for the ``get_vector_store_config`` provider factory."""

    def test_returns_chroma_config(self, monkeypatch):
        for var in ("CHROMA_PERSIST_DIR", "CHROMA_HOST", "CHROMA_PORT"):
            monkeypatch.delenv(var, raising=False)
        cfg = get_vector_store_config("chroma")
        assert isinstance(cfg, ChromaConfig)
        assert cfg.provider == "chroma"

    def test_returns_pgvector_config(self, monkeypatch):
        for var in ("PGVECTOR_HOST", "PGVECTOR_PORT", "PGVECTOR_DB", "PGVECTOR_USER", "PGVECTOR_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        cfg = get_vector_store_config("pgvector")
        assert isinstance(cfg, PGVectorConfig)
        assert cfg.provider == "pgvector"

    def test_returns_milvus_config_from_env(self, monkeypatch):
        monkeypatch.setenv("MILVUS_URI", "http://host:19530")
        monkeypatch.delenv("MILVUS_TOKEN", raising=False)
        monkeypatch.delenv("MILVUS_SERVER_CERT", raising=False)
        cfg = get_vector_store_config("milvus")
        assert isinstance(cfg, MilvusConfig)
        assert cfg.uri == "http://host:19530"

    def test_milvus_missing_uri_raises_key_error(self, monkeypatch):
        """The factory must surface the backend's own ``from_env`` failure."""
        monkeypatch.delenv("MILVUS_URI", raising=False)
        with pytest.raises(KeyError):
            get_vector_store_config("milvus")

    def test_unsupported_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_vector_store_config("qdrant")


class TestGetVectorStoreEnvVars:
    """Tests for the ``get_vector_store_env_vars`` documentation helper."""

    @pytest.mark.parametrize(
        ("provider", "config_cls"),
        [("chroma", ChromaConfig), ("milvus", MilvusConfig), ("pgvector", PGVectorConfig)],
    )
    def test_matches_config_class_env_vars(self, provider, config_cls):
        """The helper must return the exact ``env_vars`` tuple declared on the config class."""
        assert get_vector_store_env_vars(provider) == config_cls.env_vars

    def test_returns_name_description_pairs(self):
        env_vars = get_vector_store_env_vars("milvus")
        assert env_vars  # non-empty
        names = [name for name, _ in env_vars]
        assert "MILVUS_URI" in names
        assert all(isinstance(name, str) and isinstance(desc, str) for name, desc in env_vars)

    def test_unsupported_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_vector_store_env_vars("qdrant")
