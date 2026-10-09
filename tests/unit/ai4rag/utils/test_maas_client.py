# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import ssl

import pytest


class TestIsSslError:
    """Test suite for :func:`is_ssl_error`."""

    def test_detects_certificate_verify_failed(self):
        """An exception whose message contains ``CERTIFICATE_VERIFY_FAILED`` is recognized."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        exc = Exception("CERTIFICATE_VERIFY_FAILED: self-signed certificate")
        assert is_ssl_error(exc) is True

    def test_detects_ssl_keyword(self):
        """An exception whose message contains ``SSL`` (any case) is recognized."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        exc = Exception("ssl handshake error")
        assert is_ssl_error(exc) is True

    def test_returns_false_for_unrelated_error(self):
        """Non-SSL exceptions must return ``False``."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        exc = Exception("Connection refused")
        assert is_ssl_error(exc) is False

    def test_follows_cause_chain(self):
        """SSL error buried in ``__cause__`` should be detected."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        root = Exception("SSL: CERTIFICATE_VERIFY_FAILED")
        wrapper = RuntimeError("request failed")
        wrapper.__cause__ = root

        assert is_ssl_error(wrapper) is True

    def test_follows_context_chain(self):
        """SSL error in ``__context__`` (implicit chaining) should be detected."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        root = Exception("SSL error on connect")
        wrapper = RuntimeError("something happened")
        wrapper.__context__ = root

        assert is_ssl_error(wrapper) is True

    def test_handles_circular_chain_without_infinite_loop(self):
        """A circular cause chain must not cause infinite recursion."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        a = RuntimeError("error A")
        b = RuntimeError("error B")
        a.__cause__ = b
        b.__cause__ = a  # cycle

        # Must terminate without hanging.
        assert is_ssl_error(a) is False

    def test_case_insensitive_detection(self):
        """Detection should be case-insensitive (message is uppercased internally)."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        assert is_ssl_error(Exception("certificate_verify_failed")) is True
        assert is_ssl_error(Exception("Ssl connection reset")) is True

    def test_returns_false_for_empty_message(self):
        """An exception with an empty message should not match."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        assert is_ssl_error(Exception("")) is False

    def test_detects_real_ssl_verification_error(self):
        """A real ``ssl.SSLCertVerificationError`` should be detected."""
        from ai4rag.utils.clients.maas_client import is_ssl_error

        exc = ssl.SSLCertVerificationError("SSL: CERTIFICATE_VERIFY_FAILED")
        assert is_ssl_error(exc) is True


class TestCreateMaasClient:
    """Test suite for :func:`create_maas_client`."""

    def test_returns_client_on_successful_connection(self, mocker):
        """When ``models.list()`` succeeds, the original client is returned."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_client = mocker.MagicMock()
        mock_openai_cls.return_value = mock_client

        from ai4rag.utils.clients.maas_client import create_maas_client

        result = create_maas_client(base_url="https://maas.example.com", api_key="test-key")

        assert result is mock_client
        mock_client.models.list.assert_called_once()
        # OpenAI should have been instantiated exactly once (no fallback), with /v1 appended.
        mock_openai_cls.assert_called_once_with(base_url="https://maas.example.com/v1", api_key="test-key")

    def test_does_not_duplicate_v1_suffix(self, mocker):
        """A base URL that already ends with ``/v1`` is passed through unchanged."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_openai_cls.return_value = mocker.MagicMock()

        from ai4rag.utils.clients.maas_client import create_maas_client

        create_maas_client(base_url="https://maas.example.com/v1", api_key="test-key")

        mock_openai_cls.assert_called_once_with(base_url="https://maas.example.com/v1", api_key="test-key")

    def test_strips_trailing_slash_before_appending_v1(self, mocker):
        """A trailing slash on the base URL doesn't produce a double slash before ``/v1``."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_openai_cls.return_value = mocker.MagicMock()

        from ai4rag.utils.clients.maas_client import create_maas_client

        create_maas_client(base_url="https://maas.example.com/", api_key="test-key")

        mock_openai_cls.assert_called_once_with(base_url="https://maas.example.com/v1", api_key="test-key")

    def test_reraises_non_ssl_connection_error(self, mocker):
        """A connection error that is not SSL-related should propagate."""
        from httpx import ConnectError

        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        first_client = mocker.MagicMock()
        non_ssl_error = ConnectError("Connection refused")
        first_client.models.list.side_effect = non_ssl_error
        mock_openai_cls.return_value = first_client

        from ai4rag.utils.clients.maas_client import create_maas_client

        with pytest.raises(ConnectError, match="Connection refused"):
            create_maas_client(base_url="https://maas.example.com", api_key="key")

    def test_rejects_plain_http_to_external_host(self, mocker):
        """A plaintext http:// base_url against a non-local host is rejected before any network call."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")

        from ai4rag.utils.clients.maas_client import create_maas_client

        with pytest.raises(ValueError, match="MaaS base_url"):
            create_maas_client(base_url="http://maas.example.com", api_key="test-key")

        mock_openai_cls.assert_not_called()

    def test_allows_plain_http_to_localhost(self, mocker):
        """A plaintext http:// base_url against localhost is allowed."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_openai_cls.return_value = mocker.MagicMock()

        from ai4rag.utils.clients.maas_client import create_maas_client

        create_maas_client(base_url="http://localhost:8080", api_key="test-key")

        mock_openai_cls.assert_called_once_with(base_url="http://localhost:8080/v1", api_key="test-key")

    def test_allows_plain_http_to_cluster_local_host(self, mocker):
        """A plaintext http:// base_url against an in-cluster '*.cluster.local' host is allowed."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_openai_cls.return_value = mocker.MagicMock()

        from ai4rag.utils.clients.maas_client import create_maas_client

        create_maas_client(base_url="http://maas.svc.cluster.local:8080", api_key="test-key")

        mock_openai_cls.assert_called_once()

    def test_ssl_failure_propagates_without_fallback(self, mocker):
        """An SSL verification failure re-raises the original error; no unverified retry is made."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_logger = mocker.patch("ai4rag.utils.clients.maas_client._logger")

        client = mocker.MagicMock()
        ssl_error = ssl.SSLCertVerificationError("SSL: CERTIFICATE_VERIFY_FAILED")
        client.models.list.side_effect = ssl_error
        mock_openai_cls.return_value = client

        from ai4rag.utils.clients.maas_client import create_maas_client

        with pytest.raises(ssl.SSLCertVerificationError):
            create_maas_client(base_url="https://maas.example.com", api_key="key")

        mock_openai_cls.assert_called_once_with(base_url="https://maas.example.com/v1", api_key="key")
        mock_logger.error.assert_called_once()
        assert "MAAS_CA_BUNDLE" in mock_logger.error.call_args[0][0]

    def test_uses_explicit_ca_bundle(self, mocker):
        """An explicit ca_bundle is used to build the httpx.Client that verifies the connection."""
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_openai_cls.return_value = mocker.MagicMock()
        mock_httpx_client = mocker.patch("ai4rag.utils.clients.maas_client.httpx.Client")

        from ai4rag.utils.clients.maas_client import create_maas_client

        create_maas_client(base_url="https://maas.example.com", api_key="key", ca_bundle="/etc/ca.pem")

        mock_httpx_client.assert_called_once_with(verify="/etc/ca.pem")
        assert mock_openai_cls.call_args.kwargs["http_client"] is mock_httpx_client.return_value

    def test_uses_maas_ca_bundle_env_var(self, mocker, monkeypatch):
        """The MAAS_CA_BUNDLE environment variable is used when no explicit ca_bundle is passed."""
        monkeypatch.setenv("MAAS_CA_BUNDLE", "/etc/env-ca.pem")
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_openai_cls.return_value = mocker.MagicMock()
        mock_httpx_client = mocker.patch("ai4rag.utils.clients.maas_client.httpx.Client")

        from ai4rag.utils.clients.maas_client import create_maas_client

        create_maas_client(base_url="https://maas.example.com", api_key="key")

        mock_httpx_client.assert_called_once_with(verify="/etc/env-ca.pem")

    def test_explicit_ca_bundle_overrides_env_var(self, mocker, monkeypatch):
        """An explicit ca_bundle argument takes precedence over MAAS_CA_BUNDLE."""
        monkeypatch.setenv("MAAS_CA_BUNDLE", "/etc/env-ca.pem")
        mock_openai_cls = mocker.patch("ai4rag.utils.clients.maas_client.OpenAI")
        mock_openai_cls.return_value = mocker.MagicMock()
        mock_httpx_client = mocker.patch("ai4rag.utils.clients.maas_client.httpx.Client")

        from ai4rag.utils.clients.maas_client import create_maas_client

        create_maas_client(base_url="https://maas.example.com", api_key="key", ca_bundle="/etc/explicit-ca.pem")

        mock_httpx_client.assert_called_once_with(verify="/etc/explicit-ca.pem")
