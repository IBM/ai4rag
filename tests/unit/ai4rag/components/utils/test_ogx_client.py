# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Tests for :mod:`ai4rag.components.utils.ogx_client` -- OGX client factory with SSL fallback."""

from __future__ import annotations

import ssl

import pytest


class TestIsSslError:
    """Test suite for :func:`is_ssl_error`."""

    def test_detects_certificate_verify_failed(self):
        """An exception whose message contains ``CERTIFICATE_VERIFY_FAILED`` is recognized."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        exc = Exception("CERTIFICATE_VERIFY_FAILED: self-signed certificate")
        assert is_ssl_error(exc) is True

    def test_detects_ssl_keyword(self):
        """An exception whose message contains ``SSL`` (any case) is recognized."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        exc = Exception("ssl handshake error")
        assert is_ssl_error(exc) is True

    def test_returns_false_for_unrelated_error(self):
        """Non-SSL exceptions must return ``False``."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        exc = Exception("Connection refused")
        assert is_ssl_error(exc) is False

    def test_follows_cause_chain(self):
        """SSL error buried in ``__cause__`` should be detected."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        root = Exception("SSL: CERTIFICATE_VERIFY_FAILED")
        wrapper = RuntimeError("request failed")
        wrapper.__cause__ = root

        assert is_ssl_error(wrapper) is True

    def test_follows_context_chain(self):
        """SSL error in ``__context__`` (implicit chaining) should be detected."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        root = Exception("SSL error on connect")
        wrapper = RuntimeError("something happened")
        wrapper.__context__ = root

        assert is_ssl_error(wrapper) is True

    def test_handles_circular_chain_without_infinite_loop(self):
        """A circular cause chain must not cause infinite recursion."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        a = RuntimeError("error A")
        b = RuntimeError("error B")
        a.__cause__ = b
        b.__cause__ = a  # cycle

        # Must terminate without hanging.
        assert is_ssl_error(a) is False

    def test_case_insensitive_detection(self):
        """Detection should be case-insensitive (message is uppercased internally)."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        assert is_ssl_error(Exception("certificate_verify_failed")) is True
        assert is_ssl_error(Exception("Ssl connection reset")) is True

    def test_returns_false_for_empty_message(self):
        """An exception with an empty message should not match."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        assert is_ssl_error(Exception("")) is False

    def test_detects_real_ssl_verification_error(self):
        """A real ``ssl.SSLCertVerificationError`` should be detected."""
        from ai4rag.components.utils.ogx_client import is_ssl_error

        exc = ssl.SSLCertVerificationError("SSL: CERTIFICATE_VERIFY_FAILED")
        assert is_ssl_error(exc) is True


class TestCreateOgxClient:
    """Test suite for :func:`create_ogx_client`."""

    def test_returns_client_on_successful_connection(self, mocker):
        """When ``models.list()`` succeeds, the original client is returned."""
        mock_ogx_cls = mocker.patch("ai4rag.components.utils.ogx_client.OgxClient")
        mock_client = mocker.MagicMock()
        mock_ogx_cls.return_value = mock_client

        from ai4rag.components.utils.ogx_client import create_ogx_client

        result = create_ogx_client(base_url="https://ogx.example.com", api_key="test-key")

        assert result is mock_client
        mock_client.models.list.assert_called_once()
        # OgxClient should have been instantiated exactly once (no fallback).
        mock_ogx_cls.assert_called_once_with(base_url="https://ogx.example.com", api_key="test-key")

    def test_falls_back_to_unverified_tls_on_ssl_error(self, mocker):
        """An SSL verification failure triggers a retry with ``verify=False``."""
        mock_ogx_cls = mocker.patch("ai4rag.components.utils.ogx_client.OgxClient")
        mock_httpx_client = mocker.patch("ai4rag.components.utils.ogx_client.httpx.Client")

        first_client = mocker.MagicMock()
        first_client.models.list.side_effect = ssl.SSLCertVerificationError("SSL: CERTIFICATE_VERIFY_FAILED")
        fallback_client = mocker.MagicMock()
        mock_ogx_cls.side_effect = [first_client, fallback_client]

        from ai4rag.components.utils.ogx_client import create_ogx_client

        result = create_ogx_client(base_url="https://ogx.example.com", api_key="test-key")

        assert result is fallback_client
        assert mock_ogx_cls.call_count == 2
        # The fallback call should include an httpx.Client(verify=False).
        fallback_call_kwargs = mock_ogx_cls.call_args_list[1].kwargs
        assert "http_client" in fallback_call_kwargs
        mock_httpx_client.assert_called_once_with(verify=False)

    def test_falls_back_on_chained_ssl_error(self, mocker):
        """An ``httpx.ConnectError`` whose cause chain contains an SSL error triggers fallback."""
        from httpx import ConnectError

        mock_ogx_cls = mocker.patch("ai4rag.components.utils.ogx_client.OgxClient")
        mocker.patch("ai4rag.components.utils.ogx_client.httpx.Client")

        ssl_root = ssl.SSLCertVerificationError("SSL: CERTIFICATE_VERIFY_FAILED")
        chained_err = ConnectError("connection failed")
        chained_err.__cause__ = ssl_root

        first_client = mocker.MagicMock()
        first_client.models.list.side_effect = chained_err
        fallback_client = mocker.MagicMock()
        mock_ogx_cls.side_effect = [first_client, fallback_client]

        from ai4rag.components.utils.ogx_client import create_ogx_client

        result = create_ogx_client(base_url="https://ogx.example.com", api_key="key")
        assert result is fallback_client

    def test_reraises_non_ssl_connection_error(self, mocker):
        """A connection error that is not SSL-related should propagate."""
        from httpx import ConnectError

        mock_ogx_cls = mocker.patch("ai4rag.components.utils.ogx_client.OgxClient")
        first_client = mocker.MagicMock()
        non_ssl_error = ConnectError("Connection refused")
        first_client.models.list.side_effect = non_ssl_error
        mock_ogx_cls.return_value = first_client

        from ai4rag.components.utils.ogx_client import create_ogx_client

        with pytest.raises(ConnectError, match="Connection refused"):
            create_ogx_client(base_url="https://ogx.example.com", api_key="key")

    def test_logs_warning_on_ssl_fallback(self, mocker):
        """A warning should be logged when falling back to unverified TLS."""
        mock_ogx_cls = mocker.patch("ai4rag.components.utils.ogx_client.OgxClient")
        mocker.patch("ai4rag.components.utils.ogx_client.httpx.Client")
        mock_logger = mocker.patch("ai4rag.components.utils.ogx_client._logger")

        first_client = mocker.MagicMock()
        first_client.models.list.side_effect = ssl.SSLCertVerificationError("SSL: CERTIFICATE_VERIFY_FAILED")
        fallback_client = mocker.MagicMock()
        mock_ogx_cls.side_effect = [first_client, fallback_client]

        from ai4rag.components.utils.ogx_client import create_ogx_client

        create_ogx_client(base_url="https://ogx.example.com", api_key="key")

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "SSL" in warning_msg or "verify=False" in warning_msg
