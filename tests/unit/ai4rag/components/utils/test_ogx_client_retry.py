# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Tests for OGX client creation."""

from unittest.mock import MagicMock, patch

from ai4rag.components.utils.ogx_client import create_ogx_client


class TestCreateOgxClient:
    """Tests for create_ogx_client function."""

    @patch("ai4rag.components.utils.ogx_client.OgxClient")
    def test_returns_client(self, mock_ogx_client):
        """Test that function returns OgxClient."""
        mock_client_instance = MagicMock()
        mock_client_instance.models.list.return_value = []
        mock_ogx_client.return_value = mock_client_instance

        client = create_ogx_client(base_url="http://test.com", api_key="test-key")

        # Verify client is returned
        assert client == mock_client_instance

        # Verify OgxClient was called
        mock_ogx_client.assert_called_once_with(
            base_url="http://test.com",
            api_key="test-key",
        )

    @patch("ai4rag.components.utils.ogx_client.httpx.Client")
    @patch("ai4rag.components.utils.ogx_client.OgxClient")
    def test_ssl_fallback(self, mock_ogx_client, mock_httpx_client):
        """Test SSL fallback creates client with verify=False."""
        import ssl

        # First client creation fails with SSL error
        mock_client_instance_1 = MagicMock()
        mock_client_instance_1.models.list.side_effect = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")

        # Second client creation succeeds
        mock_client_instance_2 = MagicMock()
        mock_client_instance_2.models.list.return_value = []

        mock_ogx_client.side_effect = [mock_client_instance_1, mock_client_instance_2]

        mock_http_client = MagicMock()
        mock_httpx_client.return_value = mock_http_client

        client = create_ogx_client(
            base_url="http://test.com",
            api_key="test-key",
        )

        # Verify first call with SSL verification
        assert mock_ogx_client.call_count == 2
        first_call = mock_ogx_client.call_args_list[0]
        assert first_call.kwargs["base_url"] == "http://test.com"
        assert first_call.kwargs["api_key"] == "test-key"

        # Verify second call with verify=False
        second_call = mock_ogx_client.call_args_list[1]
        assert second_call.kwargs["base_url"] == "http://test.com"
        assert second_call.kwargs["api_key"] == "test-key"
        assert second_call.kwargs["http_client"] == mock_http_client

        # Verify httpx.Client was created with verify=False
        mock_httpx_client.assert_called_once_with(verify=False)

        assert client == mock_client_instance_2
