# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Tests for MaaS client creation."""

from unittest.mock import MagicMock, patch

from ai4rag.utils.clients.maas_client import create_maas_client


class TestCreateMaasClient:
    """Tests for create_maas_client function."""

    @patch("ai4rag.utils.clients.maas_client.OpenAI")
    def test_returns_client(self, mock_openai_cls):
        """Test that function returns an OpenAI client."""
        mock_client_instance = MagicMock()
        mock_client_instance.models.list.return_value = []
        mock_openai_cls.return_value = mock_client_instance

        client = create_maas_client(base_url="http://test.com", api_key="test-key")

        # Verify client is returned
        assert client == mock_client_instance

        # Verify OpenAI was called, with /v1 appended to the base URL
        mock_openai_cls.assert_called_once_with(
            base_url="http://test.com/v1",
            api_key="test-key",
        )

    @patch("ai4rag.utils.clients.maas_client.httpx.Client")
    @patch("ai4rag.utils.clients.maas_client.OpenAI")
    def test_ssl_fallback(self, mock_openai_cls, mock_httpx_client):
        """Test SSL fallback creates client with verify=False."""
        import ssl

        # First client creation fails with SSL error
        mock_client_instance_1 = MagicMock()
        mock_client_instance_1.models.list.side_effect = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")

        # Second client creation succeeds
        mock_client_instance_2 = MagicMock()
        mock_client_instance_2.models.list.return_value = []

        mock_openai_cls.side_effect = [mock_client_instance_1, mock_client_instance_2]

        mock_http_client = MagicMock()
        mock_httpx_client.return_value = mock_http_client

        client = create_maas_client(
            base_url="http://test.com",
            api_key="test-key",
        )

        # Verify first call with SSL verification
        assert mock_openai_cls.call_count == 2
        first_call = mock_openai_cls.call_args_list[0]
        assert first_call.kwargs["base_url"] == "http://test.com/v1"
        assert first_call.kwargs["api_key"] == "test-key"

        # Verify second call with verify=False
        second_call = mock_openai_cls.call_args_list[1]
        assert second_call.kwargs["base_url"] == "http://test.com/v1"
        assert second_call.kwargs["api_key"] == "test-key"
        assert second_call.kwargs["http_client"] == mock_http_client

        # Verify httpx.Client was created with verify=False
        mock_httpx_client.assert_called_once_with(verify=False)

        assert client == mock_client_instance_2
