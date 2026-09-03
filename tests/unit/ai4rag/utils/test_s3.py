# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.utils.clients.s3 import create_s3_client, get_s3_credentials_from_env


class TestGetS3CredentialsFromEnv:
    """Test suite for :func:`get_s3_credentials_from_env`."""

    def test_returns_credentials_when_all_env_vars_set(self, monkeypatch):
        """All required variables present -- should return a dict with four keys."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://s3.example.com")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        creds = get_s3_credentials_from_env()

        assert creds["AWS_ACCESS_KEY_ID"] == "AKIAIOSFODNN7EXAMPLE"
        assert creds["AWS_SECRET_ACCESS_KEY"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert creds["AWS_S3_ENDPOINT"] == "https://s3.example.com"
        assert creds["AWS_DEFAULT_REGION"] == "us-east-1"

    def test_returns_none_for_optional_region_when_unset(self, monkeypatch):
        """``AWS_DEFAULT_REGION`` is optional -- ``None`` when absent."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key-id")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://s3.example.com")
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

        creds = get_s3_credentials_from_env()

        assert creds["AWS_DEFAULT_REGION"] is None

    @pytest.mark.parametrize(
        "missing_var",
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"],
    )
    def test_raises_value_error_when_required_var_missing(self, monkeypatch, missing_var):
        """A ``ValueError`` must be raised when any required env var is absent."""
        all_vars = {
            "AWS_ACCESS_KEY_ID": "key-id",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
        }
        for k, v in all_vars.items():
            if k == missing_var:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)

        with pytest.raises(ValueError, match=missing_var):
            get_s3_credentials_from_env()

    def test_raises_value_error_when_multiple_vars_missing(self, monkeypatch):
        """Error message should list all missing variables."""
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)

        with pytest.raises(ValueError, match="Missing environment variable"):
            get_s3_credentials_from_env()


class TestCreateS3Client:
    """Test suite for :func:`create_s3_client`."""

    def test_creates_client_with_explicit_args(self, mocker):
        """Explicit arguments bypass environment variable lookup."""
        mock_client = mocker.MagicMock()
        mock_boto3 = mocker.patch("ai4rag.utils.clients.s3.boto3")
        mock_boto3.client.return_value = mock_client

        result = create_s3_client(
            endpoint_url="https://s3.example.com",
            access_key_id="key-id",
            secret_access_key="secret",
            region_name="us-west-2",
            verify=False,
        )

        assert result is mock_client
        mock_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="https://s3.example.com",
            aws_access_key_id="key-id",
            aws_secret_access_key="secret",
            region_name="us-west-2",
            verify=False,
        )

    def test_falls_back_to_env_when_endpoint_url_is_none(self, mocker, monkeypatch):
        """When ``endpoint_url`` is ``None``, credentials are read from the environment."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://env-s3.example.com")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

        mock_client = mocker.MagicMock()
        mock_boto3 = mocker.patch("ai4rag.utils.clients.s3.boto3")
        mock_boto3.client.return_value = mock_client

        result = create_s3_client()

        assert result is mock_client
        mock_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="https://env-s3.example.com",
            aws_access_key_id="env-key",
            aws_secret_access_key="env-secret",
            region_name="eu-central-1",
            verify=True,
        )

    def test_explicit_access_key_overrides_env(self, mocker, monkeypatch):
        """Explicitly supplied ``access_key_id`` takes precedence over the env value."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://env-s3.example.com")

        mock_boto3 = mocker.patch("ai4rag.utils.clients.s3.boto3")
        mock_boto3.client.return_value = mocker.MagicMock()

        create_s3_client(access_key_id="explicit-key")

        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["aws_access_key_id"] == "explicit-key"

    def test_verify_defaults_to_true(self, mocker):
        """The ``verify`` parameter should default to ``True``."""
        mock_boto3 = mocker.patch("ai4rag.utils.clients.s3.boto3")
        mock_boto3.client.return_value = mocker.MagicMock()

        create_s3_client(endpoint_url="https://s3.example.com")

        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["verify"] is True
