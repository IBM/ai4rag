# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import os
from typing import Any

import boto3

_REQUIRED_ENV_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT")


def get_s3_credentials_from_env() -> dict[str, str | None]:
    """Read S3-compatible credentials from environment variables.

    Returns a dict with keys ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    ``AWS_S3_ENDPOINT``, and (optional) ``AWS_DEFAULT_REGION``.

    Raises
    ------
    ValueError
        If any of the three required variables is missing.
    """
    creds: dict[str, str | None] = {k: os.environ.get(k) for k in _REQUIRED_ENV_KEYS}
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise ValueError(
            f"Missing environment variable(s): {missing}. " "Check that the Kubernetes secret is configured properly."
        )
    creds["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION")
    return creds


def create_s3_client(
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region_name: str | None = None,
    verify: bool = True,
) -> Any:
    """Create an S3-compatible ``boto3`` client.

    When called without arguments, credentials are read from the standard
    ``AWS_*`` environment variables via :func:`get_s3_credentials_from_env`.

    Parameters
    ----------
    endpoint_url
        S3-compatible endpoint URL.  Falls back to ``AWS_S3_ENDPOINT``.
    access_key_id
        AWS access key.  Falls back to ``AWS_ACCESS_KEY_ID``.
    secret_access_key
        AWS secret key.  Falls back to ``AWS_SECRET_ACCESS_KEY``.
    region_name
        AWS region.  Falls back to ``AWS_DEFAULT_REGION``.
    verify
        Whether to verify TLS certificates.

    Returns
    -------
    boto3.client
        A configured S3 client.
    """
    if endpoint_url is None:
        env = get_s3_credentials_from_env()
        endpoint_url = env["AWS_S3_ENDPOINT"]
        access_key_id = access_key_id or env["AWS_ACCESS_KEY_ID"]
        secret_access_key = secret_access_key or env["AWS_SECRET_ACCESS_KEY"]
        region_name = region_name or env.get("AWS_DEFAULT_REGION")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region_name,
        verify=verify,
    )
