# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
import ssl

import httpx
from ogx_client import APIConnectionError as OGXAPIConnectionError
from ogx_client import OgxClient

from ai4rag import handler

_logger = logging.getLogger("ogx-client")
_logger.addHandler(handler)


def is_ssl_error(exc: BaseException) -> bool:
    """Check whether an exception (or its cause/context chain) contains an SSL verification failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).upper()
        if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg:
            return True
        current = current.__cause__ or current.__context__
    return False


def create_ogx_client(base_url: str, api_key: str) -> OgxClient:
    """Create an :class:`OgxClient`, falling back to unverified TLS on self-signed certificates.

    The function first creates a client with default TLS settings and
    issues a lightweight request (``models.list()``) to probe connectivity.
    If that request fails with an SSL verification error, the client is
    re-created with ``verify=False`` and a warning is logged.

    Parameters
    ----------
    base_url
        URL of the OGX server.
    api_key
        API key for authentication.

    Returns
    -------
    OgxClient
        A connected client instance.
    """
    client = OgxClient(base_url=base_url, api_key=api_key)
    try:
        client.models.list()
    except (ssl.SSLCertVerificationError, httpx.ConnectError, OGXAPIConnectionError) as exc:
        if is_ssl_error(exc):
            _logger.warning("SSL verification failed for OgxClient — retrying with verify=False.")
            client = OgxClient(base_url=base_url, api_key=api_key, http_client=httpx.Client(verify=False))
        else:
            raise
    return client
