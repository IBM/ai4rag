# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
import ssl

import httpx
from openai import APIConnectionError, OpenAI

from ai4rag import handler

_logger = logging.getLogger("maas-client")
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


def create_maas_client(base_url: str, api_key: str) -> OpenAI:
    """Create the MaaS client, falling back to unverified TLS on self-signed certs.

    A single client serves everything: it lists models via ``models.list()`` and
    serves ``chat.completions`` and ``embeddings`` for every model at the same
    OpenAI-compatible endpoint. The function first creates a client with default
    TLS settings and issues a lightweight ``models.list()`` request to probe
    connectivity. If that fails with an SSL verification error, the client is
    re-created with ``verify=False`` and a warning is logged.

    Parameters
    ----------
    base_url
        Complete OpenAI-compatible MaaS endpoint URL, used verbatim
        (e.g. ``https://<host>/v1``).
    api_key
        API key for authentication.

    Returns
    -------
    OpenAI
        A connected client instance.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        client.models.list()
    except (ssl.SSLCertVerificationError, httpx.ConnectError, APIConnectionError) as exc:
        if is_ssl_error(exc):
            _logger.warning("SSL verification failed for MaaS client — retrying with verify=False.")
            client = OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(verify=False))
        else:
            raise
    return client
