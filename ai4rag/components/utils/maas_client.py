# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
import ssl
from urllib.parse import urlsplit

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


def maas_model_base_url(general_base_url: object, owned_by: str) -> str:
    """Derive a per-model MaaS base URL from the general (list) endpoint.

    MaaS exposes one endpoint per model at ``{scheme}://{netloc}/{owned_by}/v1``.
    The ``owned_by`` value carried by each listed model (e.g.
    ``ai-eng-cracow/qwen3-8b-fp8-dynamic``) is the path prefix for that model's
    OpenAI-compatible endpoint.  The scheme and host are taken from the general
    base URL; any path it carries (e.g. ``/maas-api/v1``) is discarded.

    Parameters
    ----------
    general_base_url : object
        Base URL of the general MaaS client (plain string or ``AnyUrl``).
    owned_by : str
        ``owned_by`` attribute of a listed model.

    Returns
    -------
    str
        Per-model base URL with the required ``/v1`` suffix.
    """
    parts = urlsplit(str(general_base_url))
    return f"{parts.scheme}://{parts.netloc}/{owned_by.strip('/')}/v1"


def create_maas_client(base_url: str, api_key: str) -> OpenAI:
    """Create the general MaaS client, falling back to unverified TLS on self-signed certs.

    The general client points at the ``/maas-api/v1`` endpoint and is used to
    list available models. The function first creates a client with default TLS
    settings and issues a lightweight ``models.list()`` request to probe
    connectivity. If that fails with an SSL verification error, the client is
    re-created with ``verify=False`` and a warning is logged.

    Parameters
    ----------
    base_url
        URL of the general MaaS endpoint (e.g. ``https://<host>/maas-api/v1``).
    api_key
        API key for authentication (reused for every per-model client).

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


def create_maas_model_client(base_url: str, api_key: str) -> OpenAI:
    """Create a per-model MaaS client for a single model endpoint.

    Unlike :func:`create_maas_client`, this does not probe with ``models.list()``:
    per-model endpoints do not serve the model list, so connectivity is instead
    surfaced by the downstream chat/embedding validation step.

    Parameters
    ----------
    base_url
        Per-model base URL, typically from :func:`maas_model_base_url`.
    api_key
        API key for authentication.

    Returns
    -------
    OpenAI
        A per-model client instance.
    """
    return OpenAI(base_url=base_url, api_key=api_key)
