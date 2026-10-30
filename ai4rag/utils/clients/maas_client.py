# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
import os
import ssl
from pathlib import Path

import certifi
import httpx
from openai import APIConnectionError, OpenAI

from ai4rag import handler
from ai4rag.utils.network import ensure_safe_url

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


def _get_maas_tls_verify(ca_bundle: str | None) -> ssl.SSLContext | str | None:
    """Return explicit CA settings, else certifi plus ``SSL_CERT_DIR``/``ca.crt`` if mounted."""
    explicit_bundle = ca_bundle or os.environ.get("MAAS_CA_BUNDLE")
    if explicit_bundle:
        return explicit_bundle
    if os.environ.get("SSL_CERT_FILE"):
        return None

    directories = [Path(value) for value in os.environ.get("SSL_CERT_DIR", "").split(os.pathsep) if value]
    bundles = [directory / "ca.crt" for directory in directories if (directory / "ca.crt").is_file()]
    if not bundles:
        return None

    context = ssl.create_default_context(cafile=certifi.where())
    # Keep hashed-directory trust too: loading plain PEMs must only add roots.
    for directory in directories:
        context.load_verify_locations(capath=str(directory))
    for bundle in bundles:
        context.load_verify_locations(cafile=str(bundle))
    return context


def create_maas_client(base_url: str, api_key: str, ca_bundle: str | None = None) -> OpenAI:
    """Create the MaaS client with certificate verification always enforced.

    A single client serves everything: it lists models via ``models.list()`` and
    serves ``chat.completions`` and ``embeddings`` for every model at the same
    OpenAI-compatible endpoint. ``base_url`` must use ``https://``, unless it
    targets ``localhost``, ``127.0.0.1``, ``::1``, or a ``*.cluster.local``
    in-cluster service (see :func:`ai4rag.utils.network.ensure_safe_url`). ai4rag
    never falls back to unverified TLS: if the endpoint presents a certificate
    signed by a private or self-signed CA, pass *ca_bundle* (or set the
    ``MAAS_CA_BUNDLE`` environment variable) with that CA's PEM bundle path.
    If neither is set and ``SSL_CERT_FILE`` is unset, a mounted
    ``SSL_CERT_DIR``/``ca.crt`` is added to public trust when present.

    Parameters
    ----------
    base_url
        OpenAI-compatible MaaS endpoint URL (e.g. ``https://<host>``). The
        ``/v1`` path segment is appended automatically if not already present.
    api_key
        API key for authentication.
    ca_bundle
        Path to a PEM CA bundle used to verify *base_url*'s certificate. Falls
        back to the ``MAAS_CA_BUNDLE`` environment variable; when neither is
        set, certifi's public roots are used (plus any mounted
        ``SSL_CERT_DIR``/``ca.crt`` fallback).

    Returns
    -------
    OpenAI
        A connected client instance.

    Raises
    ------
    ValueError
        If *base_url* uses plaintext ``http://`` against a non-local,
        non-cluster host.
    """
    ensure_safe_url(base_url, context="MaaS base_url")

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    verify = _get_maas_tls_verify(ca_bundle)
    client_kwargs: dict = {"base_url": base_url, "api_key": api_key}
    if verify is not None:
        client_kwargs["http_client"] = httpx.Client(verify=verify)

    client = OpenAI(**client_kwargs)
    if verify is not None:
        # The agent reuses this client for sync calls and shares its trust with async calls.
        setattr(client, "_ai4rag_tls_verify", verify)
    try:
        client.models.list()
    except (ssl.SSLCertVerificationError, httpx.ConnectError, APIConnectionError) as exc:
        if is_ssl_error(exc):
            _logger.error(
                "TLS certificate verification failed for MaaS endpoint %r. If this endpoint uses a "
                "private or self-signed CA, set MAAS_CA_BUNDLE (or pass ca_bundle=...) to that CA's "
                "PEM file. ai4rag does not fall back to unverified TLS.",
                base_url,
            )
        raise
    return client
