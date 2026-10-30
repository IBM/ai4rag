# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Resolve MaaS certificate trust without changing process-wide TLS settings."""

import os
import ssl
from pathlib import Path

import certifi


def get_maas_tls_verify(ca_bundle: str | None = None) -> ssl.SSLContext | str | None:
    """Resolve explicit trust or add PEM bundles mounted in ``SSL_CERT_DIR``.

    Explicit ``ca_bundle`` and ``MAAS_CA_BUNDLE`` retain their existing precedence.
    With ``SSL_CERT_FILE`` set, return ``None`` so HTTPX retains its environment
    handling. Otherwise, look for ``ca.crt`` in the directories already selected
    by the deployment through ``SSL_CERT_DIR``. DSP mounts its trusted bundle in
    this form, which is not an OpenSSL hashed certificate directory by itself.

    Mounted bundles supplement certifi's public roots and the configured hashed
    directories. Missing bundles leave HTTPX's defaults unchanged. Invalid or
    unreadable bundles raise instead of disabling certificate verification.

    Parameters
    ----------
    ca_bundle
        Explicit PEM bundle path, taking precedence over environment settings.

    Returns
    -------
    ssl.SSLContext | str | None
        HTTPX ``verify`` configuration, or ``None`` to retain its default behavior.
    """
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
    for directory in directories:
        context.load_verify_locations(capath=str(directory))
    for bundle in bundles:
        context.load_verify_locations(cafile=str(bundle))
    return context
