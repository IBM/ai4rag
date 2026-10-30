# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Clients for the external services the pipeline talks to.

This package groups the thin adapters over out-of-process dependencies --
object storage (:mod:`~ai4rag.utils.clients.s3`) and the OpenAI-compatible
model-serving endpoint (:mod:`~ai4rag.utils.clients.maas_client`).  It is a
sibling of :mod:`ai4rag.utils.data`, so data-processing code imports it
laterally instead of reaching up into the ``ai4rag.utils`` namespace.
"""

from ai4rag.utils.clients.maas_client import create_maas_client, is_ssl_error
from ai4rag.utils.clients.s3 import create_s3_client, get_s3_credentials_from_env

__all__ = [
    "create_maas_client",
    "create_s3_client",
    "get_s3_credentials_from_env",
    "is_ssl_error",
]
