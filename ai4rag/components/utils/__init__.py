# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag.components.utils.docling_io import load_docling_documents
from ai4rag.components.utils.maas_client import (
    create_maas_client,
    create_maas_model_client,
    is_ssl_error,
    maas_model_base_url,
)
from ai4rag.components.utils.s3 import create_s3_client, get_s3_credentials_from_env

# Package-level re-export surface; `ai4rag.components.__init__` deliberately
# mirrors a curated subset of these names, so the duplicate-code heuristic flags
# an intentional, by-design overlap here as a false positive.
# pylint: disable=duplicate-code
__all__ = [
    "create_maas_client",
    "create_maas_model_client",
    "create_s3_client",
    "get_s3_credentials_from_env",
    "is_ssl_error",
    "load_docling_documents",
    "maas_model_base_url",
]
