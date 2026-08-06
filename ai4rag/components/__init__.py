# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag.components.utils import (
    create_maas_client,
    create_maas_model_client,
    create_s3_client,
    get_s3_credentials_from_env,
    load_docling_documents,
    maas_model_base_url,
)

# The public re-export surface here intentionally mirrors a curated subset of
# `ai4rag.components.utils.__all__`; the resulting overlap is by design, not
# accidental copy-paste, so the duplicate-code heuristic is a false positive here.
# pylint: disable=duplicate-code
__all__ = [
    "create_maas_client",
    "create_maas_model_client",
    "create_s3_client",
    "get_s3_credentials_from_env",
    "load_docling_documents",
    "maas_model_base_url",
]
