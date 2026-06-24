# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag.components.data.documents_discovery import DiscoveryResult, DocumentDescriptor, discover_documents
from ai4rag.components.data.documents_indexing import index_documents
from ai4rag.components.data.test_data_loader import TestDataLoaderError, TestDataResult, load_test_data
from ai4rag.components.data.text_extraction import ExtractionResult, extract_text

__all__ = [
    "discover_documents",
    "DiscoveryResult",
    "DocumentDescriptor",
    "extract_text",
    "ExtractionResult",
    "index_documents",
    "load_test_data",
    "TestDataLoaderError",
    "TestDataResult",
]
