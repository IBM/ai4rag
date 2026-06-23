# Data Components

Data processing functions for the AutoRAG pipeline.

## Discovery

::: ai4rag.components.data.documents_discovery
    options:
      members:
        - discover_documents
        - DiscoveryResult
        - DocumentDescriptor

## Text Extraction

::: ai4rag.components.data.text_extraction
    options:
      members:
        - extract_text
        - ExtractionResult

## Document Indexing

::: ai4rag.components.data.documents_indexing
    options:
      members:
        - index_documents

## Test Data Loading

::: ai4rag.components.data.test_data_loader
    options:
      members:
        - load_test_data
        - TestDataResult
        - TestDataLoaderError
