# Search Space API

## AI4RAGSearchSpace

::: ai4rag.search_space.src.search_space.AI4RAGSearchSpace
    options:
      show_root_heading: true
      show_source: true

## Parameter

::: ai4rag.search_space.src.parameter.Parameter
    options:
      show_root_heading: true
      show_source: true

## Search Space Preparation

::: ai4rag.search_space.prepare.prepare_search_space.prepare_search_space_with_maas
    options:
      show_root_heading: true
      show_source: true

## Search Space Report

The file-format contract exchanged between the search-space preparation,
model pre-selection, and optimization steps.

::: ai4rag.search_space.prepare.report
    options:
      members:
        - build_search_space_report
        - SearchSpaceReport

## Model Instantiation & Serialization

::: ai4rag.search_space.prepare.models
    options:
      members:
        - get_foundation_models
        - get_embedding_models
        - serialize_model
