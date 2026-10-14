# Data Components

Data processing functions for the AutoRAG pipeline.

## Discovery

::: ai4rag.utils.data.documents_discovery
    options:
      members:
        - discover_documents
        - DiscoveryResult
        - DocumentDescriptor
        - BenchmarkKeyError

## Text Extraction

PDF extraction uses Docling's `PyPdfiumDocumentBackend` to preserve text that the
native parser can emit as NUL characters. OCR and table processing remain controlled
by `DoclingExtractionConfig`.

::: ai4rag.utils.data.text_extraction
    options:
      members:
        - extract_text
        - DoclingExtractionConfig
        - ExtractionResult

## Local Whisper ASR

Audio extraction uses a local Transformers-format Whisper model configured through
`DoclingExtractionConfig.asr_model_path` or `HF_MODEL_DIR`. See
[Audio Transcription](../../user-guide/pipeline-components.md#audio-transcription)
for the required artifacts and migration from openai-whisper checkpoints.

::: ai4rag.utils.data.local_hf_asr_pipeline
    options:
      members:
        - LocalHuggingFaceAsrPipelineOptions
        - LocalHuggingFaceAsrPipeline

## Test Data Loading

::: ai4rag.utils.data.test_data_loader
    options:
      members:
        - load_test_data
        - TestDataResult
        - TestDataLoaderError
