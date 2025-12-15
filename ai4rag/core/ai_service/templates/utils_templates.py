# pylint: skip-file


def indexing_recursive_chunking():
    input_data_references = REPLACE_THIS_CODE_WITH_INPUT_DATA_REFERENCES
    dataset = DocumentsIterableDataset(
        connections=input_data_references,
        enable_sampling=False,
        api_client=client,
    )

    chunker = LangChainChunker(
        method=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_METHOD,
        chunk_size=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_CHUNK_SIZE,
        chunk_overlap=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_CHUNK_OVERLAP,
        encoding_name=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_ENCODING_NAME,
        model_name=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_MODEL_NAME,
    )
    documents = chunker.split_documents(dataset)
    vector_store.add_documents(documents)


def indexing_recursive_chunking_imports():
    from ibm_watsonx_ai.data_loaders.datasets.documents import (
        DocumentsIterableDataset,
    )
    from ibm_watsonx_ai.foundation_models.extensions.rag.chunker.langchain_chunker import (
        LangChainChunker,
    )
    from ibm_watsonx_ai.helpers.connections import DataConnection


def indexing_semantic_chunking():
    input_data_references = REPLACE_THIS_CODE_WITH_INPUT_DATA_REFERENCES

    dataset = DocumentsIterableDataset(
        connections=input_data_references,
        enable_sampling=False,
        api_client=client,
    )
    semantic_chunker_embeddings = Embeddings(
        api_client=client,
        model_id=REPLACE_THIS_CODE_WITH_HYBRID_SEMANTIC_CHUNKER_EMBEDDING_MODEL_NAME,
        params=REPLACE_THIS_CODE_WITH_HYBRID_SEMANTIC_CHUNKER_EMBEDDING_MODEL_PARAMS,
    )
    chunker = HybridSemanticChunker(
        chunk_size=REPLACE_THIS_CODE_WITH_HYBRID_SEMANTIC_CHUNKER_CHUNK_SIZE, embeddings=semantic_chunker_embeddings
    )
    documents = chunker.split_documents(dataset)
    vector_store.add_documents(documents)


def indexing_semantic_chunking_imports():
    from ibm_watsonx_ai.foundation_models import Embeddings
    from ibm_watsonx_ai.data_loaders.datasets.documents import (
        DocumentsIterableDataset,
    )
    from ibm_watsonx_ai.foundation_models.extensions.rag.chunker import HybridSemanticChunker
    from ibm_watsonx_ai.helpers.connections import DataConnection


def vector_store_initialization():
    vector_store = VectorStore.from_dict(client=client, data=vector_store_init_data)


def milvus_vector_store_initialization():
    vector_store = MilvusVectorStore.from_dict(api_client=client, data=vector_store_init_data)


def elasticsearch_vector_store_initialization():
    vector_store = ElasticsearchVectorStore.from_dict(api_client=client, data=vector_store_init_data)
