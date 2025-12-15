# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Literal

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import VectorStore
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.adapters.es_adapter import (
    ElasticsearchVectorStore,
)

from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.adapters.milvus_adapter import MilvusVectorStore
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.vector_store_connector import (
    VectorStoreDataSourceType,
)

from ai4rag import logger
from ai4rag.utils.constants import DefaultVectorStoreFieldNames

__all__ = [
    "get_vector_store",
]

# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
def get_vector_store(
    vs_type: Literal["milvus", "chroma", "local_milvus", "elasticsearch", "local_elasticsearch"],
    embeddings: BaseEmbeddings,
    distance_metric: Literal["euclidean", "cosine"],
    index_name: str,
    connection_id: str | None = None,
    api_client: APIClient | None = None,
    text: str | None = None,
    chunk_sequence_number: str = DefaultVectorStoreFieldNames.CHUNK_SEQUENCE_NUMBER_FIELD,
    document_name: str = DefaultVectorStoreFieldNames.METADATA_DOCUMENT_NAME_FIELD,
) -> VectorStore:
    """
    Helper function that is responsible for creating VectorStore instance that
    will be used within AutoRAG process.

    Parameters
    ----------
    vs_type : Literal["milvus", "chroma", "local_milvus", "elasticsearch", "local_elasticsearch"]
        Specified type of Vector Store that should be one of the above.

    embeddings : BaseEmbeddings
        Dense embedding model implementing the ibm-watsonx-ai.Embeddings interface.

    distance_metric : Literal["euclidean", "cosine"]
        Metric used to calculate similarity between context and query
        in the vector store.

    index_name : str | None
        Name of the given collection in the vector database

    text : str
        Field name inside vector store where chunked text is stored.

    connection_id : None | str, default=None
        ID of the connection used to create instance of VectorStore.

    api_client : APIClient | None, default=None
        Instance of APIClient.

    dense_vector_embeddings : str, default="vector"
        Name of the field in the vector store collection which stores embeddings.

    sparse_vector_embeddings : str, default="sparse_embeddings"
        Name of the field in the vector store collection which stores sparse vectors.

    chunk_sequence_number: str, default=DefaultVectorStoreFieldNames.CHUNK_SEQUENCE_NUMBER_FIELD,
        Custom chunk_sequence_number field name.

    document_name: str, default=DefaultVectorStoreFieldNames.METADATA_DOCUMENT_NAME_FIELD,
        Custom document_name field name.

    hybrid_search_ranker_settings : RRFRankerParamsType | WeightedRankerParamsType, default=None
            Dicts containing configurations for each ranker type.
            If None is provided it means that hybrid search is turned off.

    Returns
    -------
    VectorStore
        Instance that can be consumed by the AutoRAG experiment and contains
        all logic responsible for communicating wth Vector DB.

    Raises
    ------
    ValueError
        Raised when:
          - incorrect vs_type is provided
          - unsupported sparse_vectors value is provided with relation to the requested vs_type


    """
    match vs_type:
        case "local_milvus":

            vector_store = VectorStore(
                datasource_type=VectorStoreDataSourceType.MILVUS,
                embeddings=embeddings,
                index_name=index_name,
                distance_metric=distance_metric,
                connection_args={"address": "localhost:19530"},
                drop_old=False,
                auto_id=True,
            )

        case "local_elasticsearch":

            vector_store = VectorStore(
                datasource_type=VectorStoreDataSourceType.ELASTICSEARCH,
                embeddings=embeddings,
                index_name=index_name,
                distance_metric=distance_metric,
                es_url="http://localhost:9200",
                es_user="",
                es_password="",
            )

        case "milvus":

            if distance_metric == "cosine":
                distance_metric = distance_metric.upper()
            elif distance_metric == "euclidean":
                distance_metric = "L2"

            vector_store = MilvusVectorStore(
                api_client=api_client,
                connection_id=connection_id,
                embedding_function=embeddings,
                collection_name=index_name,
                distance_metric=distance_metric,
                secure=True,
                vector_field=dense_vector_embeddings,
                text_field=text,
                document_name_field=document_name,
                chunk_sequence_number_field=chunk_sequence_number,
            )

        case "chroma":

            vector_store = VectorStore(
                datasource_type=VectorStoreDataSourceType.CHROMA,
                embeddings=embeddings,
                distance_metric=distance_metric,
                index_name=index_name,
            )

        case "elasticsearch":

            # there are differences in default strategy between `VectorStore` and `ElasticsearchVectorStore`
            # For now we want to keep it would require refactor of RAGService not to use to_dict() methods.
            es_distance_metric = distance_metric
            if distance_metric == "cosine":
                es_distance_metric = DistanceStrategy.COSINE.value
            elif distance_metric == "euclidean":
                es_distance_metric = DistanceStrategy.EUCLIDEAN_DISTANCE.value

            vector_store = ElasticsearchVectorStore(
                connection_id=connection_id,
                embedding=embeddings,
                index_name=index_name,
                distance_strategy=es_distance_metric,
                api_client=api_client,
                query_field=text,
                vector_query_field=dense_vector_embeddings,
                document_name_field=document_name,
                chunk_sequence_number_field=chunk_sequence_number,
            )

        case _:
            raise ValueError(f"VectorStore type {vs_type} is not supported.")

    logger.info(
        "Created VectorStore of type '%s', with index_name='%s' and distance_metric='%s'.",
        vs_type,
        index_name,
        distance_metric,
    )

    return vector_store
