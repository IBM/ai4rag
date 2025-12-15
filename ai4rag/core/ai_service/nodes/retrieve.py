from ai4rag.core.ai_service.states import AI4RAGState


# pylint: disable=undefined-variable
def retrieve_node(state: AI4RAGState) -> dict:
    """
    Node responsible for retrieving relevant context from the vector store.

    Parameters
    ----------
    state : AI4RAGState
        The current execution state, including the question to retrieve the context for.

    Returns
    -------
    list
        Context retrieved from the vector store.
    """
    question = state["question"]
    retrieved_documents = retriever.retrieve(query=question, **retrieve_params)
    return {"retrieved_documents": retrieved_documents}


def multi_index_retrieve_node(state: AI4RAGState) -> dict:
    """
    Node responsible for retrieving relevant context from the multiple vector stores indexes.

    Parameters
    ----------
    state : AI4RAGState
        The current execution state, including the question to retrieve the context for.

    Returns
    -------
    list
        Context retrieved from the vector stores indexes.
    """
    query = state["question"]
    retrieved_context = []
    for i, retriever in enumerate(retrievers):
        retrieve_args = {
            "query": query,
            **retrieve_params[i],
        }
        retrieved_context.extend(retriever.retrieve(**retrieve_args, include_scores=True))

    sorted_context = list(sorted(retrieved_context, key=lambda chunk: chunk[1]))
    retrieved_chunks = sorted_context[:number_of_retrieved_chunks]
    custom_document_name_fields = [
        custom_field
        for vector_store_data in vector_store_init_data
        if (custom_field := vector_store_data["document_name_field"]) != "document_id"
    ]

    if custom_document_name_fields:
        for retrieved_chunk in retrieved_chunks:
            if retrieved_chunk[0].metadata.get("document_id"):
                continue
            for custom_field_name in custom_document_name_fields:
                if retrieved_chunk[0].metadata.get(custom_field_name):
                    retrieved_chunk[0].metadata["document_id"] = retrieved_chunk[0].metadata.pop(custom_field_name)

    return {"retrieved_documents": [chunk[0] for chunk in retrieved_chunks]}
