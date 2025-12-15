# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# pylint: skip-file


def inference_service(context, url=None, vector_store_settings=None):  # pragma: no cover
    """
    Default inference AI service function.

    :param vector_store_settings: Vector store settings, such as `connection_id`, `index_name` and scope identifier: space_id/project_id,
                                 encapsulated in a dictionary. Setting these parameters when creating deployment one can overwrite the defaults.
    :type vector_store_settings: dict, optional

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials

        client = APIClient(
            credentials=Credentials(url="<url>", token="<token>"), space_id=space_id
        )

        meta_props = {
            client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT NAME",
            client.deployments.ConfigurationMetaNames.ONLINE: {
                "parameters": {
                    "vector_store_settings": {
                        "connection_id": "<connection_to_vector_store>",
                        "index_name": "<index_name>",
                        "project_id": "<project_id>",
                    }
                }
            },
        }

        deployment_details = client.deployments.create(ai_service_id, meta_props)

    Input schema:
    payload = {
       "messages":[
            {
                "role" : "user",
                "content" : "question_1"
            }
        ]
    }

    Output schema:
    result = {
        'choices': [
            {
                'index': 0,
                'message': {
                    'content': 'generated_content',
                    'role': 'assistant'
                },
                "reference_documents" : [
                            {
                                'sequence_number': [1, 2, 3],
                                'document_id': '<document_id>'
                            }
                        ]
            }
        ]
    }
    """
    from typing import TypedDict

    from langgraph.graph import StateGraph
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.extensions.rag import Retriever
    from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
        REPLACE_THIS_CODE_WITH_VECTOR_STORE_CLASS_NAME,
    )
    from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.prompt_builder import build_prompt
    from langgraph.types import StreamWriter

    REPLACE_THIS_CODE_WITH_INDEXING_IMPORTS

    vector_store_settings = dict(vector_store_settings) if vector_store_settings is not None else {}

    default_url = REPLACE_THIS_CODE_WITH_CREDENTIALS_URL

    client = APIClient(
        credentials=Credentials(
            url=url or default_url,
            token=context.generate_token(),
            name=REPLACE_THIS_CODE_WITH_CREDENTIALS_NAME,
            iam_serviceid_crn=REPLACE_THIS_CODE_WITH_CREDENTIALS_IAM_SERVICEID_CRN,
            projects_token=REPLACE_THIS_CODE_WITH_CREDENTIALS_PROJECTS_TOKEN,
            username=REPLACE_THIS_CODE_WITH_CREDENTIALS_USERNAME,
            instance_id=REPLACE_THIS_CODE_WITH_CREDENTIALS_INSTANCE_ID,
            version=REPLACE_THIS_CODE_WITH_CREDENTIALS_VERSION,
            bedrock_url=REPLACE_THIS_CODE_WITH_CREDENTIALS_BEDROCK_URL,
            platform_url=REPLACE_THIS_CODE_WITH_CREDENTIALS_PLATFORM_URL,
            proxies=REPLACE_THIS_CODE_WITH_CREDENTIALS_PROXIES,
            verify=REPLACE_THIS_CODE_WITH_CREDENTIALS_VERIFY,
        ),
        space_id=vector_store_settings.pop("space_id", None),
        project_id=vector_store_settings.pop("project_id", None),
    )

    vector_store_init_data = REPLACE_THIS_CODE_WITH_VECTOR_STORE_ASSIGN

    # update vector store init data
    vector_store_init_data |= vector_store_settings

    REPLACE_THIS_CODE_WITH_VECTOR_STORE_INITIALIZATION

    REPLACE_THIS_CODE_WITH_INDEXING_CODE

    retriever = Retriever.from_vector_store(
        vector_store=vector_store,
        init_parameters=REPLACE_THIS_CODE_WITH_RETRIEVER,
    )
    retrieve_params = REPLACE_THIS_CODE_WITH_RETRIEVE_PARAMS
    model = ModelInference(
        api_client=client,
        model_id=REPLACE_THIS_CODE_WITH_MODEL_MODEL_ID,
        deployment_id=REPLACE_THIS_CODE_WITH_MODEL_DEPLOYMENT_ID,
        params=REPLACE_THIS_CODE_WITH_MODEL_PARAMS,
        validate=False,
    )
    build_prompt_additional_kwargs = REPLACE_THIS_CODE_WITH_BUILD_PROMPT_KWARGS

    word_to_token_ratio = REPLACE_THIS_CODE_WITH_WORD_TO_TOKEN_RATIO
    if word_to_token_ratio is not None:
        build_prompt_additional_kwargs["word_to_token_ratio"] = word_to_token_ratio

    REPLACE_THIS_CODE_WITH_AUTOAIRAG_STATE

    REPLACE_THIS_CODE_WITH_RETRIEVE_NODE

    REPLACE_THIS_CODE_WITH_GENERATE_NODE_SOURCE

    graph = (
        StateGraph(AI4RAGState)
        .add_node("retrieve", REPLACE_THIS_CODE_WITH_RETRIEVE_NODE_NAME)
        .add_node("generate", REPLACE_THIS_CODE_WITH_GENERATE_NODE_NAME)
        .add_edge("retrieve", "generate")
        .set_entry_point("retrieve")
        .compile()
    )

    def _validate_messages(messages: list[dict]):
        if messages and isinstance(messages, (list, tuple)) and messages[-1]["role"] == "user":
            return None

        raise ValueError(
            "The `messages` field must be an array containing objects, where the last one is representing user's message."
        )

    def generate(context):
        """
        The `generate` function handles the REST call to the inference endpoint
        POST /ml/v4/deployments/{id_or_name}/ai_service

        A JSON body sent to the above endpoint should follow the format:
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<user_query>",
                },
            ]
        }
        """
        client.set_token(context.get_token())

        messages = context.get_json()["messages"]
        _validate_messages(messages=messages)
        question = messages[-1]["content"]

        start_state = AI4RAGState(question=question)
        final_state = graph.invoke(start_state)

        return {
            "body": {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "system",
                            "content": final_state["response"],
                        },
                        "reference_documents": [
                            {"page_content": doc.page_content, "metadata": doc.metadata}
                            for doc in final_state["retrieved_documents"]
                        ],
                    }
                ]
            }
        }

    def generate_stream(context):
        """
        The `generate_stream` function handles the REST call to the Server-Sent Events (SSE) inference endpoint
        POST /ml/v4/deployments/{id_or_name}/ai_service_stream

        A JSON body sent to the above endpoint should follow the format:
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<user_query>",
                },
            ]
        }
        """
        client.set_token(context.get_token())

        messages = context.get_json()["messages"]
        _validate_messages(messages=messages)
        question = messages[-1]["content"]

        start_state = AI4RAGState(question=question)

        stream = graph.stream(start_state, stream_mode="custom")
        chunk = next(stream)
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": chunk["chunk_content"],
                    "reference_documents": [
                        {"page_content": doc.page_content, "metadata": doc.metadata}
                        for doc in chunk["reference_documents"]
                    ],
                    "finish_reason": chunk["finish_reason"],
                }
            ]
        }
        for chunk in stream:
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": chunk["chunk_content"],
                        "finish_reason": chunk["finish_reason"],
                    }
                ]
            }

    return (generate, generate_stream)
