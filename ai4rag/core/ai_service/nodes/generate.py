# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from langgraph.types import StreamWriter

from ai4rag.core.ai_service.states import AI4RAGState

# pylint: disable=stop-iteration-return
# pylint: disable=undefined-variable


def chat_node(state: AI4RAGState, writer: StreamWriter) -> dict:
    """
    Node responsible for generating answer based on the retrieved context.

    Parameters
    ----------
    state : AI4RAGState
        The current execution state, including the question and retrieved context needed for generation.

    writer : StreamWriter
        Stream writer for emitting intermediate updates during execution.

    Returns
    -------
    dict
        Response for the user's question.
    """
    prompt = build_prompt(
        question=state["question"],
        reference_documents=[doc.page_content for doc in state["retrieved_documents"]],
        **build_prompt_additional_kwargs,
    )
    response_stream = model.chat_stream(
        messages=[
            {"role": "system", "content": build_prompt_additional_kwargs.get("system_message_text")},
            {"role": "user", "content": prompt},
        ]
    )
    chunk = next(response_stream)
    answer = chunk["choices"][0]["delta"]["content"]
    writer(
        {
            "chunk_content": chunk["choices"][0]["delta"],
            "reference_documents": state["retrieved_documents"],
            "finish_reason": chunk["choices"][0]["finish_reason"],
        }
    )
    for chunk in response_stream:
        if (
            chunk.get("choices")
            and chunk["choices"]
            and chunk["choices"][0].get("delta", {}).get("content")
            and ("finish_reason" in chunk["choices"][0])
        ):
            writer(
                {"chunk_content": chunk["choices"][0]["delta"], "finish_reason": chunk["choices"][0]["finish_reason"]}
            )
            answer += chunk["choices"][0]["delta"]["content"]

    return {"response": answer}
