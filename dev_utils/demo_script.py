# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage, SystemMessage

base_url = os.getenv("REMOTE_BASE_URL", "http://localhost:8321")

client = LlamaStackClient(base_url=base_url)

example_question = "What is the meaning of life?"

EMBEDDING_MODEL = "ibm/slate-125m-english-rtrvr-v2"
EMBEDDING_MODEL_DIMENSION = 768
LLM_MODEL = "meta-llama/llama-3-3-70b-instruct"

DEFAULT_PROMPT_TEMPLATE = "{reference_documents}\n[conversation]: {question}. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer. Respond exclusively in the language of the question, regardless of any other language used in the provided context. Ensure that your entire response is in the same language as the question.\n"
DEFAULT_CONTEXT_TEMPLATE = "[document]: {document}\n"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n"


def build_prompt(
        question: str,
        reference_documents: list[str] = None,
        prompt_template_text: str = DEFAULT_PROMPT_TEMPLATE,
        context_template_text: str = DEFAULT_CONTEXT_TEMPLATE,
) -> str:
    """
    Warning: It's simplified prompt builder, without sampling of the reference documents
    """
    if reference_documents:
        reference_documents = [
            context_template_text.format(document=reference_document)
            for reference_document in reference_documents
        ]
    else:
        reference_documents = []
    prompt_variables = {
        "question": question,
        "reference_documents": "\n".join(reference_documents)
    }
    return prompt_template_text.format(**prompt_variables)


sample_prompt = build_prompt(question=example_question)

response_chat = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[SystemMessage(role="system", content=DEFAULT_SYSTEM_PROMPT), UserMessage(role="user", content=sample_prompt)]
)
answer = response_chat.choices[0].message.content

print(answer)
