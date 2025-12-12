#
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
#

from ai4rag.search_space.src.models import FoundationModels

__all__ = [
    "get_system_message_text",
    "get_user_message_text",
    "get_context_template_text",
    "QUESTION_PLACEHOLDER",
    "REFERENCE_DOCUMENTS_PLACEHOLDER",
    "CONTEXT_TEXT_PLACEHOLDER",
    "MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER",
]


QUESTION_PLACEHOLDER = "question"
REFERENCE_DOCUMENTS_PLACEHOLDER = "reference_documents"
CONTEXT_TEXT_PLACEHOLDER = "document"
MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER = "multilingual_support"


# A mapping from model name into their corresponding prompt templates.
# The parameters for the prompt templates are QUESTION_PLACEHOLDER and REFERENCE_DOCUMENTS_PLACEHOLDER

_MULTILINGUAL_SUPPORT_ENABLED_PROMPT = (
    "Respond exclusively in the language of the question, "
    "regardless of any other language used in the provided context. "
    "Ensure that your entire response is in the same language as the question."
)


_MULTILINGUAL_SUPPORT_DISABLED_PROMPT = (
    "Respond exclusively in English, "
    "regardless of the language of the question or any other language used in the provided context. "
    "Ensure that your entire response is in English only."
)


_DEFAULT_SYSTEM_MESSAGE_TEXT = (
    "Please answer the question I provide in the Question section below, "
    "based solely on the information I provide in the Context section. "
    "If the question is unanswerable, please say you cannot answer."
)


_DEFAULT_USER_MESSAGE_TEXT = (
    f"\n\nContext:\n{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}:\n\n"
    f"Question: {{{QUESTION_PLACEHOLDER}}}. \n"
    "Again, please answer the question based on the context provided only. If the context is not related to "
    "the question, just say you cannot answer. "
    f"{{{MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER}}}"
)


_DEFAULT_GRANITE_SYSTEM_MESSAGE_TEXT = (
    "You are Granite Chat, an AI language model developed by IBM. "
    "You are a cautious assistant. You carefully follow instructions. "
    "You are helpful and harmless and you follow ethical guidelines and promote positive behaviour."
)


_DEFAULT_GRANITE_USER_MESSAGE_TEXT = (
    "You are an AI language model designed to function as a specialized Retrieval Augmented Generation (RAG) "
    "assistant. When generating responses, prioritize correctness, i.e., ensure that your response is grounded in "
    "context and user query. Always make sure that your response is relevant to the question. "
    "\n"
    "Answer Length: detailed"
    "\n"
    f"{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}"
    "\n"
    f"{{{MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER}}}"
    "\n"
    f"{{{QUESTION_PLACEHOLDER}}} "
    "\n"
    "\n"
)


_DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don’t know the answer to a question, please don’t share false information.\n"
)


_DEFAULT_LLAMA_USER_MESSAGE_TEXT = (
    f"{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n"
    f"[conversation]: {{{QUESTION_PLACEHOLDER}}}. Answer with no more than 150 words. If you cannot base your "
    "answer on the given document, please state that you do not have an answer. "
    f"{{{MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER}}}\n"
)


_DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please don't share false information.\n\n"
)


_DEFAULT_MISTRAL_USER_MESSAGE_TEXT = (
    "Generate the next agent response by answering the question. You are provided several documents with titles. "
    "If the answer comes from different documents please mention all possibilities and use the titles of documents "
    "to separate between topics or domains. If you cannot base your answer on the given documents, "
    f"please state that you do not have an answer. "
    f"{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n\n"
    f"{{{MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER}}}\n\n"
    f"{{{QUESTION_PLACEHOLDER}}}"
)


_DEFAULT_OPENAI_SYSTEM_MESSAGE_TEXT = (
    "You are a AI language model designed to function as a specialized Retrieval Augmented Generation (RAG) assistant. "
    "When generating responses, prioritize correctness, i.e., ensure that your response is correct given the context "
    "and user query, and that it is grounded in the context. "
    "Furthermore, make sure that the response is supported by the given document or context. "
    "When the question cannot be answered using the context or document, output the following response: "
    "'I am sorry, I do not have the information you are looking for in my knowledge base.'. "
    "Always make sure that your response is relevant to the question. If an explanation is needed, "
    "first provide the explanation or reasoning, and then give the final answer.\nAnswer Length: concise.\n\n"
)


_DEFAULT_OPENAI_USER_MESSAGE_TEXT = (
    f"[Document]\n{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n[End]\n"
    f"{{{QUESTION_PLACEHOLDER}}}. \n"
    f"{{{MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER}}}"
)


_model_name_to_system_message_text = {
    FoundationModels.META_LLAMA_3_1_70B_INSTRUCT: _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    FoundationModels.META_LLAMA_3_1_8B_INSTRUCT: _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    FoundationModels.META_LLAMA_3_3_70B_INSTRUCT: _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    FoundationModels.META_LLAMA_4_MAVERICK_17B_128E_INSTRUCT_FP8: _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    FoundationModels.GRANITE_3_8B_INSTRUCT: _DEFAULT_GRANITE_SYSTEM_MESSAGE_TEXT,
    FoundationModels.GRANITE_3_3_8B_INSTRUCT: _DEFAULT_GRANITE_SYSTEM_MESSAGE_TEXT,
    FoundationModels.MISTRAL_SMALL_3_1_24B_INSTRUCT: _DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT,
    FoundationModels.MISTRAL_MEDIUM_2505: _DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT,
    FoundationModels.MISTRAL_MISTRAL_LARGE: _DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT,
    FoundationModels.OPENAI_GPT_OSS_120B: _DEFAULT_OPENAI_SYSTEM_MESSAGE_TEXT,
}


_model_name_to_user_message_text = {
    FoundationModels.META_LLAMA_3_1_70B_INSTRUCT: _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    FoundationModels.META_LLAMA_3_1_8B_INSTRUCT: _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    FoundationModels.META_LLAMA_3_3_70B_INSTRUCT: _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    FoundationModels.META_LLAMA_4_MAVERICK_17B_128E_INSTRUCT_FP8: _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    FoundationModels.GRANITE_3_8B_INSTRUCT: _DEFAULT_GRANITE_USER_MESSAGE_TEXT,
    FoundationModels.GRANITE_3_3_8B_INSTRUCT: _DEFAULT_GRANITE_USER_MESSAGE_TEXT,
    FoundationModels.MISTRAL_SMALL_3_1_24B_INSTRUCT: _DEFAULT_MISTRAL_USER_MESSAGE_TEXT,
    FoundationModels.MISTRAL_MEDIUM_2505: _DEFAULT_MISTRAL_USER_MESSAGE_TEXT,
    FoundationModels.MISTRAL_MISTRAL_LARGE: _DEFAULT_MISTRAL_USER_MESSAGE_TEXT,
    FoundationModels.OPENAI_GPT_OSS_120B: _DEFAULT_OPENAI_USER_MESSAGE_TEXT,
}


# A mapping from model names into their corresponding context template texts. These templates describe how each
# retrieved context is to be wrapped, before being integrated into a full RAG prompt text.
# The parameter for the context template text is CONTEXT_TEXT_PLACEHOLDER
_DEFAULT_GRANITE_CONTEXT_TEMPLATE = f"[Document]\n{{{CONTEXT_TEXT_PLACEHOLDER}}}\n[End]"
_DEFAULT_LLAMA_CONTEXT_TEMPLATE = f"[document]: {{{CONTEXT_TEXT_PLACEHOLDER}}}\n"
_DEFAULT_CONTEXT_TEMPLATE = f"{{{CONTEXT_TEXT_PLACEHOLDER}}}"

_model_name_to_context_template_text = {
    FoundationModels.META_LLAMA_3_1_70B_INSTRUCT: _DEFAULT_LLAMA_CONTEXT_TEMPLATE,
    FoundationModels.META_LLAMA_3_1_8B_INSTRUCT: _DEFAULT_LLAMA_CONTEXT_TEMPLATE,
    FoundationModels.META_LLAMA_3_3_70B_INSTRUCT: _DEFAULT_LLAMA_CONTEXT_TEMPLATE,
    FoundationModels.META_LLAMA_4_MAVERICK_17B_128E_INSTRUCT_FP8: _DEFAULT_LLAMA_CONTEXT_TEMPLATE,
    FoundationModels.GRANITE_3_8B_INSTRUCT: _DEFAULT_GRANITE_CONTEXT_TEMPLATE,
    FoundationModels.GRANITE_3_3_8B_INSTRUCT: _DEFAULT_GRANITE_CONTEXT_TEMPLATE,
    FoundationModels.OPENAI_GPT_OSS_120B: _DEFAULT_CONTEXT_TEMPLATE,
}


def get_context_template_text(model_name: str) -> str:
    """
    Get a model-specific context template text.

    The context template text is a template with one placeholder: "context_text".
    This field should be populated before use within a RAG prompt.

    Parameters
    ----------
    model_name : str
        The name of the model for which we should return the context template text.

    Returns
    -------
    str
        The context template text str for the given model name.
    """
    context_template = _model_name_to_context_template_text.get(model_name, None)

    if not context_template:
        if "granite" in model_name:
            context_template = _DEFAULT_GRANITE_CONTEXT_TEMPLATE
        elif "llama" in model_name:
            context_template = _DEFAULT_LLAMA_CONTEXT_TEMPLATE
        else:
            context_template = _DEFAULT_CONTEXT_TEMPLATE

    return context_template


def get_system_message_text(model_name: str) -> str:
    """
    Get model-specific system prompt text.

    Parameters
    ----------
    model_name : str
        The name of the model for which we should return the system prompt text.

    Returns
    -------
    str
        The system prompt text str for the given model name.
    """

    system_message_text = _model_name_to_system_message_text.get(model_name, None)

    if not system_message_text:
        if "granite" in model_name:
            system_message_text = _DEFAULT_GRANITE_SYSTEM_MESSAGE_TEXT
        elif "llama" in model_name:
            system_message_text = _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT
        elif "mistral" in model_name:
            system_message_text = _DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT
        else:
            system_message_text = _DEFAULT_SYSTEM_MESSAGE_TEXT

    return system_message_text


def get_user_message_text(model_name: str, language_autodetect: bool = True) -> str:
    """
    Get a model-specific prompt text.

    The user message text is a template, with two markers for fields: "question" and "reference_documents".
    These fields should be filled appropriately before delivering the prompt to a model.

    Parameters
    ----------
    model_name : str
        The name of the model for which we should return the prompt text.

    language_autodetect : bool
        If True, language of the question will be automatically detected.

    Returns
    -------
    str
        The prompt text str matching the given model name.
    """
    user_message_text = _model_name_to_user_message_text.get(model_name, None)

    if not user_message_text:
        if "granite" in model_name:
            user_message_text = _DEFAULT_GRANITE_USER_MESSAGE_TEXT
        elif "llama" in model_name:
            user_message_text = _DEFAULT_LLAMA_USER_MESSAGE_TEXT
        elif "mistral" in model_name:
            user_message_text = _DEFAULT_MISTRAL_USER_MESSAGE_TEXT
        else:
            user_message_text = _DEFAULT_USER_MESSAGE_TEXT

    user_message_text = user_message_text.replace(
        "{multilingual_support}",
        _MULTILINGUAL_SUPPORT_ENABLED_PROMPT if language_autodetect else _MULTILINGUAL_SUPPORT_DISABLED_PROMPT,
    )

    return user_message_text
