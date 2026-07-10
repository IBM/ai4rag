# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
__all__ = [
    "get_system_message_text",
    "get_user_message_text",
    "get_context_template_text",
    "QUESTION_PLACEHOLDER",
    "REFERENCE_DOCUMENTS_PLACEHOLDER",
    "CONTEXT_TEXT_PLACEHOLDER",
    "DOCUMENT_NUMBER_PLACEHOLDER",
    "MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER",
]


QUESTION_PLACEHOLDER = "question"
REFERENCE_DOCUMENTS_PLACEHOLDER = "reference_documents"
CONTEXT_TEXT_PLACEHOLDER = "document"
DOCUMENT_NUMBER_PLACEHOLDER = "doc_number"
MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER = "multilingual_support"


_LANGUAGE_AUTODETECT_PROMPT = (
    "You MUST write your entire answer in the same language as the question. "
    "Do NOT respond in any other language, even if the documents use a different language. "
    "Every word of your answer must match the question's language."
)


_LANGUAGE_INSTRUCTION = "You MUST respond in {lang_name}."


_DEFAULT_NUMBERED_CONTEXT_TEMPLATE = f"Document {{{DOCUMENT_NUMBER_PLACEHOLDER}}}:\n{{{CONTEXT_TEXT_PLACEHOLDER}}}\n"


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
    f"{{{QUESTION_PLACEHOLDER}}}"
    "\n"
    "\n"
)


_DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please don't share false information.\n"
)


_DEFAULT_LLAMA_USER_MESSAGE_TEXT = (
    f"{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n"
    f"[conversation]: {{{QUESTION_PLACEHOLDER}}}. Be concise. If you cannot base your "
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
    "You are an AI language model designed to function as a specialized Retrieval Augmented Generation (RAG) assistant. "
    "When generating responses, prioritize correctness, i.e., ensure that your response is correct given the context "
    "and user query, and that it is grounded in the context. "
    "Furthermore, make sure that the response is supported by the given document or context. "
    "When the question cannot be answered using the context or document, output the following response: "
    "'I am sorry, I do not have the information you are looking for in my knowledge base.'. "
    "Always make sure that your response is relevant to the question. If an explanation is needed, "
    "first provide the explanation or reasoning, and then give the final answer.\n\n"
)


_DEFAULT_OPENAI_USER_MESSAGE_TEXT = (
    f"[Document]\n{{{REFERENCE_DOCUMENTS_PLACEHOLDER}}}\n[End]\n"
    f"{{{QUESTION_PLACEHOLDER}}}. \n"
    f"{{{MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER}}}"
)


_model_name_to_system_message_text = {
    "meta-llama/llama-3-1-70b-instruct": _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    "meta-llama/llama-3-1-8b-instruct": _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    "meta-llama/llama-3-3-70b-instruct": _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": _DEFAULT_LLAMA_SYSTEM_MESSAGE_TEXT,
    "ibm/granite-3-8b-instruct": _DEFAULT_GRANITE_SYSTEM_MESSAGE_TEXT,
    "ibm/granite-3-3-8b-instruct": _DEFAULT_GRANITE_SYSTEM_MESSAGE_TEXT,
    "mistralai/mistral-small-3-1-24b-instruct-2503": _DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT,
    "mistralai/mistral-medium-2505": _DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT,
    "mistralai/mistral-large": _DEFAULT_MISTRAL_SYSTEM_MESSAGE_TEXT,
    "openai/gpt-oss-120b": _DEFAULT_OPENAI_SYSTEM_MESSAGE_TEXT,
}


_model_name_to_user_message_text = {
    "meta-llama/llama-3-1-70b-instruct": _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    "meta-llama/llama-3-1-8b-instruct": _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    "meta-llama/llama-3-3-70b-instruct": _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": _DEFAULT_LLAMA_USER_MESSAGE_TEXT,
    "ibm/granite-3-8b-instruct": _DEFAULT_GRANITE_USER_MESSAGE_TEXT,
    "ibm/granite-3-3-8b-instruct": _DEFAULT_GRANITE_USER_MESSAGE_TEXT,
    "mistralai/mistral-small-3-1-24b-instruct-2503": _DEFAULT_MISTRAL_USER_MESSAGE_TEXT,
    "mistralai/mistral-medium-2505": _DEFAULT_MISTRAL_USER_MESSAGE_TEXT,
    "mistralai/mistral-large": _DEFAULT_MISTRAL_USER_MESSAGE_TEXT,
    "openai/gpt-oss-120b": _DEFAULT_OPENAI_USER_MESSAGE_TEXT,
}


def get_context_template_text() -> str:
    """
    Get a model-specific context template text.

    The context template text is a template with placeholders ``document`` and,
    optionally, ``doc_number``.

    **kwargs are set for backward compatibility and future improvements.

    Returns
    -------
    str
        The context template text str for the given model name.
    """
    return _DEFAULT_NUMBERED_CONTEXT_TEMPLATE


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
        elif "openai" in model_name or "gpt" in model_name:
            system_message_text = _DEFAULT_OPENAI_SYSTEM_MESSAGE_TEXT
        else:
            system_message_text = _DEFAULT_SYSTEM_MESSAGE_TEXT

    return system_message_text


def get_user_message_text(model_name: str, language: str = "auto") -> str:
    """
    Get a model-specific prompt text.

    The user message text is a template, with two markers for fields: "question" and "reference_documents".
    These fields should be filled appropriately before delivering the prompt to a model.

    Parameters
    ----------
    model_name : str
        The name of the model for which we should return the prompt text.

    language : str, default="auto"
        Language in which model should respond. If "auto" is selected,
        model responds in the language of question.

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
        elif "openai" in model_name or "gpt" in model_name:
            user_message_text = _DEFAULT_OPENAI_USER_MESSAGE_TEXT
        else:
            user_message_text = _DEFAULT_USER_MESSAGE_TEXT

    if language == "auto":
        language_instruction = _LANGUAGE_AUTODETECT_PROMPT
    else:
        language_instruction = _LANGUAGE_INSTRUCTION.format(lang_name=language)

    user_message_text = user_message_text.replace(
        f"{{{MULTILINGUAL_SUPPORT_INSTRUCTION_PLACEHOLDER}}}", language_instruction
    )

    return user_message_text
