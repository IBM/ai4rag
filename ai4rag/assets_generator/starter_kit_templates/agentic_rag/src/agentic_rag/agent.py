from os import getenv
from typing import Callable

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from .config import AgentConfig, get_chat_base_url
from .tools import retriever_tool

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a retriever tool for searching a knowledge base.\n\n"
    "CRITICAL INSTRUCTION: When a user asks ANY question, you MUST respond by calling the 'retriever' tool FIRST. "
    "Do NOT attempt to answer from your own knowledge until AFTER you have called the retriever tool and seen its results.\n\n"
    "Process:\n"
    "1. User asks a question → immediately call retriever tool with relevant search query\n"
    "2. Receive retriever results → use them to formulate your answer\n"
    "3. If no relevant information found → then use general knowledge\n"
    "4. Always cite sources when available\n\n"
    "Example:\n"
    "User: 'What are appropriate chunk sizes?'\n"
    "You: [MUST call retriever tool with query='chunk sizes' or 'appropriate chunk sizes']"
)


def get_graph_closure(
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Callable:
    """Build and return a LangGraph ReAct agent with the configured LLM and retrieval tool.

    Args:
        model_id: LLM model identifier. Uses MODEL_ID env if omitted.
        base_url: Base URL for the LLM API. Uses CHAT_BASE_URL, BASE_URL, or MAAS_BASE_URL if omitted.
        api_key: API key for the LLM. Uses MAAS_API_KEY env if omitted.

    Returns:
        A function that creates a CompiledGraph agent.
    """
    if not api_key:
        api_key = getenv("MAAS_API_KEY")
    if not base_url:
        base_url = get_chat_base_url()
    if not model_id:
        model_id = getenv("MODEL_ID")

    if not base_url:
        raise ValueError(
            "CHAT_BASE_URL or BASE_URL is required for the chat model."
        )
    is_local = any(host in base_url for host in ["localhost", "127.0.0.1"])

    if not is_local and not api_key:
        raise ValueError("MAAS_API_KEY is required for non-local environments.")

    config = AgentConfig.from_env()

    chat = ChatOpenAI(
        model=model_id,
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens,
        api_key=api_key or "not-needed-for-local-development",
        base_url=base_url,
        model_kwargs={"tool_choice": "auto"},
    )

    TOOLS = [retriever_tool]

    system_prompt_text = config.system_message or _DEFAULT_SYSTEM_PROMPT
    if config.language_name and config.language_name.lower() != "auto":
        system_prompt_text += f"\n\nRespond in {config.language_name}."

    def get_graph(
        instruction_prompt: SystemMessage | None = None,
    ) -> CompiledStateGraph:
        system_message_text = system_prompt_text
        if instruction_prompt is not None:
            content = instruction_prompt.content
            if isinstance(content, str):
                system_message_text = system_prompt_text + "\n\n" + content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                system_message_text = (
                    system_prompt_text + "\n\n" + " ".join(text_parts)
                )

        graph = create_agent(
            model=chat,
            tools=TOOLS,
            system_prompt=system_message_text,
        )

        return graph

    return get_graph
