"""Prompt generator based on model / context."""

from collections import namedtuple
from typing import Any, Optional

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages.base import BaseMessage

from ols.constants import (
    GPT35_TURBO,
    GRANITE_13B_CHAT_V2,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
    PROVIDER_WATSONX,
)

from .prompts import (
    QUERY_SYSTEM_PROMPT,
    USE_PREVIOUS_HISTORY,
    USE_RETRIEVED_CONTEXT,
)

PromptConfiguration = namedtuple("PromptConfiguration", "provider model")

# TODO: this might be expanded in the future.
prompt_configurations = {
    PromptConfiguration(PROVIDER_BAM, GRANITE_13B_CHAT_V2): QUERY_SYSTEM_PROMPT,
    PromptConfiguration(PROVIDER_OPENAI, GPT35_TURBO): QUERY_SYSTEM_PROMPT,
    PromptConfiguration(PROVIDER_WATSONX, GRANITE_13B_CHAT_V2): QUERY_SYSTEM_PROMPT,
    PromptConfiguration(PROVIDER_AZURE_OPENAI, GPT35_TURBO): QUERY_SYSTEM_PROMPT,
}


def prompt_for_configuration(
    provider: str,
    model: str,
    rag_exists: bool,
    history_exists: bool,
    default_prompt: ChatPromptTemplate,
) -> ChatPromptTemplate:
    """Find prompt for given configuration parameters or return the default one."""
    # construct system prompt first
    selector = PromptConfiguration(provider, model)
    system_prompt = prompt_configurations.get(selector, default_prompt)

    # add optional parts into system prompt
    if rag_exists:
        system_prompt += USE_RETRIEVED_CONTEXT
    if history_exists:
        system_prompt += USE_PREVIOUS_HISTORY

    # construct chat prompt from sequence of messages
    messages = []
    messages.append(SystemMessagePromptTemplate.from_template(system_prompt))
    if history_exists:
        messages.append(MessagesPlaceholder(variable_name="chat_history"))
    messages.append(HumanMessagePromptTemplate.from_template("{query}"))

    return ChatPromptTemplate.from_messages(messages)


def generate_prompt(
    provider: str,
    model: str,
    model_options: Optional[dict[str, Any]],
    query: str,
    history: list[BaseMessage],
    rag_context: str,
) -> tuple[ChatPromptTemplate, dict[str, Any]]:
    """Dynamically creates prompt template and input values specification for LLM.

    Prompt template is created by combining these values:
    - provider
    - model
    - model-specific configuration
    - RAG (if enabled and exists)
    - history
    - system prompt
    - user query
    """
    rag_exists = len(rag_context) > 0
    history_exists = len(history) > 0
    prompt = prompt_for_configuration(
        provider, model, rag_exists, history_exists, QUERY_SYSTEM_PROMPT
    )

    # construct LLM input values
    llm_input_values: dict[str, Any] = {"query": query}
    if rag_exists:
        llm_input_values["context"] = rag_context
    if history_exists:
        llm_input_values["chat_history"] = history

    # TODO: this is place to insert logic there for specific cases, for example:
    #
    # ```python
    # if model == PROVIDER_BAM and len(rag_context) > model_options.max_rag_context:
    #    prompt.expand(whatever is needed)
    # ```

    return prompt, llm_input_values
