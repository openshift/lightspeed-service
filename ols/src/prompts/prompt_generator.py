"""Prompt generator based on model / context."""

from collections import namedtuple
from typing import Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.messages.base import BaseMessage

from ols.constants import (
    GPT35_TURBO,
    GRANITE_13B_CHAT_V2,
    NO_RAG_CONTENT_RESP,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
    PROVIDER_WATSONX,
)

from .prompts import (
    CONTEXT_PLACEHOLDER,
    HISTORY_PLACEHOLDER,
    QUERY_PLACEHOLDER,
    QUERY_SYSTEM_INSTRUCTION,
    USE_CONTEXT_INSTRUCTION,
    USE_HISTORY_INSTRUCTION,
)

PromptConfiguration = namedtuple("PromptConfiguration", "provider model")

# This dictionary might be expanded in the future, if some combination of model+provider
# requires specifically updated prompt.
prompt_configurations = {
    PromptConfiguration(PROVIDER_BAM, GRANITE_13B_CHAT_V2): QUERY_SYSTEM_INSTRUCTION,
    PromptConfiguration(PROVIDER_OPENAI, GPT35_TURBO): QUERY_SYSTEM_INSTRUCTION,
    PromptConfiguration(
        PROVIDER_WATSONX, GRANITE_13B_CHAT_V2
    ): QUERY_SYSTEM_INSTRUCTION,
    PromptConfiguration(PROVIDER_AZURE_OPENAI, GPT35_TURBO): QUERY_SYSTEM_INSTRUCTION,
}


def generate_prompt(
    provider: str,
    model: str,
    model_options: Optional[dict[str, Any]],
    query: str,
    history: list[BaseMessage],
    rag_context: str,
) -> tuple[PromptTemplate, dict[str, Any]]:
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
    # Add system instruction to the prompt first.
    selector = PromptConfiguration(provider, model)
    prompt = prompt_configurations.get(selector, QUERY_SYSTEM_INSTRUCTION)
    llm_input_values = {"query": query}

    # Add optional instruction to the prompt & create prompt inputs.
    if len(rag_context) > 0:
        prompt += USE_CONTEXT_INSTRUCTION
        prompt += CONTEXT_PLACEHOLDER

        llm_input_values["context"] = rag_context

    if len(history) > 0:
        prompt += USE_HISTORY_INSTRUCTION
        prompt += HISTORY_PLACEHOLDER
        formatted_history = [
            conversation.type.capitalize()
            + ": "
            + conversation.content.strip().replace(NO_RAG_CONTENT_RESP, "")
            for conversation in history
            if conversation
        ]
        llm_input_values["chat_history"] = "\n".join(formatted_history)

    prompt += QUERY_PLACEHOLDER

    return PromptTemplate.from_template(prompt), llm_input_values
