"""Prompt generator based on model / context."""

from collections import namedtuple
from typing import Any, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages.base import BaseMessage

from ols.constants import (
    GPT35_TURBO,
    GRANITE_13B_CHAT_V1,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
)

from .prompts import CHAT_PROMPT

PromptConfiguration = namedtuple(
    "PromptConfiguration", "provider model rag_exists history_exists"
)

# TODO: this might be expanded in the future.
prompt_configurations = {
    PromptConfiguration(PROVIDER_BAM, GRANITE_13B_CHAT_V1, True, True): CHAT_PROMPT,
    PromptConfiguration(PROVIDER_OPENAI, GPT35_TURBO, True, True): CHAT_PROMPT,
}


def prompt_for_configuration(
    provider: str,
    model: str,
    rag_exists: bool,
    history_exists: bool,
    default_prompt: ChatPromptTemplate,
) -> ChatPromptTemplate:
    """Find prompt for given configuration parameters or return the default one."""
    selector = PromptConfiguration(provider, model, rag_exists, history_exists)
    return prompt_configurations.get(selector, default_prompt)


def generate_prompt(
    provider: str,
    model: str,
    model_options: Optional[dict[str, Any]],
    conversation_id: str,
    query: str,
    history: list[BaseMessage],
    rag_context: str,
    referenced_documents: list[str],
) -> ChatPromptTemplate:
    """Dynamically creates prompt template.

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
        provider, model, rag_exists, history_exists, CHAT_PROMPT
    )
    # TODO: this is place to insert logic there for specific cases, for example:
    #
    # ```python
    # if model == PROVIDER_BAM and len(rag_context) > model_options.max_rag_context:
    #    CHAT_PROMPT.expand(whatever is needed)
    # ```

    return prompt
