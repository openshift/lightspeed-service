"""Unit tests for PromptGenerator."""

import pytest
from langchain.prompts import PromptTemplate

from ols.constants import (
    GPT35_TURBO,
    GRANITE_13B_CHAT_V2,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
    PROVIDER_WATSONX,
)
from ols.src.prompts.prompt_generator import generate_prompt
from ols.src.prompts.prompts import (
    CONTEXT_PLACEHOLDER,
    HISTORY_PLACEHOLDER,
    QUERY_PLACEHOLDER,
    QUERY_SYSTEM_INSTRUCTION,
    USE_CONTEXT_INSTRUCTION,
    USE_HISTORY_INSTRUCTION,
)

provider_and_model = (
    (PROVIDER_BAM, GRANITE_13B_CHAT_V2),
    (PROVIDER_OPENAI, GPT35_TURBO),
    (PROVIDER_WATSONX, GRANITE_13B_CHAT_V2),
    (PROVIDER_AZURE_OPENAI, GPT35_TURBO),
)

queries = ("What is Kubernetes?", "When is my birthday?")


@pytest.fixture
def conversation_history():
    """Non-empty conversation history."""
    return [
        "First human message",
        "First AI response",
    ]


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
@pytest.mark.parametrize("query", queries)
def test_generate_prompt_default_prompt(provider, model, query, conversation_history):
    """Test if prompt generator returns default prompt for given input."""
    model_options = {}
    rag_context = "context"

    prompt, llm_input_values = generate_prompt(
        provider,
        model,
        model_options,
        query,
        conversation_history,
        rag_context,
    )

    # prompt with system message, history, and query, should be returned
    # system message should mention context and history
    assert prompt == PromptTemplate.from_template(
        QUERY_SYSTEM_INSTRUCTION
        + USE_CONTEXT_INSTRUCTION
        + CONTEXT_PLACEHOLDER
        + USE_HISTORY_INSTRUCTION
        + HISTORY_PLACEHOLDER
        + QUERY_PLACEHOLDER
    )
    assert "context" in llm_input_values
    assert "query" in llm_input_values
    assert "chat_history" in llm_input_values


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
@pytest.mark.parametrize("query", queries)
def test_generate_prompt_without_rag_context(
    provider, model, query, conversation_history
):
    """Test what prompt will be returned for non-existent RAG context."""
    model_options = {}
    rag_context = ""

    prompt, llm_input_values = generate_prompt(
        provider,
        model,
        model_options,
        query,
        conversation_history,
        rag_context,
    )

    # prompt with system message, history, and query, should be returned
    # system message should mention history, but not context
    assert prompt == PromptTemplate.from_template(
        QUERY_SYSTEM_INSTRUCTION
        + USE_HISTORY_INSTRUCTION
        + HISTORY_PLACEHOLDER
        + QUERY_PLACEHOLDER
    )
    assert "context" not in llm_input_values
    assert "query" in llm_input_values
    assert "chat_history" in llm_input_values
    assert llm_input_values["chat_history"] == "First human message\nFirst AI response"


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
@pytest.mark.parametrize("query", queries)
def test_generate_prompt_without_history(provider, model, query):
    """Test if prompt generator returns prompt without history."""
    model_options = {}
    history = []
    rag_context = "context"

    prompt, llm_input_values = generate_prompt(
        provider,
        model,
        model_options,
        query,
        history,
        rag_context,
    )

    # prompt with system message and query, should be returned
    # system message should mention context, but not history
    assert prompt == PromptTemplate.from_template(
        QUERY_SYSTEM_INSTRUCTION
        + USE_CONTEXT_INSTRUCTION
        + CONTEXT_PLACEHOLDER
        + QUERY_PLACEHOLDER
    )
    assert "context" in llm_input_values
    assert "query" in llm_input_values
    assert "chat_history" not in llm_input_values


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
@pytest.mark.parametrize("query", queries)
def test_generate_prompt_without_rag_without_history(provider, model, query):
    """Test if prompt generator returns prompt without RAG and without history."""
    model_options = {}
    history = []
    rag_context = ""

    prompt, llm_input_values = generate_prompt(
        provider,
        model,
        model_options,
        query,
        history,
        rag_context,
    )

    # prompt with system message and query, should be returned
    # system message should not mention context nor history
    assert prompt == PromptTemplate.from_template(
        QUERY_SYSTEM_INSTRUCTION + QUERY_PLACEHOLDER
    )
    assert "context" not in llm_input_values
    assert "query" in llm_input_values
    assert "chat_history" not in llm_input_values
