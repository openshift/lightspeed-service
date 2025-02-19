"""Benchmarks for PromptGenerator."""

# pylint: disable=W0621

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols.constants import (
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
    PROVIDER_RHELAI_VLLM,
    PROVIDER_RHOAI_VLLM,
    PROVIDER_WATSONX,
)
from ols.src.prompts.prompt_generator import GeneratePrompt

# providers and models used by parametrized benchmarks
provider_and_model = (
    (PROVIDER_BAM, "some-granite-model"),
    (PROVIDER_OPENAI, "some-gpt-model"),
    (PROVIDER_WATSONX, "some-granite-model"),
    (PROVIDER_AZURE_OPENAI, "some-gpt-model"),
    (PROVIDER_RHOAI_VLLM, "some-granite-model"),
    (PROVIDER_RHELAI_VLLM, "some-granite-model"),
)


@pytest.fixture
def empty_history():
    """Empty conversation history."""
    return []


@pytest.fixture
def conversation_history():
    """Non-empty conversation history."""
    return [
        HumanMessage("First human message"),
        AIMessage("First AI response"),
    ] * 50


@pytest.fixture
def long_history():
    """Long conversation history."""
    return [
        HumanMessage("First human message"),
        AIMessage("First AI response"),
    ] * 10000


def generate_prompt(provider, model, query, history, rag_content):
    """Initialize and call prompt generator."""
    prompt_generator = GeneratePrompt(query, rag_content, history)
    prompt_generator.generate_prompt(model)


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_default_prompt(
    benchmark, provider, model, conversation_history
):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?"
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        query,
        conversation_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_long_query(benchmark, provider, model, conversation_history):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?" * 10000
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        query,
        conversation_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_without_rag_context(
    benchmark, provider, model, conversation_history
):
    """Benchmark what prompt will be returned for non-existent RAG context."""
    query = "What is Kubernetes?"
    rag_context = ""

    benchmark(
        generate_prompt,
        provider,
        model,
        query,
        conversation_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_without_history(benchmark, provider, model, empty_history):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?"
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        query,
        empty_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_without_rag_context_nor_history(
    benchmark, provider, model, empty_history
):
    """Benchmark what prompt will be returned for non-existent RAG context."""
    query = "What is Kubernetes?"
    rag_context = ""

    benchmark(
        generate_prompt,
        provider,
        model,
        query,
        empty_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_long_history(benchmark, provider, model, long_history):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?"
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        query,
        long_history,
        rag_context,
    )
