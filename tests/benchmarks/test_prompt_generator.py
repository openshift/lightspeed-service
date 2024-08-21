"""Benchmarks for PromptGenerator."""

import pytest

from ols.constants import (
    GPT35_TURBO,
    GRANITE_13B_CHAT_V2,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
    PROVIDER_WATSONX,
)
from ols.src.prompts.prompt_generator import generate_prompt

# providers and models used by parametrized benchmarks
provider_and_model = (
    (PROVIDER_BAM, GRANITE_13B_CHAT_V2),
    (PROVIDER_OPENAI, GPT35_TURBO),
    (PROVIDER_WATSONX, GRANITE_13B_CHAT_V2),
    (PROVIDER_AZURE_OPENAI, GPT35_TURBO),
)


@pytest.fixture
def empty_history():
    """Empty conversation history."""
    return []


@pytest.fixture
def conversation_history():
    """Non-empty conversation history."""
    return [
        "First human message",
        "First AI response",
    ] * 50


@pytest.fixture
def long_history():
    """Long conversation history."""
    return [
        "First human message",
        "First AI response",
    ] * 10000


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_default_prompt(
    benchmark, provider, model, conversation_history
):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?"
    model_options = {}
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        model_options,
        query,
        conversation_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_long_query(benchmark, provider, model, conversation_history):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?" * 10000
    model_options = {}
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        model_options,
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
    model_options = {}
    rag_context = ""

    benchmark(
        generate_prompt,
        provider,
        model,
        model_options,
        query,
        conversation_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_without_history(benchmark, provider, model, empty_history):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?"
    model_options = {}
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        model_options,
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
    model_options = {}
    rag_context = ""

    benchmark(
        generate_prompt,
        provider,
        model,
        model_options,
        query,
        empty_history,
        rag_context,
    )


@pytest.mark.parametrize(("provider", "model"), provider_and_model)
def test_generate_prompt_long_history(benchmark, provider, model, long_history):
    """Benchmark for prompt generator."""
    query = "What is Kubernetes?"
    model_options = {}
    rag_context = "context"

    benchmark(
        generate_prompt,
        provider,
        model,
        model_options,
        query,
        long_history,
        rag_context,
    )
