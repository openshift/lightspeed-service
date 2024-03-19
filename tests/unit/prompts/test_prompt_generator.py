"""Unit tests for PromptGenerator."""

import pytest

from ols.constants import (
    GPT35_TURBO,
    GRANITE_13B_CHAT_V1,
    PROVIDER_BAM,
    PROVIDER_OPENAI,
)
from ols.src.prompts.prompt_generator import generate_prompt
from ols.src.prompts.prompts import CHAT_PROMPT

provider_and_model = (
    (PROVIDER_BAM, GRANITE_13B_CHAT_V1),
    (PROVIDER_OPENAI, GPT35_TURBO),
)

queries = ("What is Kubernetes?", "When is my birthday?")


@pytest.mark.parametrize("provider,model", provider_and_model)
@pytest.mark.parametrize("query", queries)
def test_generate_prompt_default_prompt(provider, model, query):
    """Test if prompt generator returns default prompt for given input."""
    model_options = {}
    conversation_id = "154e0444-db72-44fd-a6be-4ba0c15059a4"
    history = []
    rag_context = ""
    referenced_documents = []

    # currently, default CHAT_PROMPT should be returned for all cases
    prompt = generate_prompt(
        provider,
        model,
        model_options,
        conversation_id,
        query,
        history,
        rag_context,
        referenced_documents,
    )
    assert prompt == CHAT_PROMPT
