"""Unit tests for GenericTokenCounter and TokenMetricUpdater classes."""

import pytest

from ols import config

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"


from ols.app.metrics import GenericTokenCounter  # noqa:E402
from ols.app.metrics.token_counter import TokenMetricUpdater  # noqa:E402


class MockLLM:
    """Mocked LLM to be used in unit tests."""

    def get_num_tokens(self, prompt):
        """Poor man's token counter."""
        return len(prompt.split(" "))


@pytest.mark.asyncio
async def test_on_llm_start():
    """Test the GenericTokenCounter.on_llm_start method."""
    llm = MockLLM()

    # initialize new token counter
    generic_token_counter = GenericTokenCounter(llm)

    # a beginning the counters should be zeroed
    assert generic_token_counter.token_counter.llm_calls == 0
    assert generic_token_counter.token_counter.input_tokens == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 "
        "reasoning_tokens: 0 LLM calls: 0"
    )
    assert str(generic_token_counter) == expected

    # token count for empty input
    await generic_token_counter.on_llm_start({}, [])

    # token counter needs to be zero as mocked LLM does not process anything
    assert generic_token_counter.token_counter.llm_calls == 1
    assert generic_token_counter.token_counter.input_tokens == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 "
        "reasoning_tokens: 0 LLM calls: 1"
    )
    assert str(generic_token_counter) == expected

    # now the prompt will be tokenized into 5 tokens
    await generic_token_counter.on_llm_start({}, ["this is just a test"])
    assert generic_token_counter.token_counter.llm_calls == 2
    assert generic_token_counter.token_counter.input_tokens == 5

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 5 output_tokens: 0 "
        "reasoning_tokens: 0 LLM calls: 2"
    )
    assert str(generic_token_counter) == expected


@pytest.mark.asyncio
async def test_on_llm_end():
    """Test the GenericTokenCounter.on_llm_new_token method."""
    llm = MockLLM()

    # initialize new token counter
    generic_token_counter = GenericTokenCounter(llm)
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 "
        "reasoning_tokens: 0 LLM calls: 0"
    )
    assert str(generic_token_counter) == expected

    # empty token
    await generic_token_counter.on_llm_new_token("")
    await generic_token_counter.on_llm_new_token(None)

    # for empty response, counters should not change
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 0

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 0 "
        "reasoning_tokens: 0 LLM calls: 0"
    )
    assert str(generic_token_counter) == expected

    # non-empty response
    await generic_token_counter.on_llm_new_token("hello")
    await generic_token_counter.on_llm_new_token("there")

    # for non-empty response, counters should change
    assert generic_token_counter.token_counter.input_tokens == 0
    assert generic_token_counter.token_counter.output_tokens == 2

    # check the textual representation as well
    expected = (
        "GenericTokenCounter: input_tokens: 0 output_tokens: 2 "
        "reasoning_tokens: 0 LLM calls: 0"
    )
    assert str(generic_token_counter) == expected


@pytest.mark.asyncio
async def test_on_llm_new_token_list_content_with_text_and_reasoning():
    """Test on_llm_new_token counts text and reasoning tokens from list content."""
    llm = MockLLM()
    counter = GenericTokenCounter(llm)

    list_content = [
        {"type": "text", "text": "hello world"},
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thinking step"}],
        },
        "not-a-dict-ignored",
        {"type": "unknown_block"},
    ]
    await counter.on_llm_new_token(list_content)

    assert counter.token_counter.output_tokens == 2
    assert counter.token_counter.reasoning_tokens == 2


@pytest.mark.asyncio
async def test_on_llm_new_token_list_content_empty_blocks():
    """Test on_llm_new_token handles empty text and reasoning gracefully."""
    llm = MockLLM()
    counter = GenericTokenCounter(llm)

    list_content = [
        {"type": "text", "text": ""},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": ""}]},
        {"type": "reasoning", "summary": []},
    ]
    await counter.on_llm_new_token(list_content)

    assert counter.token_counter.output_tokens == 0
    assert counter.token_counter.reasoning_tokens == 0


def test_token_metric_updater_reports_reasoning_tokens():
    """Test TokenMetricUpdater.__exit__ increments the reasoning token metric."""
    llm = MockLLM()
    updater = TokenMetricUpdater(llm=llm, provider="test_provider", model="test_model")

    updater.token_counter.token_counter.input_tokens = 10
    updater.token_counter.token_counter.output_tokens = 20
    updater.token_counter.token_counter.reasoning_tokens = 5
    updater.token_counter.token_counter.llm_calls = 1

    updater.__exit__(None, None, None)
