"""Benchmarks for the token handler."""

from ols.utils.token_handler import TokenHandler


def benchmark_token_hander(
    benchmark, prompt, context_window_size=500, max_tokens_for_response=20
):
    """Benchmark the method to calculate available tokens."""
    token_handler = TokenHandler()

    benchmark(
        token_handler.calculate_and_check_available_tokens,
        prompt,
        context_window_size,
        max_tokens_for_response,
    )


def test_available_tokens_for_empty_prompt(benchmark):
    """Benchmark the method to calculate available tokens for empty prompt."""
    prompt = ""
    benchmark_token_hander(benchmark, prompt)


def test_available_tokens_for_regular_prompt(benchmark):
    """Benchmark the method to calculate available tokens for regular prompt."""
    prompt = "What is Kubernetes?"
    benchmark_token_hander(benchmark, prompt)


def test_available_tokens_for_large_prompt(benchmark):
    """Benchmark the method to calculate available tokens for large prompt."""
    prompt = "What is Kubernetes?" * 100
    benchmark_token_hander(benchmark, prompt)


def test_available_tokens_for_huge_prompt(benchmark):
    """Benchmark the method to calculate available tokens for huge prompt."""
    # this prompt will not exceed default context window size
    # it means we will update the size accordingly
    prompt = "What is Kubernetes?" * 10000
    benchmark_token_hander(benchmark, prompt, context_window_size=50000)


def benchmark_limit_conversation_history(benchmark, history, limit=1000):
    """Benchmark the method to calculate available tokens."""
    token_handler = TokenHandler()

    benchmark(token_handler.limit_conversation_history, history, limit)


def test_limit_conversation_history_no_history(benchmark):
    """Benchark for limiting conversation history if it does not exists."""
    # benchmark with empty conversation history
    benchmark_limit_conversation_history(benchmark, [])


def test_limit_conversation_history_short_history(benchmark):
    """Benchark for limiting conversation history."""
    # short conversation history with just 3 questions and 3 answers
    history = [
        "first message from human",
        "first answer from AI",
        "second message from human",
        "second answer from AI",
        "third message from human",
        "third answer from AI",
    ]

    benchmark_limit_conversation_history(benchmark, history)


def test_limit_conversation_history_long_history(benchmark):
    """Benchark for limiting conversation history."""
    # longer history with 600 messages in overall
    history = [
        "first message from human",
        "first answer from AI",
        "second message from human",
        "second answer from AI",
        "third message from human",
        "third answer from AI",
    ] * 100

    benchmark_limit_conversation_history(benchmark, history)


def test_limit_conversation_history_huge_history(benchmark):
    """Benchark for limiting conversation history."""
    # huge history consisting of 60000 messages
    history = [
        "first message from human",
        "first answer from AI",
        "second message from human",
        "second answer from AI",
        "third message from human",
        "third answer from AI",
    ] * 10000

    benchmark_limit_conversation_history(benchmark, history)
