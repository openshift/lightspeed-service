"""Helper classes to count tokens sent and received by the LLM."""

import logging
from typing import Any

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models.llms import LLM

from ols.app.metrics.metrics import (
    llm_calls_total,
    llm_reasoning_token_total,
    llm_token_received_total,
    llm_token_sent_total,
)
from ols.app.models.models import TokenCounter
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


class GenericTokenCounter(AsyncCallbackHandler):  # pylint: disable=R0901
    """A callback handler to count tokens sent and received by the LLM.

    It provides counters via TokenCounter dataclass stored as an attribute:
    - input_tokens: number of input tokens counted by the handler (tiktoken)
    - output_tokens: number of output tokens counted by the handler (tiktoken)
    - reasoning_tokens: number of reasoning summary tokens (tiktoken)
    - llm_calls: number of LLM calls
    """

    def __init__(self, llm: LLM) -> None:
        """Initialize the token counter callback handler.

        Args:
            llm: The LLM instance.
        """
        self.token_counter = TokenCounter()
        self.token_counter.llm = llm  # actual LLM instance
        self.token_handler = TokenHandler()  # used for counting input and output tokens

    async def on_llm_new_token(  # type: ignore[override]
        self,
        token: str | list[Any],
        **kwargs: Any,
    ) -> None:
        """Compute token count when llm token is yielded.

        LangChain declares ``token`` as ``str``, but reasoning-capable
        models emit ``list[dict]`` content blocks for mixed text/reasoning
        output.  We widen the signature here to reflect the runtime reality.
        """
        if token and isinstance(token, str):
            self.token_counter.output_tokens += self.tokens_count(token)
        elif isinstance(token, list):
            self._count_list_content_tokens(token)

    def _count_list_content_tokens(self, blocks: list[Any]) -> None:
        """Count text and reasoning tokens from list-format content blocks.

        OpenAI reasoning models emit content as a list of typed blocks::

            [
                {"type": "text", "text": "..."},
                {"type": "reasoning", "summary": [
                    {"type": "summary_text", "text": "..."}
                ]},
            ]
        """
        for block in blocks:
            if not isinstance(block, dict):
                continue
            match block.get("type"):
                case "text":
                    text = block.get("text", "")
                    if text:
                        self.token_counter.output_tokens += self.tokens_count(text)
                case "reasoning":
                    for part in block.get("summary", []):
                        if isinstance(part, dict):
                            text = part.get("text", "")
                            if text:
                                self.token_counter.reasoning_tokens += (
                                    self.tokens_count(text)
                                )

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.token_counter.llm_calls += 1
        for p in prompts:
            self.token_counter.input_tokens += self.tokens_count(p)

    def tokens_count(self, text: str) -> int:
        """Compute tokens count for given input text."""
        return len(self.token_handler.text_to_tokens(text))

    def __str__(self) -> str:
        """Textual representation of GenericTokenCounter instance."""
        return (
            f"{self.__class__.__name__}: "
            + f"input_tokens: {self.token_counter.input_tokens} "
            + f"output_tokens: {self.token_counter.output_tokens} "
            + f"reasoning_tokens: {self.token_counter.reasoning_tokens} "
            + f"LLM calls: {self.token_counter.llm_calls}"
        )


class TokenMetricUpdater:
    """A context manager to update token metrics in a callback handler.

    These metrics are updated:
    - llm_token_sent_total
    - llm_token_received_total
    - llm_reasoning_token_total
    - llm_calls_total

    Example usage:
        ```python
        bare_llm = self.llm_loader(
            self.provider, self.model, llm_params=self.llm_params
        )
        llm_chain = prompt_instructions | bare_llm
        with TokenMetricUpdater(
            llm=bare_llm,
            provider=self.provider,
            model=self.model,
        ) as token_counter:
            response = llm_chain.invoke(
                input={"query": query}, config={"callbacks": [token_counter]}
            )
        ```
    """

    def __init__(self, llm: LLM, provider: str, model: str) -> None:
        """Initialize the token counter context manager.

        Args:
            llm: The LLM instance.
            provider: The provider name for labeling the metrics.
            model: The model name for labeling the metrics.
        """
        self.token_counter = GenericTokenCounter(llm=llm)
        self.provider = provider
        self.model = model

    def __enter__(self) -> GenericTokenCounter:
        """Initialize the token counter when entering the context."""
        return self.token_counter

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Update the metrics when exiting the context."""
        llm_calls_total.labels(provider=self.provider, model=self.model).inc(
            self.token_counter.token_counter.llm_calls
        )
        llm_token_sent_total.labels(provider=self.provider, model=self.model).inc(
            self.token_counter.token_counter.input_tokens
        )
        llm_token_received_total.labels(provider=self.provider, model=self.model).inc(
            self.token_counter.token_counter.output_tokens
        )
        llm_reasoning_token_total.labels(provider=self.provider, model=self.model).inc(
            self.token_counter.token_counter.reasoning_tokens
        )
