"""Helper classes to count tokens sent and received by the LLM."""

import logging
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import LLM
from langchain_core.outputs.llm_result import LLMResult

from ols.utils.token_handler import TokenHandler

from .metrics import llm_calls_total, llm_token_received_total, llm_token_sent_total

logger = logging.getLogger(__name__)


class GenericTokenCounter(BaseCallbackHandler):
    """A callback handler to count tokens sent and received by the LLM.

    It provides 3 counters:
    - input_tokens: number of tokens sent to LLM
    - output_tokens: number of tokens received from LLM
    - llm_calls: number of LLM calls

    Example usage:
        ```python
        bare_llm = self.llm_loader(
            self.provider, self.model, llm_params=self.llm_params
        )
        token_counter = GenericTokenCounter(bare_llm)
        llm_chain = LLMChain(
            llm=bare_llm,
            prompt=prompt_instructions,
            verbose=verbose,
        )
        response = llm_chain.invoke(
            input={"query": query}, config={"callbacks": [token_counter]}
        )
        metrics.llm_token_sent_total.labels(
            provider=self.provider, model=self.model
        ).inc(token_counter.input_tokens)
        metrics.llm_token_received_total.labels(
            provider=self.provider, model=self.model
        ).inc(token_counter.output_tokens)
        ```
    """

    def __init__(self, llm: LLM) -> None:
        """Initialize the token counter callback handler.

        Args:
            llm: The LLM instance.
        """
        self.llm = llm  # LLM instance
        self.input_tokens = 0  # number of tokens sent to LLM
        self.output_tokens = 0  # number of tokens received from LLM
        self.input_tokens_counted = 0  # number of input tokens counted by the handler
        self.llm_calls = 0  # number of LLM calls
        self.token_handler = TokenHandler()  # used for counting input and output tokens

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.llm_calls += 1
        self.input_tokens_counted = 0
        for p in prompts:
            self.input_tokens_counted += self.tokens_count(p)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM completes running."""
        results = response.flatten()
        input_tokens_llm_reported = 0
        for r in results:
            if r.llm_output is not None and "token_usage" in r.llm_output:
                token_usage = r.llm_output["token_usage"]
                # typical token_usage: {'prompt_tokens': 252, 'completion_tokens': 4, 'total': 256}
                if "prompt_tokens" in token_usage:
                    input_tokens_llm_reported += token_usage["prompt_tokens"]

                if "completion_tokens" in token_usage:
                    self.output_tokens += token_usage["completion_tokens"]
                else:
                    # fallback to token counting if counter is not provided by LLM
                    text = r.generations[0][0].text
                    self.output_tokens += self.tokens_count(text)

            else:
                # fallback to token counting if LLM does not return token_usage metadata
                text = r.generations[0][0].text
                self.output_tokens += self.tokens_count(text)

        # override the input tokens count if we have a value from LLM response
        if input_tokens_llm_reported > 0:
            self.input_tokens += input_tokens_llm_reported
        else:
            self.input_tokens += self.input_tokens_counted

    def tokens_count(self, text: str) -> int:
        """Compute tokens count for given input text."""
        return len(self.token_handler.text_to_tokens(text))


class TokenMetricUpdater:
    """A context manager to update token metrics in a callback handler.

    These metrics are updated:
    - llm_token_sent_total
    - llm_token_received_total
    - llm_calls_total

    Example usage:
        ```python
        bare_llm = self.llm_loader(
            self.provider, self.model, llm_params=self.llm_params
        )
        llm_chain = LLMChain(
            llm=bare_llm,
            prompt=prompt_instructions,
            verbose=verbose,
        )
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
            self.token_counter.llm_calls
        )
        llm_token_sent_total.labels(provider=self.provider, model=self.model).inc(
            self.token_counter.input_tokens
        )
        llm_token_received_total.labels(provider=self.provider, model=self.model).inc(
            self.token_counter.output_tokens
        )
