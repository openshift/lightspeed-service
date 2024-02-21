"""Helper classes to count tokens sent and received by the LLM."""

import logging
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult

from .metrics import llm_calls_total, llm_token_received_total, llm_token_sent_total

logger = logging.getLogger(__name__)


class GenericTokenCounter(BaseCallbackHandler):
    """A callback handler to count tokens sent and received by the LLM.

    It provides 3 counters:
    - input_tokens: number of tokens sent to LLM
    - output_tokens: number of tokens received from LLM
    - llm_calls: number of LLM calls

    Example usage:

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

    """

    def __init__(self, llm):
        """Initialize the token counter callback handler.

        Parameters:
            llm (LLM): the LLM instance
        """
        self.llm = llm  # LLM instance
        self.input_tokens = 0  # number of tokens sent to LLM
        self.output_tokens = 0  # number of tokens received from LLM
        self.input_tokens_counted = 0  # number of input tokens counted by the handler
        self.llm_calls = 0  # number of LLM calls

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ):
        """Run when LLM starts running."""
        self.llm_calls += 1
        self.input_tokens_counted = 0
        for p in prompts:
            self.input_tokens_counted += self.llm.get_num_tokens(p)

    def on_llm_end(self, response: LLMResult, **kwargs: Any):
        """Run when LLM completes running."""
        results = response.flatten()
        input_tokens_llm_reported = 0
        for r in results:
            if "token_usage" in r.llm_output:
                token_usage = r.llm_output["token_usage"]
                # typical token_usage: {'prompt_tokens': 252, 'completion_tokens': 4, 'total': 256}
                if "prompt_tokens" in token_usage:
                    input_tokens_llm_reported += token_usage["prompt_tokens"]

                if "completion_tokens" in token_usage:
                    self.output_tokens += token_usage["completion_tokens"]
                else:
                    self.output_tokens += self.llm.get_num_tokens(
                        r.generations[0][0].text
                    )

            else:
                # fallback to counting
                self.output_tokens += self.llm.get_num_tokens(r.generations[0][0].text)

        # override the input tokens count if we have a value from LLM response
        if input_tokens_llm_reported > 0:
            self.input_tokens += input_tokens_llm_reported
        else:
            self.input_tokens += self.input_tokens_counted


class TokenMetricUpdater:
    """A context manager to update token metrics in a callback handler.

    These metrics are updated:
    - llm_token_sent_total
    - llm_token_received_total
    - llm_calls_total

    Example usage:

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
    """

    def __init__(self, llm, provider: str, model: str):
        """Initialize the token counter context manager.

        Parameters:
            llm (LLM): the LLM instance
            provider (str): the provider name for labeling the metrics
            model (str): the model name for labeling the metrics
        """
        self.token_counter = GenericTokenCounter(llm=llm)
        self.provider = provider
        self.model = model

    def __enter__(self):
        """Initialize the token counter when entering the context."""
        return self.token_counter

    def __exit__(self, exc_type, exc_value, traceback):
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
