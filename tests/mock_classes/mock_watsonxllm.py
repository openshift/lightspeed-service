"""Mocked WatsonxLLM class to avoid accessing real Watsonx API."""

from langchain.llms.base import LLM


class WatsonxLLM(LLM):
    """Mocked WatsonxLLM class to avoid accessing real Watsonx API."""

    def __init__(self):
        """Initialize mocked WatsonxLLM."""

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        """Override abstract method from LLM abstract class."""
        return "foo"

    def __call__(self, prompt=None, stop=None, tags=None, **kwargs):
        """Override abstract method from LLM abstract class."""
        return self

    @property
    def _llm_type(self):
        """Override abstract method from LLM abstract class."""
        return "baz"
