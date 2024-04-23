"""Mocked WatsonxLLM class to avoid accessing real Watsonx API."""

from langchain.llms.base import LLM


class WatsonxLLM(LLM):
    """Mocked WatsonxLLM class to avoid accessing real Watsonx API."""

    def __init__(self):
        """Initialize mocked WatsonxLLM."""

    def _call(self):
        """Override abstract method from LLM abstract class."""
        return self

    def __call__(self, **kwargs):
        """Override abstract method from LLM abstract class."""
        return self

    def _llm_type(self):
        """Override abstract method from LLM abstract class."""
