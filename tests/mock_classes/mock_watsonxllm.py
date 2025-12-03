"""Mocked ChatWatsonx class to avoid accessing real Watsonx API."""

from langchain_core.language_models.llms import LLM


class ChatWatsonx(LLM):
    """Mocked ChatWatsonx class to avoid accessing real Watsonx API."""

    def __init__(self):
        """Initialize mocked ChatWatsonx."""

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
