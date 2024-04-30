"""Mock for LangChainInterface to be used in unit tests."""

import json
from types import SimpleNamespace


def mock_langchain_interface(retval):
    """Construct mock for LangChainInterface."""

    class MockLangChainInterface:
        """Mock LangChainInterface class for testing.

        Example usage in a test:


        """

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return retval

        def invoke(self, question):
            """Return query result."""
            result = '{"content": "mock success for question: ' + question + '"}'
            return json.loads(result, object_hook=lambda d: SimpleNamespace(**d))

    return MockLangChainInterface
