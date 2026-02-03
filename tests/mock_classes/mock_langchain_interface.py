"""Mock for LangChainInterface to be used in unit tests."""


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
            return retval

    return MockLangChainInterface
