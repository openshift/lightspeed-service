def mock_langchain_interface(retval):
    class MockLangChainInterface:
        """
        Mock LangChainInterface class for testing

        Example usage in a test:


        """

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return retval

    return MockLangChainInterface
