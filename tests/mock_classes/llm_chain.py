def mock_llm_chain(retval):
    class MockLLMChain:
        """Mock LLMChain class for testing

        Example usage in a test:

            from tests.mock_classes.llm_chain import mock_llm_chain
            ml = mock_llm_chain({"text": "default"})
            monkeypatch.setattr(src.query_helpers.yes_no_classifier, "LLMChain", ml)

        """

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return retval

    return MockLLMChain
