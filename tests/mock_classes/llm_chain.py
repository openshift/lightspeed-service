"""Mock for LLMChain to be used in unit tests."""


def mock_llm_chain(retval):
    """Construct mock for LLMChain."""

    class MockLLMChain:
        """Mock LLMChain class for testing.

        Example usage in a test:

            from tests.mock_classes.llm_chain import mock_llm_chain
            ml = mock_llm_chain({"text": "default"})

            @patch("ols.src.query_helpers.yes_no_classifier.LLMChain", new=ml)
            def test_xyz():

            or within test function or test method:
            with patch("ols.src.query_helpers.question_validator.LLMChain", new=ml):
                some test steps

        """

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return retval

        def invoke(self, question, **kwargs):
            """Return query result with the chat history to verify llm inputs."""
            result = {"text": question, "history": kwargs.get("chat_history")}
            return result

    return MockLLMChain
