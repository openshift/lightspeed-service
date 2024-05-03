"""Mock for LLMLoader to be used in unit tests."""

from types import SimpleNamespace


class MockLLMLoader:
    """Mock for LLMLoader."""

    def __init__(self, llm=None):
        """Store the selected LLM into object's attribute."""
        if llm is None:
            llm = SimpleNamespace()
            llm.provider = "mock_provider"
            llm.model = "mock_model"
        self.llm = llm


def mock_llm_loader(llm=None, expected_params=None):
    """Construct mock for load_llm."""

    def loader(*args, **kwargs):
        # if expected params are provided, check if (mocked) LLM loader
        # was called with expected parameters
        if expected_params is not None:
            assert expected_params == args
        return MockLLMLoader(llm)

    return loader
