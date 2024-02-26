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


def mock_llm_loader(llm=None):
    """Construct mock for load_llm."""

    def loader(*args, **kwargs):
        return MockLLMLoader(llm)

    return loader
