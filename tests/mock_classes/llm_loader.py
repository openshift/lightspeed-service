"""Mock for LLMLoader to be used in unit tests."""


class MockLLMLoader:
    """Mock for LLMLoader."""

    def __init__(self, llm=None):
        """Constructor that just stores the selected LLM into object's attribute."""
        self.llm = llm


def mock_llm_loader(llm=None):
    """Constructs mock for LLMLoader."""

    def loader(*args, **kwargs):
        return MockLLMLoader(llm)

    return loader
