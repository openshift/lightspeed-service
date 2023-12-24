class MockLLMLoader:
    def __init__(self, llm=None):
        self.llm = llm


def mock_llm_loader(llm=None):
    def loader(*args, **kwargs):
        return MockLLMLoader(llm)

    return loader
