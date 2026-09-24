"""Mock for LLMLoader to be used in unit tests."""

from types import SimpleNamespace

from langchain_core.runnables import Runnable


class MockStructuredOutputLLM:
    """Mock structured-output model used by classifier tests."""

    async def ainvoke(self, *args, **kwargs):
        """Return a benign structured classifier decision."""
        return {"injectionDetected": False, "category": "none"}


class MockLLMLoader(Runnable):
    """Mock for LLMLoader."""

    def __init__(self, llm=None):
        """Store the selected LLM into object's attribute."""
        if llm is None:
            llm = SimpleNamespace()
            llm.provider = "mock_provider"
            llm.model = "mock_model"
        self.llm = llm

    def invoke(self, *args, **kwargs):
        """Mock model invoke."""
        if hasattr(args[0], "messages"):
            return args[0].messages[1]
        # If llm is a class (mock_langchain_interface), instantiate and invoke
        if isinstance(self.llm, type):
            return self.llm().invoke(args[0])
        return self.llm

    def with_structured_output(self, *args, **kwargs):
        """Return a separate structured-output mock."""
        return MockStructuredOutputLLM()

    @classmethod
    def bind_tools(cls, *args, **kwargs):
        """Mock bind tools."""
        return cls()

    async def astream(self, *args, **kwargs):
        """Return query result."""
        # yield input prompt/user query
        yield args[0].messages[1]


def mock_llm_loader(llm=None, expected_params=None):
    """Construct mock for load_llm."""

    def loader(*args, **kwargs):
        # if expected params are provided, check if (mocked) LLM loader
        # was called with expected parameters
        if expected_params is not None:
            assert expected_params == args, expected_params
        return MockLLMLoader(llm)

    return loader
