"""fake provider implementation."""

import logging
from typing import Any

import requests
from langchain_community.llms import FakeListLLM
from langchain_community.llms.fake import FakeStreamingListLLM
from langchain_core.language_models.llms import LLM

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_FAKE)
class FakeProvider(LLMProvider):
    """Fake provider for testing purposes."""

    stream: bool = False
    mcp_tool_call: bool = False
    response: str = "This is a preconfigured fake response."
    chunks: int = len(response)
    sleep: float = 0.1

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        if self.provider_config.fake_provider_config is not None:
            fake_provider_config = self.provider_config.fake_provider_config
            if fake_provider_config.stream:
                self.stream = fake_provider_config.stream
            if fake_provider_config.mcp_tool_call:
                self.mcp_tool_call = fake_provider_config.mcp_tool_call
            if fake_provider_config.response:
                self.response = fake_provider_config.response
            if fake_provider_config.chunks:
                self.chunks = fake_provider_config.chunks
            if fake_provider_config.sleep:
                self.sleep = fake_provider_config.sleep

        return {
            "stream": self.stream,
            "mcp_tool_call": self.mcp_tool_call,
            "response": self.response,
            "chunks": self.chunks,
            "sleep": self.sleep,
        }

    def load(self) -> LLM:
        """Load the fake LLM with dynamic response property."""

        def bind_tools(tools: Any, *args: Any, **kwargs: Any) -> LLM:
            return llm

        def dynamic_response() -> str:
            if self.mcp_tool_call:
                mcp_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "resources_list",
                        "arguments": {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "namespace": "openshift-lightspeed",
                        },
                    },
                }

                try:
                    mcp_response = requests.post(
                        "http://localhost:8080/mcp",
                        json=mcp_payload,
                        timeout=60,
                    )
                    mcp_response.raise_for_status()
                    mcp_result = mcp_response.json()
                except Exception as e:
                    mcp_result = f"Failed to fetch MCP result: {e!s}"
                    return self.response

                return f"{self.response}\n\nMCP Result:\n{mcp_result}"
            return self.response

        if self.stream:
            final_response = dynamic_response()
            i = self.chunks // (len(final_response) + 1)
            j = self.chunks % (len(final_response) + 1)
            response = ((final_response + " ") * i) + final_response[0:j]
            llm = FakeStreamingListLLM(responses=[response], sleep=self.sleep)
        else:
            llm = FakeListLLM(responses=[dynamic_response()])

        object.__setattr__(llm, "bind_tools", bind_tools)

        return llm
