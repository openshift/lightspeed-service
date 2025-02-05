"""LLM function calling."""

import logging

from ols.src.tools.tools import tools_map


from typing import Any, AsyncGenerator, Optional

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import VectorStoreIndex

from ols import config
from ols.app.metrics import TokenMetricUpdater
from ols.app.models.models import RagChunk, SummarizerResponse
from ols.constants import RAG_CONTENT_LIMIT, GenericLLMParameters, MAX_ITERATIONS
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.prompts.prompts import SYSTEM_PROMPT_AGENT
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils.token_handler import TokenHandler

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


class OLSAgent(QueryHelper):
    """A class for OLS agentic flow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the OLSAgent."""
        super().__init__(*args, **kwargs)
        self._prepare_llm()
        self._get_system_prompt()
        self.verbose = config.ols_config.logging_config.app_log_level == logging.DEBUG

    def _prepare_llm(self) -> None:
        """Prepare the LLM configuration."""
        self.provider_config = config.llm_config.providers.get(self.provider)
        self.model_config = self.provider_config.models.get(self.model)
        self.generic_llm_params = {
            GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: self.model_config.parameters.max_tokens_for_response  # noqa: E501
        }
        self.bare_llm = self.llm_loader(
            self.provider, self.model, self.generic_llm_params
        )

    def _get_system_prompt(self) -> None:
        """Retrieve the system prompt."""
        # use system prompt from config if available otherwise use
        # default system prompt fine-tuned for the service
        if config.ols_config.system_prompt is not None:
            self.system_prompt = config.ols_config.system_prompt
        else:
            self.system_prompt = SYSTEM_PROMPT_AGENT
        logger.debug("System prompt: %s", self.system_prompt)

    def _prepare_prompt(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[str]] = None,
    ) -> tuple[ChatPromptTemplate, dict[str, str], list[RagChunk], bool]:
        """Summarize the given query based on the provided conversation context.

        Args:
            query: The query to be summarized.
            vector_index: Vector index to get RAG data/context.
            history: The history of the conversation (if available).

        Returns:
            A tuple containing the final prompt, input values, RAG chunks,
            and a flag for truncated history.
        """
        settings_string = (
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {self.verbose}"
        )
        logger.debug("call settings: %s", settings_string)

        if len(history) > 0:
            chat_history = []
            for h in history:
                if h.startswith("human: "):
                    chat_history.append(HumanMessage(content=h.removeprefix("human: ")))
                else:
                    chat_history.append(AIMessage(content=h.removeprefix("ai: ")))

        messages = []
        messages.append(SystemMessage(self.system_prompt))
        if len(history) > 0:
            messages = messages + chat_history
        messages.append(HumanMessage(query))
        return messages, {}, [], False


    def create_response(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[str]] = None,
    ) -> SummarizerResponse:
        """Create a response for the given query based on the provided conversation context."""
        final_prompt, llm_input_values, rag_chunks, truncated = self._prepare_prompt(
            query, vector_index, history
        )

        messages = final_prompt.copy()
        for i in range(MAX_ITERATIONS):
            llm_with_tools = self.bare_llm.bind_tools(tools_map.values())

            with TokenMetricUpdater(
                llm=self.bare_llm,
                provider=self.provider_config.type,
                model=self.model,
            ) as generic_token_counter:
                ai_msg = llm_with_tools.invoke(
                    messages,
                    config={"callbacks": [generic_token_counter]},
                )

            # if (not ai_msg.tool_calls) and (ai_msg.content):
            if ai_msg.response_metadata["finish_reason"] == "stop":
                response = ai_msg.content
                break            
            if i >= MAX_ITERATIONS - 2:
                ai_msg = self.bare_llm.invoke(messages)
                response = ai_msg.content
                break

            messages.append(ai_msg)
            for tool_call in ai_msg.tool_calls:
                selected_tool = tools_map[tool_call["name"].lower()]
                try:
                    tool_output = selected_tool.invoke(tool_call["args"])
                except Exception as e:
                    tool_output = f"error while executing {selected_tool}, error_msg:\n{str(e)}"
                print(
                    f"tool: {selected_tool}\n"
                    f"tool args: {tool_call['args']}\n"
                    f"tool_output:{tool_output}"
                )
                messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        return SummarizerResponse(
            response, rag_chunks, truncated, generic_token_counter.token_counter
        )
