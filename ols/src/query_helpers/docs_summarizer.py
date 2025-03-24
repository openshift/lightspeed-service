"""A class for summarizing documentation context."""

import logging
from typing import Any, AsyncGenerator, Optional

from langchain.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.base import BaseTool
from llama_index.core import VectorStoreIndex

from ols import config
from ols.app.metrics import TokenMetricUpdater
from ols.app.models.models import RagChunk, SummarizerResponse, TokenCounter, ToolCall
from ols.constants import MAX_ITERATIONS, RAG_CONTENT_LIMIT, GenericLLMParameters
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.query_helpers.query_helper import QueryHelper
from ols.src.tools.oc_cli import token_works_for_oc
from ols.src.tools.tools import execute_oc_tool_calls, oc_tools, parse_tool_request
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the QuestionValidator."""
        super().__init__(*args, **kwargs)
        self._prepare_llm()
        self.verbose = config.ols_config.logging_config.app_log_level == logging.DEBUG
        self._introspection_enabled = config.ols_config.introspection_enabled
        set_debug(False)

    def _prepare_llm(self) -> None:
        """Prepare the LLM configuration."""
        self.provider_config = config.llm_config.providers.get(self.provider)
        self.model_config = self.provider_config.models.get(self.model)
        self.generic_llm_params = {
            GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: self.model_config.parameters.max_tokens_for_response  # noqa: E501
        }
        self.bare_llm = self.llm_loader(
            self.provider, self.model, self.generic_llm_params, self.streaming
        )

    def _prepare_prompt(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[BaseMessage]] = None,
        tools_map: dict[str, BaseTool] = {},
    ) -> tuple[ChatPromptTemplate, dict[str, str], list[RagChunk], bool]:
        """Summarize the given query based on the provided conversation context.

        Args:
            query: The query to be summarized.
            vector_index: Vector index to get RAG data/context.
            history: The history of the conversation (if available).
            tools_map: Available tools

        Returns:
            A tuple containing the final prompt, input values, RAG chunks,
            and a flag for truncated history.
        """
        # if history is not provided, initialize to empty history
        if history is None:
            history = []

        settings_string = (
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {self.verbose}"
        )
        logger.debug("call settings: %s", settings_string)

        token_handler = TokenHandler()

        # Use sample text for context/history to get complete prompt
        # instruction. This is used to calculate available tokens.
        temp_prompt = GeneratePrompt(
            # Sample prompt's context/history must be re-structured for the given model,
            # to ensure the further right available token calculation.
            query,
            ["sample"],
            [AIMessage("sample")],
            self._system_prompt,
            tools_map,
        ).generate_prompt(self.provider_config.type, self.model)
        available_tokens = token_handler.calculate_and_check_available_tokens(
            temp_prompt,
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
        )

        # Retrieve RAG content
        if vector_index:
            retriever = vector_index.as_retriever(similarity_top_k=RAG_CONTENT_LIMIT)
            rag_chunks, available_tokens = token_handler.truncate_rag_context(
                retriever.retrieve(query), available_tokens
            )
        else:
            logger.warning("Proceeding without RAG content. Check start up messages.")
            rag_chunks = []
        rag_context = [rag_chunk.text for rag_chunk in rag_chunks]
        if len(rag_context) == 0:
            logger.debug("Using llm to answer the query without reference content")

        # Truncate history
        history, truncated = token_handler.limit_conversation_history(
            history or [], self.provider_config.type, self.model, available_tokens
        )

        final_prompt = GeneratePrompt(
            query,
            rag_context,
            history,
            self._system_prompt,
            tools_map,
        ).generate_prompt(self.provider_config.type, self.model)

        # Tokens-check: We trigger the computation of the token count
        # without care about the return value. This is to ensure that
        # the query is within the token limit.
        token_handler.calculate_and_check_available_tokens(
            final_prompt,
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
        )

        return final_prompt, rag_chunks, truncated

    def _invoke_llm(
        self,
        messages: list[BaseMessage] | str,
        tools_map: Optional[dict] = None,
        is_final_round: bool = False,
    ) -> tuple[AIMessage, TokenCounter]:
        """Invoke LLM with optional tools."""
        llm = (
            self.bare_llm
            if is_final_round or not tools_map
            else self.bare_llm.bind_tools(tools_map.values())
        )

        with TokenMetricUpdater(
            llm=self.bare_llm,
            provider=self.provider_config.type,
            model=self.model,
        ) as generic_token_counter:
            out = llm.invoke(
                messages,
                config={"callbacks": [generic_token_counter]},
            )
        return out, generic_token_counter.token_counter

    def _get_available_tools(self, user_token: Optional[str]) -> dict[str, BaseTool]:
        """Get available tools based on introspection and user token."""
        if not self._introspection_enabled:
            return {}

        logger.info("Introspection enabled - using default tools selection")

        if user_token and user_token.strip() and token_works_for_oc(user_token):
        # if True: 
            logger.info("Authenticated to 'oc' CLI; adding 'oc' tools")
            return oc_tools

        return {}

    def create_response(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[str]] = None,
        user_token: Optional[str] = None,
    ) -> SummarizerResponse:
        """Create a response for the given query based on the provided conversation context."""
        # TODO: for the specific tools type (oc) we need specific additional
        # context (user_token) to get the tools, we need to think how to make
        # it more generic to avoid low-level code changes with new tools type
        tools_map = self._get_available_tools(user_token)

        final_prompt, rag_chunks, truncated = self._prepare_prompt(
            query, vector_index, history, tools_map
        )
        messages = final_prompt #.copy()
        tool_calls = []

        # TODO: Tune system prompt
        # TODO: Handle context for each iteration
        # TODO: Handle tokens for tool response
        # TODO: Improvement for granite
        for i in range(MAX_ITERATIONS):

            # Force llm to give final response when introspection is disabled
            # or max iteration is reached
            is_final_round = (not self._introspection_enabled) or (
                i == MAX_ITERATIONS - 1
            )
            out, token_counter = self._invoke_llm(messages, tools_map, is_final_round)

            # Check if model is ready with final response
            # if (not ai_msg.tool_calls) and (ai_msg.content):

            if is_final_round or not (
                (out.response_metadata["finish_reason"] == "tool_calls")
                or out.content.startswith("<tool_call>")
            ):
                response = out.content
                break

            # Before we can add tool output to messages, we need to add
            # complete model response
            messages.append(out)

            tool_requests = parse_tool_request(
                out, self.provider_config.type, self.model
            )
            tool_calls.append(
                [ToolCall.from_langchain_tool_call(t) for t in tool_requests]
            )
            tool_calls_messages = execute_oc_tool_calls(
                tools_map, tool_requests, user_token
            )
            messages.extend(tool_calls_messages)

        print(f"--------------------------{i+1}")
        return SummarizerResponse(
            response, rag_chunks, truncated, token_counter, tool_calls
        )

    async def generate_response(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, SummarizerResponse]:
        """Generate a response for the given query based on the provided conversation context."""
        final_prompt, rag_chunks, truncated = self._prepare_prompt(
            query, vector_index, history
        )

        with TokenMetricUpdater(
            llm=self.bare_llm,
            provider=self.provider_config.type,
            model=self.model,
        ) as generic_token_counter:
            async for chunk in self.bare_llm.astream(
                final_prompt,
                config={"callbacks": [generic_token_counter]},
            ):
                # TODO: it is bad to have provider specific code here
                # the reason we have provider classes is to hide specific
                # implementation details there. But it requires expanding
                # the current providers interface, eg. to stream messages

                # openai returns an `AIMessageChunk` while Watsonx plain string
                chunk_content = chunk.content if hasattr(chunk, "content") else chunk
                if "<|endoftext|>" in chunk_content:
                    chunk_content = chunk_content.replace("<|endoftext|>", "")
                yield chunk_content

        # NOTE: we are not providing tool calls here as it is not currently
        # supported for streaming response
        yield SummarizerResponse("", rag_chunks, truncated, generic_token_counter.token_counter)  # type: ignore[misc]
