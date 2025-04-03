"""Tool calling agent."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Literal, Optional

from langchain.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import VectorStoreIndex

from ols import config
from ols.app.metrics import TokenMetricUpdater
from ols.app.metrics.token_counter import GenericTokenCounter
from ols.app.models.models import RagChunk, SummarizerResponse
from ols.constants import MAX_ITERATIONS, RAG_CONTENT_LIMIT, GenericLLMParameters
from ols.customize import reranker
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.query_helpers.query_helper import QueryHelper
from ols.src.tools.tools import execute_oc_tool_calls, get_available_tools
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


@dataclass
class StreamedChunk:
    """Streamed chunk dataclass."""

    type: Literal["text", "tool_call", "tool_result", "end"]
    text: str = ""
    data: dict = field(default_factory=dict)


def tool_calls_from_tool_calls_chunks(
    tool_calls_chunks: list[AIMessageChunk],
) -> list[dict]:
    """Extract tool calls from tool calls chunks."""
    # The tool args are streamed partially like this:
    # ...'tool_calls': [{...'function': {'arguments': '', 'name': 'oc_get'},...
    # ...'tool_calls': [{...'function': {'arguments': '{"', 'name': None},...
    # ...'tool_calls': [{...'function': {'arguments': 'command', 'name': None...
    # ...'tool_calls': [{...'function': {'arguments': '_args', 'name': None...
    # ...'tool_calls': [{...'function': {'arguments': '":["', 'name': None},...
    # ...'tool_calls': [{...'function': {'arguments': 'names', 'name': None...
    # ...'tool_calls': [{...'function': {'arguments': 'paces', 'name': None...
    # ...'tool_calls': [{...'function': {'arguments': '"]', 'name': None},...
    # ...'tool_calls': [{...'function': {'arguments': '}', 'name': None},...
    # there is a langchain magic to put these messages together to create
    # a final list of tool calls with full args - concat messages.

    response = AIMessageChunk(content="")
    for chunk in tool_calls_chunks:
        response += chunk  # type: ignore [assignment]
    return response.tool_calls


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    def __init__(
        self, *args: Any, user_token: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize the QuestionValidator."""
        super().__init__(*args, **kwargs)
        self._prepare_llm()
        self.verbose = config.ols_config.logging_config.app_log_level == logging.DEBUG

        # tools part
        self._introspection_enabled = config.ols_config.introspection_enabled
        self.tools_map = get_available_tools(self._introspection_enabled, user_token)
        self.user_token = user_token

        # disabled - leaks token to logs when set to True
        set_debug(False)

    def _prepare_llm(self) -> None:
        """Prepare the LLM configuration."""
        self.provider_config = config.llm_config.providers.get(self.provider)
        self.model_config = self.provider_config.models.get(self.model)
        self.generic_llm_params = {
            GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: self.model_config.parameters.max_tokens_for_response  # noqa: E501
        }
        self.bare_llm = self.llm_loader(
            self.provider,
            self.model,
            self.generic_llm_params,
        )

    def _prepare_prompt(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[BaseMessage]] = None,
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
        # if history is not provided, initialize to empty history
        if history is None:
            history = []

        token_handler = TokenHandler()

        # Use sample text for context/history to get complete prompt
        # instruction. This is used to calculate available tokens.
        temp_prompt, temp_prompt_input = GeneratePrompt(
            # Sample prompt's context/history must be re-structured for the given model,
            # to ensure the further right available token calculation.
            query,
            ["sample"],
            [AIMessage("sample")],
            self._system_prompt,
            self._introspection_enabled,
        ).generate_prompt(self.model)
        available_tokens = token_handler.calculate_and_check_available_tokens(
            temp_prompt.format(**temp_prompt_input),
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
        )

        # Retrieve RAG content
        if vector_index:
            retriever = vector_index.as_retriever(similarity_top_k=RAG_CONTENT_LIMIT)
            retrieved_nodes = retriever.retrieve(query)
            retrieved_nodes = reranker.rerank(retrieved_nodes)
            rag_chunks, available_tokens = token_handler.truncate_rag_context(
                retrieved_nodes, available_tokens
            )
        else:
            logger.warning("Proceeding without RAG content. Check start up messages.")
            rag_chunks = []
        rag_context = [rag_chunk.text for rag_chunk in rag_chunks]
        if len(rag_context) == 0:
            logger.debug("Using llm to answer the query without reference content")

        # Truncate history
        history, truncated = token_handler.limit_conversation_history(
            history or [], available_tokens
        )

        final_prompt, llm_input_values = GeneratePrompt(
            query,
            rag_context,
            history,
            self._system_prompt,
            self._introspection_enabled,
        ).generate_prompt(self.model)

        # Tokens-check: We trigger the computation of the token count
        # without care about the return value. This is to ensure that
        # the query is within the token limit.
        token_handler.calculate_and_check_available_tokens(
            final_prompt.format(**llm_input_values),
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
        )

        return final_prompt, llm_input_values, rag_chunks, truncated

    async def _invoke_llm(
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        tools_map: dict,
        is_final_round: bool,
        token_counter: GenericTokenCounter,
    ) -> AsyncGenerator[AIMessageChunk, None]:
        """Invoke LLM with optional tools."""
        # Force llm to give final response when introspection is disabled
        # or max iteration is reached
        llm = (
            self.bare_llm
            if is_final_round or not tools_map
            else self.bare_llm.bind_tools(tools_map.values())
        )
        chain = messages | llm

        async for chunk in chain.astream(
            input=llm_input_values,
            config={"callbacks": [token_counter]},
        ):
            yield chunk  # type: ignore [misc]

    async def iterate_with_tools(
        self,
        messages: ChatPromptTemplate,
        max_rounds: int,
        tools_map: dict[str, Callable],
        llm_input_values: dict[str, str],
        token_counter: GenericTokenCounter,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Iterate with tools.

        Args:
            messages: The initial messages.
            max_rounds: The maximum rounds.
            tools_map: The tools map.
            llm_input_values: The LLM input values.
            token_counter: The token counter.
        """
        for i in range(max_rounds):

            is_final_round = (not self._introspection_enabled) or (i == max_rounds - 1)
            tool_calls_chunks = []
            streaming_tool_calls = False

            async for chunk in self._invoke_llm(
                messages,
                llm_input_values,
                tools_map=tools_map,
                is_final_round=is_final_round,
                token_counter=token_counter,
            ):
                if chunk.response_metadata.get("finish_reason") == "stop":  # type: ignore [attr-defined]
                    return

                if getattr(chunk, "tool_calls", None) or streaming_tool_calls:
                    # tools request - drain the full stream
                    streaming_tool_calls = True
                    tool_calls_chunks.append(chunk)
                else:
                    # text streaming
                    yield StreamedChunk(type="text", text=chunk.content)

            if is_final_round:
                break

            # tool calling part
            if tool_calls_chunks:
                # assess tool calls and add to messages
                tool_calls = tool_calls_from_tool_calls_chunks(tool_calls_chunks)
                messages.append(AIMessage(content="", type="ai", tool_calls=tool_calls))
                for tool_call in tool_calls:
                    yield StreamedChunk(type="tool_call", data=tool_call)

                # execute tools and add to messages
                tool_calls_messages = execute_oc_tool_calls(
                    tools_map, tool_calls, self.user_token
                )
                messages.extend(tool_calls_messages)
                for tool_call_message in tool_calls_messages:
                    yield StreamedChunk(
                        type="tool_result",
                        data={
                            "id": tool_call_message.tool_call_id,
                            "status": tool_call_message.status,
                            "content": tool_call_message.content,
                            "type": "tool_result",
                            "round": i,
                        },
                    )

    async def generate_response(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Generate a response for the given query.

        Args:
            query: The query to be answered.
            vector_index: Vector index to get RAG data/context.
            history: The history of the conversation.
        """
        final_prompt, llm_input_values, rag_chunks, truncated = self._prepare_prompt(
            query, vector_index, history
        )
        messages = final_prompt.model_copy()

        with TokenMetricUpdater(
            llm=self.bare_llm,
            provider=self.provider_config.type,
            model=self.model,
        ) as token_counter:
            async for response in self.iterate_with_tools(
                messages=messages,
                max_rounds=MAX_ITERATIONS,
                tools_map=self.tools_map,
                token_counter=token_counter,
                llm_input_values=llm_input_values,
            ):
                yield response

        yield StreamedChunk(
            type="end",
            data={
                "rag_chunks": rag_chunks,
                "truncated": truncated,
                "token_counter": token_counter.token_counter,
            },
        )

    def create_response(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> SummarizerResponse:
        """Create a response for the given query.

        Args:
            query: The query to be answered.
            vector_index: Vector index to get RAG data/context.
            history: The history of the conversation.
        """

        async def drain_generate_response() -> SummarizerResponse:
            chunks = []
            response_end: dict = {}
            tools_calls = []
            tools_results = []
            async for chunk in self.generate_response(query, vector_index, history):
                if chunk.type == "end":
                    response_end = chunk.data
                    break
                if chunk.type == "tool_call":
                    tools_calls.append(chunk.data)
                elif chunk.type == "tool_result":
                    tools_results.append(chunk.data)
                elif chunk.type == "text":
                    chunks.append(chunk.text)
                else:
                    raise NotImplementedError("this would be case for granite for sure")

            return SummarizerResponse(
                response="".join(chunks),
                rag_chunks=response_end.get("rag_chunks"),
                history_truncated=response_end.get("truncated"),
                token_counter=response_end.get("token_counter"),
                tools_calls=tools_calls,
                tools_results=tools_results,
            )

        # run the async function synchronously
        # TODO: will this fails if another event loop is running?
        return asyncio.run(drain_generate_response())
