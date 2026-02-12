"""Tool calling agent."""

# TODO: Refactor/rename this file
import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Callable, Optional

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.retrievers import BaseRetriever

from ols import config, constants
from ols.app.metrics import TokenMetricUpdater
from ols.app.metrics.token_counter import GenericTokenCounter
from ols.app.models.models import RagChunk, StreamedChunk, SummarizerResponse
from ols.constants import MAX_ITERATIONS, GenericLLMParameters
from ols.customize import reranker
from ols.src.mcp.tool_registry import get_tool_ui_metadata, is_model_visible
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.query_helpers.query_helper import QueryHelper
from ols.src.tools.tools import execute_tool_calls
from ols.utils.mcp_utils import build_mcp_config, gather_mcp_tools
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


def skip_special_chunk(
    chunk_text: str,
    chunk_counter: int,
    model_name: str,
    final_round: bool,
) -> bool:
    """Handle special chunk."""
    # Handle granite tool call identifier chunks.
    # This is a workaround as until these chunks are recieved, it is
    # difficult to associate these with tool call (with langchain).
    # We can implement more sophisticated solution but may not be worth doing.
    if constants.ModelFamily.GRANITE in model_name and not final_round:
        return (
            (chunk_counter == 0 and chunk_text == "")
            or (chunk_counter == 1 and chunk_text == "<")
            or (chunk_counter == 2 and chunk_text == "tool")
            or (chunk_counter == 3 and chunk_text == "_")
            or (chunk_counter == 4 and chunk_text == "call")
            or (chunk_counter == 5 and chunk_text == ">")
        )
    return False


def tool_calls_from_tool_calls_chunks(
    tool_calls_chunks: list[AIMessageChunk],
) -> list[dict[str, Any]]:
    """Extract complete tool calls from a series of tool call chunks.

    The LLM streams tool calls in partial chunks that need to be combined to form
    complete tool call objects.

    Args:
        tool_calls_chunks: List of AIMessageChunk objects containing partial tool calls

    Returns:
        List of complete tool call dictionaries
    """
    # there is a langchain magic to put these messages together to create
    # a final list of tool calls with full args - concat messages.
    response = AIMessageChunk(content="")
    for chunk in tool_calls_chunks:
        response += chunk  # type: ignore [assignment]
    return response.tool_calls


def run_async_safely(coro: Callable) -> Any:
    """Run an async function safely."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "already running" in str(e).lower():
            logger.warning("Using existing event loop as one is already running")
            return asyncio.get_event_loop().run_until_complete(coro)
        raise


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    def __init__(
        self,
        *args: Any,
        user_token: Optional[str] = None,
        client_headers: dict[str, dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DocsSummarizer.

        Args:
            user_token: Optional user authentication token for tool access
            client_headers: Optional client-provided MCP headers for authentication
            *args: Additional positional arguments passed to the parent class
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(*args, **kwargs)
        self._prepare_llm()
        self.verbose = config.ols_config.logging_config.app_log_level == logging.DEBUG

        # tools part
        self.client_headers = client_headers or {}
        self.user_token = user_token
        self.mcp_servers = build_mcp_config(
            config.mcp_servers, self.user_token, self.client_headers
        )

        if self.mcp_servers:
            logger.info("MCP servers provided: %s", list(self.mcp_servers.keys()))
            self._tool_calling_enabled = True
        else:
            logger.debug("No MCP servers provided, tool calling is disabled")
            self._tool_calling_enabled = False

        set_debug(self.verbose)

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
        rag_retriever: Optional[BaseRetriever] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> tuple[ChatPromptTemplate, dict[str, str], list[RagChunk], bool]:
        """Summarize the given query based on the provided conversation context.

        Args:
            query: The query to be summarized.
            rag_retriever: The retriever to get RAG data/context.
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
            self._tool_calling_enabled,
        ).generate_prompt(self.model)
        max_tokens_for_tools = (
            self.model_config.parameters.max_tokens_for_tools if self.mcp_servers else 0
        )
        available_tokens = token_handler.calculate_and_check_available_tokens(
            temp_prompt.format(**temp_prompt_input),
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
            max_tokens_for_tools,
        )

        # Retrieve RAG content
        if rag_retriever:
            retrieved_nodes = rag_retriever.retrieve(query)
            logger.info("Retrieved %d documents from indexes", len(retrieved_nodes))

            retrieved_nodes = reranker.rerank(retrieved_nodes)
            logger.info("After reranking: %d documents", len(retrieved_nodes))

            # Logging top retrieved candidates with scores
            for i, node in enumerate(retrieved_nodes[:5]):
                logger.info(
                    "Retrieved doc #%d: title='%s', url='%s', index='%s', score=%.4f",
                    i + 1,
                    node.metadata.get("title", "unknown"),
                    node.metadata.get("docs_url", "unknown"),
                    node.metadata.get("index_origin", "unknown"),
                    node.get_score(raise_error=False),
                )

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
            self._tool_calling_enabled,
        ).generate_prompt(self.model)

        # Tokens-check: We trigger the computation of the token count
        # without care about the return value. This is to ensure that
        # the query is within the token limit.
        token_handler.calculate_and_check_available_tokens(
            final_prompt.format(**llm_input_values),
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
            max_tokens_for_tools,
        )

        return final_prompt, llm_input_values, rag_chunks, truncated

    async def _invoke_llm(
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        tools_map: list[Callable],
        is_final_round: bool,
        token_counter: GenericTokenCounter,
    ) -> AsyncGenerator[AIMessageChunk, None]:
        """Invoke the LLM with optional tools.

        Args:
            messages: The prompt template with messages
            llm_input_values: Input values for the prompt
            tools_map: Map of available tools
            is_final_round: Flag indicating if this is the final round of tool calling
            token_counter: Counter for tracking token usage

        Yields:
            AIMessageChunk objects from the LLM response stream
        """
        logger.debug("provided %s tools", len(tools_map))
        # determine whether to use tools based on round and availability
        llm = (
            self.bare_llm
            if is_final_round or not tools_map
            else self.bare_llm.bind_tools(tools_map)
        )

        # create and execute the chain
        chain = messages | llm
        async for chunk in chain.astream(
            input=llm_input_values,
            config={"callbacks": [token_counter]},
        ):
            yield chunk  # type: ignore [misc]

    async def iterate_with_tools(  # noqa: C901
        self,
        messages: ChatPromptTemplate,
        max_rounds: int,
        llm_input_values: dict[str, str],
        token_counter: GenericTokenCounter,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Iterate through multiple rounds of LLM invocation with tool calling.

        Args:
            messages: The initial messages
            max_rounds: Maximum number of tool calling rounds
            tools_map: Map of available tools
            llm_input_values: Input values for the LLM
            token_counter: Counter for tracking token usage

        Yields:
            StreamedChunk objects representing parts of the response
        """
        async with asyncio.timeout(constants.TOOL_CALL_ROUND_TIMEOUT * max_rounds):
            all_mcp_tools = await gather_mcp_tools(self.mcp_servers)

            # Filter out app-only tools (visibility: ["app"]) before LLM binding
            app_only = [t.name for t in all_mcp_tools if not is_model_visible(t.name)]
            if app_only:
                logger.info("Excluding %d app-only tools from LLM: %s", len(app_only), app_only)
                all_mcp_tools = [t for t in all_mcp_tools if is_model_visible(t.name)]

            # Track cumulative token usage for tool outputs
            tool_tokens_used = 0
            max_tokens_for_tools = self.model_config.parameters.max_tokens_for_tools
            max_tokens_per_tool = (
                self.model_config.parameters.max_tokens_per_tool_output
            )
            token_handler = TokenHandler()

            # Account for tool definitions tokens (schemas sent to LLM)
            if all_mcp_tools:

                def _safe_tool_args(tool: Any) -> dict:
                    """Get tool args safely; some tools lack 'properties' in their schema."""
                    try:
                        return tool.args
                    except KeyError:
                        return {}

                tool_definitions_text = json.dumps(
                    [
                        {
                            "name": t.name,
                            "description": t.description,
                            "schema": _safe_tool_args(t),
                        }
                        for t in all_mcp_tools
                    ]
                )
                tool_definitions_tokens = TokenHandler._get_token_count(
                    token_handler.text_to_tokens(tool_definitions_text)
                )
                tool_tokens_used += tool_definitions_tokens
                logger.debug(
                    "Tool definitions consume %d tokens", tool_definitions_tokens
                )

            # Tool calling in a loop
            for i in range(1, max_rounds + 1):

                is_final_round = (not self._tool_calling_enabled) or (i == max_rounds)
                logger.debug("Tool calling round %s (final: %s)", i, is_final_round)

                tool_call_chunks = []
                chunk_counter = 0
                # invoke LLM and process response chunks
                async for chunk in self._invoke_llm(
                    messages,
                    llm_input_values,
                    tools_map=all_mcp_tools,
                    is_final_round=is_final_round,
                    token_counter=token_counter,
                ):
                    # TODO: Temporary fix for fake-llm (load test) which gives
                    # output as string. Currently every method that we use gives us
                    # proper output, except fake-llm. We need to move to a different
                    # fake-llm (or custom fake-llm) which can handle streaming/non-streaming
                    # & tool calling and gives response not as string. Even below
                    # temp fix will fail for tool calling.
                    # (load test can be run with tool calling set to False till we
                    # have a permanent fix)
                    if isinstance(chunk, str):
                        yield StreamedChunk(type="text", text=chunk)
                        break

                    # check if LLM has finished generating
                    if chunk.response_metadata.get("finish_reason") == "stop":  # type: ignore [attr-defined]
                        return

                    # collect tool chunk or yield text
                    if getattr(chunk, "tool_call_chunks", None):
                        tool_call_chunks.append(chunk)
                    else:
                        if not skip_special_chunk(
                            chunk.content, chunk_counter, self.model, is_final_round
                        ):
                            # stream text chunks directly
                            yield StreamedChunk(type="text", text=chunk.content)

                    chunk_counter += 1

                # exit if this was the final round
                if is_final_round:
                    break

                # tool calling part
                if tool_call_chunks:
                    # assess tool calls and add to messages
                    tool_calls = tool_calls_from_tool_calls_chunks(tool_call_chunks)
                    ai_tool_call_message = AIMessage(
                        content="", type="ai", tool_calls=tool_calls
                    )
                    messages.append(ai_tool_call_message)

                    # Count tokens used by the AIMessage with tool calls
                    ai_message_tokens = TokenHandler._get_token_count(
                        token_handler.text_to_tokens(json.dumps(tool_calls))
                    )
                    tool_tokens_used += ai_message_tokens

                    for tool_call in tool_calls:
                        # Log tool call in JSON format
                        logger.info(
                            json.dumps(
                                {
                                    "event": "tool_call",
                                    "tool_name": tool_call.get("name", "unknown"),
                                    "arguments": tool_call.get("args", {}),
                                    "tool_id": tool_call.get("id", "unknown"),
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )

                        yield StreamedChunk(type="tool_call", data=tool_call)

                    # Calculate remaining budget for tools
                    remaining_tool_budget = max_tokens_for_tools - tool_tokens_used
                    # Use the smaller of per-tool limit or remaining budget
                    effective_per_tool_limit = min(
                        max_tokens_per_tool, remaining_tool_budget
                    )

                    logger.debug(
                        "Tool budget: used=%d, remaining=%d, per_tool_limit=%d",
                        tool_tokens_used,
                        remaining_tool_budget,
                        effective_per_tool_limit,
                    )

                    # execute tools and add to messages
                    tool_calls_messages = await execute_tool_calls(
                        tool_calls,
                        all_mcp_tools,
                        max(
                            effective_per_tool_limit, 100
                        ),  # Minimum 100 tokens per tool
                    )

                    tool_id_to_name = {
                        tc.get("id"): tc.get("name") for tc in tool_calls
                    }

                    # Build LLM context messages: use placeholders for MCP App
                    # tools to save tokens (the UI shows the full data instead).
                    # The original tool_calls_messages are NOT mutated.
                    llm_messages = []
                    for tool_call_message in tool_calls_messages:
                        tool_name = tool_id_to_name.get(
                            tool_call_message.tool_call_id, "unknown"
                        )
                        ui_metadata = get_tool_ui_metadata(tool_name)
                        if ui_metadata and ui_metadata.resource_uri:
                            llm_messages.append(
                                ToolMessage(
                                    content=(
                                        f"[Data displayed in interactive UI:"
                                        f" {tool_name}]"
                                    ),
                                    status=tool_call_message.status,
                                    tool_call_id=tool_call_message.tool_call_id,
                                )
                            )
                        else:
                            llm_messages.append(tool_call_message)

                    messages.extend(llm_messages)

                    # Track tokens used by tool outputs (after placeholder replacement)
                    for llm_message in llm_messages:
                        content_tokens = token_handler.text_to_tokens(
                            str(llm_message.content)
                        )
                        tool_tokens_used += len(content_tokens)

                    # Stream tool results to UI (original content, not placeholders)
                    for tool_call_message in tool_calls_messages:
                        was_truncated = tool_call_message.additional_kwargs.get(
                            "truncated", False
                        )
                        tool_status = (
                            "truncated" if was_truncated else tool_call_message.status
                        )

                        tool_name = tool_id_to_name.get(
                            tool_call_message.tool_call_id, "unknown"
                        )
                        ui_metadata = get_tool_ui_metadata(tool_name)

                        logger.info(
                            json.dumps(
                                {
                                    "event": "tool_result",
                                    "tool_id": tool_call_message.tool_call_id,
                                    "tool_name": tool_name,
                                    "status": tool_call_message.status,
                                    "truncated": was_truncated,
                                    "has_ui": ui_metadata is not None,
                                    "output_snippet": str(
                                        tool_call_message.content
                                    )[:1000],
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )

                        tool_result_data: dict[str, Any] = {
                            "id": tool_call_message.tool_call_id,
                            "name": tool_name,
                            "status": tool_status,
                            "is_error": tool_status == "error",
                            "content": tool_call_message.content,
                            "type": "tool_result",
                            "round": i,
                        }

                        if ui_metadata and ui_metadata.resource_uri:
                            tool_result_data["ui_resource_uri"] = (
                                ui_metadata.resource_uri
                            )
                            tool_result_data["server_name"] = ui_metadata.server_name

                        # Forward structured_content from tool result if available
                        structured = tool_call_message.additional_kwargs.get(
                            "structured_content"
                        ) or tool_call_message.additional_kwargs.get(
                            "structuredContent"
                        )
                        if structured:
                            tool_result_data["structured_content"] = structured

                        yield StreamedChunk(
                            type="tool_result",
                            data=tool_result_data,
                        )

    async def generate_response(
        self,
        query: str,
        rag_retriever: Optional[BaseRetriever] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Generate a response for the given query.

        Args:
            query: The query to be answered
            rag_retriever: Retriever for RAG context
            history: Optional conversation history

        Yields:
            StreamedChunk objects representing parts of the response
        """
        final_prompt, llm_input_values, rag_chunks, truncated = self._prepare_prompt(
            query, rag_retriever, history
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
        rag_retriever: Optional[BaseRetriever] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> SummarizerResponse:
        """Create a synchronous response for the given query.

        This method wraps the asynchronous generate_response method to provide
        a synchronous interface.

        Args:
            query: The query to be answered
            rag_retriever: Retriever for RAG context
            history: Optional conversation history

        Returns:
            A SummarizerResponse object containing the complete response
        """

        async def drain_generate_response() -> SummarizerResponse:
            """Inner async function to collect all response chunks."""
            chunks = []
            response_end: dict[str, Any] = {}
            tool_calls = []
            tool_results = []
            async for chunk in self.generate_response(query, rag_retriever, history):
                if chunk.type == "end":
                    response_end = chunk.data
                    break
                if chunk.type == "tool_call":
                    tool_calls.append(chunk.data)
                elif chunk.type == "tool_result":
                    tool_results.append(chunk.data)
                elif chunk.type == "text":
                    chunks.append(chunk.text)
                else:
                    # this "can't" happen as we control what chunk types
                    # are yielded in the generator directly
                    msg = f"Unknown chunk type: {chunk.type}"
                    logger.warning(msg)
                    raise ValueError(msg)

            return SummarizerResponse(
                response="".join(chunks),
                rag_chunks=response_end.get("rag_chunks", []),
                history_truncated=response_end.get("truncated", False),
                token_counter=response_end.get("token_counter", None),
                tool_calls=tool_calls,
                tool_results=tool_results,
            )

        # TODO: if we define the non-streaming endpoint as async, we don't
        # need to handle any of this, we would just await it
        return run_async_safely(drain_generate_response())
