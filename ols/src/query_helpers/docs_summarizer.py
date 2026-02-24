"""Tool calling agent."""

# TODO: Refactor/rename this file
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Coroutine, Optional, TypeAlias

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.structured import StructuredTool
from llama_index.core.retrievers import BaseRetriever

from ols import config, constants
from ols.app.metrics import TokenMetricUpdater
from ols.app.metrics.token_counter import GenericTokenCounter
from ols.app.models.models import ChunkType, RagChunk, StreamedChunk, SummarizerResponse
from ols.constants import MAX_ITERATIONS, GenericLLMParameters
from ols.customize import reranker
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.query_helpers.query_helper import QueryHelper
from ols.src.tools.tools import execute_tool_calls_stream
from ols.utils.mcp_utils import ClientHeaders, build_mcp_config, get_mcp_tools
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)

MIN_TOOL_EXECUTION_TOKENS = 100
ToolCallDefinition: TypeAlias = tuple[str, dict[str, object], StructuredTool]


@dataclass(slots=True)
class ToolTokenUsage:
    """Mutable holder for cumulative tool-token usage across helper boundaries."""

    used: int


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
) -> list[dict[str, object]]:
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


def run_async_safely(coro: Coroutine[Any, Any, Any]) -> Any:
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
        *args: object,
        user_token: Optional[str] = None,
        client_headers: ClientHeaders | None = None,
        streaming: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialize the DocsSummarizer.

        Args:
            user_token: Optional user authentication token for tool access
            client_headers: Optional client-provided MCP headers for authentication
            streaming: Whether this summarizer is used for the streaming endpoint
            *args: Additional positional arguments passed to the parent class
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(*args, **kwargs)
        self._prepare_llm()
        self.verbose = config.ols_config.logging_config.app_log_level == logging.DEBUG
        self.streaming = streaming

        # tools part
        self.client_headers = client_headers or {}
        self.user_token = user_token
        self.mcp_servers = build_mcp_config(
            config.mcp_servers.servers, self.user_token, self.client_headers
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
        tools_map: list[StructuredTool],
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

    def _resolve_tool_call_definitions(
        self,
        tool_calls: list[dict[str, object]],
        all_tools_dict: dict[str, StructuredTool],
        duplicate_tool_names: set[str],
    ) -> tuple[
        list[ToolCallDefinition],
        list[ToolMessage],
    ]:
        """Resolve LLM tool calls into executable definitions and skipped outcomes."""
        tool_call_definitions: list[ToolCallDefinition] = []
        skipped_tool_messages: list[ToolMessage] = []

        # Resolve each LLM-emitted tool call to the execution triple:
        # (tool_call_id, parsed_args, resolved StructuredTool).
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_id = str(tool_call.get("id", "unknown"))
            if not isinstance(tool_name, str):
                skipped_tool_messages.append(
                    ToolMessage(
                        content=(
                            "Tool call skipped: missing or invalid tool name. "
                            "Do not retry this exact tool call."
                        ),
                        status="error",
                        tool_call_id=tool_id,
                    )
                )
                continue
            if tool_name in duplicate_tool_names:
                logger.error(
                    "Tool '%s' is ambiguous (duplicate name across servers)",
                    tool_name,
                )
                skipped_tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Tool '{tool_name}' call skipped: ambiguous tool name "
                            "across servers. Do not retry this exact tool call."
                        ),
                        status="error",
                        tool_call_id=tool_id,
                    )
                )
                continue
            resolved_tool = all_tools_dict.get(tool_name)
            if resolved_tool is None:
                logger.error("Tool '%s' was requested but is unavailable", tool_name)
                skipped_tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Tool '{tool_name}' call skipped: tool is unavailable. "
                            "Do not retry this exact tool call."
                        ),
                        status="error",
                        tool_call_id=tool_id,
                    )
                )
                continue
            raw_args = tool_call.get("args", {})
            if raw_args is None:
                tool_args: dict[str, object] = {}
            elif isinstance(raw_args, dict):
                tool_args = {str(key): value for key, value in raw_args.items()}
            else:
                logger.error(
                    "Tool '%s' requested with invalid args type '%s'; skipping call",
                    tool_name,
                    type(raw_args).__name__,
                )
                skipped_tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Tool '{tool_name}' call skipped: invalid args type "
                            f"'{type(raw_args).__name__}'. Do not retry this exact tool call."
                        ),
                        status="error",
                        tool_call_id=tool_id,
                    )
                )
                continue
            tool_call_definitions.append(
                (
                    tool_id,
                    tool_args,
                    resolved_tool,
                )
            )

        return tool_call_definitions, skipped_tool_messages

    async def _collect_round_llm_chunks(
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        all_mcp_tools: list[StructuredTool],
        is_final_round: bool,
        token_counter: GenericTokenCounter,
        round_index: int,
    ) -> tuple[list[AIMessageChunk], list[StreamedChunk], bool]:
        """Collect one round of LLM chunks and streamed text output.

        Returns:
            Tuple of (tool_call_chunks, streamed_chunks, should_stop_iteration).
        """
        tool_call_chunks: list[AIMessageChunk] = []
        streamed_chunks: list[StreamedChunk] = []
        chunk_counter = 0
        try:
            async with asyncio.timeout(constants.TOOL_CALL_ROUND_TIMEOUT):
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
                        streamed_chunks.append(
                            StreamedChunk(type=ChunkType.TEXT, text=chunk)
                        )
                        break

                    if chunk.response_metadata.get("finish_reason") == "stop":  # type: ignore [attr-defined]
                        return tool_call_chunks, streamed_chunks, True

                    if getattr(chunk, "tool_call_chunks", None):
                        tool_call_chunks.append(chunk)
                    else:
                        if not skip_special_chunk(
                            chunk.content, chunk_counter, self.model, is_final_round
                        ):
                            streamed_chunks.append(
                                StreamedChunk(type=ChunkType.TEXT, text=chunk.content)
                            )

                    chunk_counter += 1
        except TimeoutError:
            logger.error(
                "Timed out waiting for LLM chunks in round %s after %s seconds",
                round_index,
                constants.TOOL_CALL_ROUND_TIMEOUT,
            )
            streamed_chunks.append(
                StreamedChunk(
                    type=ChunkType.TEXT,
                    text=(
                        "I could not complete this request in time. "
                        "Please try again."
                    ),
                )
            )
            return tool_call_chunks, streamed_chunks, True

        return tool_call_chunks, streamed_chunks, False

    def _tool_result_chunk_for_message(
        self,
        *,
        tool_call_message: ToolMessage,
        token_handler: TokenHandler,
        round_index: int,
    ) -> tuple[int, StreamedChunk]:
        """Convert a ToolMessage into a streamed tool_result chunk.

        Returns:
            A tuple of (token_count_for_tool_content, streamed_tool_result_chunk).
        """
        # Account emitted tool content against cumulative tool-token budget.
        content_tokens = token_handler.text_to_tokens(str(tool_call_message.content))
        content_token_count = len(content_tokens)

        # Normalize status for UI, with truncation taking precedence.
        was_truncated = tool_call_message.additional_kwargs.get("truncated", False)
        base_status = tool_call_message.status
        tool_status = "truncated" if was_truncated else base_status

        # Structured server-side log for observability/debugging.
        logger.info(
            json.dumps(
                {
                    "event": "tool_result",
                    "tool_id": tool_call_message.tool_call_id,
                    "status": tool_status,
                    "truncated": was_truncated,
                    "output_snippet": str(tool_call_message.content)[:1000],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        # Client-facing streamed tool_result payload for this round.
        tool_result_data: dict[str, Any] = {
            "id": tool_call_message.tool_call_id,
            "status": tool_status,
            "content": tool_call_message.content,
            "type": ChunkType.TOOL_RESULT.value,
            "round": round_index,
        }
        structured_content = tool_call_message.additional_kwargs.get(
            "structured_content"
        )
        if structured_content:
            tool_result_data["structured_content"] = structured_content

        return content_token_count, StreamedChunk(
            type=ChunkType.TOOL_RESULT, data=tool_result_data
        )

    async def _process_tool_calls_for_round(
        self,
        *,
        round_index: int,
        tool_call_chunks: list[AIMessageChunk],
        all_tools_dict: dict[str, StructuredTool],
        duplicate_tool_names: set[str],
        messages: ChatPromptTemplate,
        token_handler: TokenHandler,
        tool_token_usage: ToolTokenUsage,
        max_tokens_for_tools: int,
        max_tokens_per_tool: int,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Resolve, execute, and stream one round of tool calls."""
        tool_tokens_used = tool_token_usage.used

        # Finalize streamed chunks into complete tool calls.
        tool_calls = tool_calls_from_tool_calls_chunks(tool_call_chunks)
        tool_call_definitions, skipped_tool_messages = (
            self._resolve_tool_call_definitions(
                tool_calls,
                all_tools_dict,
                duplicate_tool_names,
            )
        )
        if not tool_call_definitions and not skipped_tool_messages:
            logger.warning(
                "No executable tools resolved from tool calls in round %s", round_index
            )
            tool_token_usage.used = tool_tokens_used
            return

        # Persist the AI tool-call message for the next LLM turn.
        ai_tool_call_message = AIMessage(content="", type="ai", tool_calls=tool_calls)
        messages.append(ai_tool_call_message)

        # Charge token budget for the assistant tool-call message itself, so
        # subsequent per-tool limits are computed from the remaining budget.
        ai_message_tokens = TokenHandler._get_token_count(
            token_handler.text_to_tokens(json.dumps(tool_calls))
        )
        tool_tokens_used += ai_message_tokens

        # Log and emit raw tool-call intents immediately so UI can show planned
        # actions before execution (and before potential approval prompts/results).
        for tool_call in tool_calls:
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
            yield StreamedChunk(type=ChunkType.TOOL_CALL, data=tool_call)

        # Derive per-tool execution cap from remaining global tool budget so we
        # do not exceed max_tokens_for_tools across this request.
        remaining_tool_budget = max_tokens_for_tools - tool_tokens_used
        effective_per_tool_limit = min(max_tokens_per_tool, remaining_tool_budget)
        logger.debug(
            "Tool budget: used=%d, remaining=%d, per_tool_limit=%d",
            tool_tokens_used,
            remaining_tool_budget,
            effective_per_tool_limit,
        )

        tool_calls_messages: list[ToolMessage] = []
        # Execute resolved tool calls and consume streamed execution events
        # (approval prompts + final tool results).
        if tool_call_definitions:
            # Enforce strict global tool budget. If model config uses a lower
            # max per-tool cap, lower the minimum accordingly so tools are not
            # permanently skipped due to configuration mismatch.
            minimum_required_tokens = min(
                MIN_TOOL_EXECUTION_TOKENS, max_tokens_per_tool
            )
            if effective_per_tool_limit < minimum_required_tokens:
                logger.warning(
                    "Skipping %d tool call(s) in round %s due to low remaining tool budget "
                    "(remaining=%d, minimum_required=%d)",
                    len(tool_call_definitions),
                    round_index,
                    remaining_tool_budget,
                    minimum_required_tokens,
                )
                # Emit synthetic tool results for skipped executions so client/UI
                # and conversation state remain consistent (one call -> one outcome).
                for tool_id, _tool_args, tool in tool_call_definitions:
                    tool_calls_messages.append(
                        ToolMessage(
                            content=(
                                f"Tool '{tool.name}' call skipped: remaining tool token budget "
                                f"({remaining_tool_budget}) is below minimum required "
                                f"({minimum_required_tokens}). "
                                "Do not retry this exact tool call."
                            ),
                            status="error",
                            tool_call_id=tool_id,
                        )
                    )
            else:
                async for execution_event in execute_tool_calls_stream(
                    tool_call_definitions,
                    effective_per_tool_limit,
                    streaming=self.streaming,
                ):
                    match execution_event.event:
                        case ChunkType.APPROVAL_REQUIRED:
                            # Forward approval requests immediately so the client can
                            # render an approval prompt and unblock execution.
                            yield StreamedChunk(
                                type=ChunkType.APPROVAL_REQUIRED,
                                data=execution_event.data,
                            )
                        case ChunkType.TOOL_RESULT:
                            # Keep ToolMessage objects for unified post-processing:
                            # conversation-state update, token accounting, and result streaming.
                            tool_calls_messages.append(execution_event.data)
                        case _:
                            logger.warning(
                                "Ignoring unexpected tool execution event: %s",
                                execution_event,
                            )

        # Merge synthetic skipped outcomes with real execution outcomes and
        # append all of them to conversation state for the next LLM turn.
        all_tool_messages = skipped_tool_messages + tool_calls_messages
        messages.extend(all_tool_messages)

        for tool_call_message in all_tool_messages:
            content_token_count, tool_result_chunk = (
                self._tool_result_chunk_for_message(
                    tool_call_message=tool_call_message,
                    token_handler=token_handler,
                    round_index=round_index,
                )
            )
            tool_tokens_used += content_token_count
            yield tool_result_chunk

        tool_token_usage.used = tool_tokens_used

    async def iterate_with_tools(  # noqa: C901
        self,
        messages: ChatPromptTemplate,
        max_rounds: int,
        llm_input_values: dict[str, str],
        token_counter: GenericTokenCounter,
        all_mcp_tools: list[StructuredTool],
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Iterate through multiple rounds of LLM invocation with tool calling.

        Args:
            messages: The initial messages
            max_rounds: Maximum number of tool calling rounds
            llm_input_values: Input values for the LLM
            token_counter: Counter for tracking token usage
            all_mcp_tools: All resolved MCP tools available for the request.

        Yields:
            StreamedChunk objects representing parts of the response
        """
        all_tools_dict: dict[str, StructuredTool] = {}
        duplicate_tool_names: set[str] = set()
        # Build a stable name->tool map once per request and disable ambiguous
        # duplicates so a tool name resolves to at most one executable tool.
        for tool in all_mcp_tools:
            if tool.name in all_tools_dict:
                duplicate_tool_names.add(tool.name)
            else:
                all_tools_dict[tool.name] = tool
        for tool_name in duplicate_tool_names:
            all_tools_dict.pop(tool_name, None)
        if duplicate_tool_names:
            logger.error(
                "Duplicate MCP tool names detected and disabled: %s",
                sorted(duplicate_tool_names),
            )

        # Track cumulative token usage for tool outputs
        tool_tokens_used = 0
        max_tokens_for_tools = self.model_config.parameters.max_tokens_for_tools
        max_tokens_per_tool = self.model_config.parameters.max_tokens_per_tool_output
        token_handler = TokenHandler()

        # Account for tool definitions tokens (schemas sent to LLM)
        if all_mcp_tools:
            tool_definitions_text = json.dumps(
                [
                    {"name": t.name, "description": t.description, "schema": t.args}
                    for t in all_mcp_tools
                ]
            )
            tool_definitions_tokens = TokenHandler._get_token_count(
                token_handler.text_to_tokens(tool_definitions_text)
            )
            tool_tokens_used += tool_definitions_tokens
            logger.debug("Tool definitions consume %d tokens", tool_definitions_tokens)

        # Tool calling in a loop
        for i in range(1, max_rounds + 1):
            # Final round must produce only the assistant answer (no more tool calls),
            # either because tools are disabled or we reached the max tool-call rounds.
            is_final_round = (not all_mcp_tools) or (i == max_rounds)
            logger.debug("Tool calling round %s (final: %s)", i, is_final_round)

            # Phase 1: collect one LLM round (text chunks + potential tool-call chunks).
            tool_call_chunks, round_streamed_chunks, should_stop_iteration = (
                await self._collect_round_llm_chunks(
                    messages=messages,
                    llm_input_values=llm_input_values,
                    all_mcp_tools=all_mcp_tools,
                    is_final_round=is_final_round,
                    token_counter=token_counter,
                    round_index=i,
                )
            )
            # Emit all text chunks produced during this LLM round.
            for streamed_chunk in round_streamed_chunks:
                yield streamed_chunk
            # Stop immediately when helper indicates terminal condition
            # (final answer reached or timeout fallback emitted).
            if should_stop_iteration:
                return

            # exit if this was the final round
            if is_final_round:
                break

            # tool calling part
            if tool_call_chunks:
                # Phase 2: resolve and execute tool calls for this round.
                tool_token_usage = ToolTokenUsage(used=tool_tokens_used)
                async for streamed_chunk in self._process_tool_calls_for_round(
                    round_index=i,
                    tool_call_chunks=tool_call_chunks,
                    all_tools_dict=all_tools_dict,
                    duplicate_tool_names=duplicate_tool_names,
                    messages=messages,
                    token_handler=token_handler,
                    tool_token_usage=tool_token_usage,
                    max_tokens_for_tools=max_tokens_for_tools,
                    max_tokens_per_tool=max_tokens_per_tool,
                ):
                    # Emit tool_call / approval_required / tool_result chunks.
                    yield streamed_chunk
                tool_tokens_used = tool_token_usage.used

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
        all_mcp_tools = await get_mcp_tools(query, self.user_token, self.client_headers)
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
                all_mcp_tools=all_mcp_tools,
            ):
                yield response

        yield StreamedChunk(
            type=ChunkType.END,
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

        This method drains the async response stream and aggregates it into
        a SummarizerResponse for non-streaming callers.
        """

        async def drain_generate_response() -> SummarizerResponse:
            """Collect all generated chunks into a single response object."""
            chunks: list[str] = []
            response_end: dict[str, object] = {}
            tool_calls: list[dict[str, object]] = []
            tool_results: list[dict[str, object]] = []
            async for chunk in self.generate_response(query, rag_retriever, history):
                match chunk.type:
                    case ChunkType.END:
                        response_end = chunk.data
                        break
                    case ChunkType.TOOL_CALL:
                        tool_calls.append(chunk.data)
                    case ChunkType.TOOL_RESULT:
                        tool_results.append(chunk.data)
                    case ChunkType.TEXT:
                        chunks.append(chunk.text)
                    case _:
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

        return run_async_safely(drain_generate_response())
