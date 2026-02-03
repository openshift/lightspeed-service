"""Tool calling agent."""

# TODO: Refactor/rename this file
import asyncio
import json
import logging
import time
from collections.abc import Coroutine
from typing import Any, AsyncGenerator, Optional, TypeAlias

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.structured import StructuredTool
from llama_index.core.retrievers import BaseRetriever

from ols import config, constants
from ols.app.metrics import TokenMetricUpdater
from ols.app.metrics.token_counter import GenericTokenCounter
from ols.app.models.models import (
    RagChunk,
    StreamChunkType,
    StreamedChunk,
    SummarizerResponse,
)
from ols.constants import MAX_ITERATIONS, GenericLLMParameters
from ols.customize import reranker
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.query_helpers.history_support import prepare_history
from ols.src.query_helpers.query_helper import QueryHelper
from ols.src.tools.tools import execute_tool_calls
from ols.utils.mcp_utils import ClientHeaders, get_mcp_tools
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)

# Type aliases for clarity and reusability
ToolsList: TypeAlias = list[StructuredTool]


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


def run_async_safely(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async function safely."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "already running" in str(e).lower():
            logger.warning("Using existing event loop as one is already running")
            return asyncio.get_event_loop().run_until_complete(coro)
        raise


def _enrich_tool_call(
    tool_call: dict[str, Any],
    all_mcp_tools: ToolsList,
) -> dict[str, Any]:
    """Enrich a tool_call dict with metadata from the StructuredTool.

    Adds tool_meta and server_name so the UI can start preloading
    UI resources (e.g. MCP Apps iframes) before the tool result arrives.

    Args:
        tool_call: LLM-generated tool call dict (name, args, id).
        all_mcp_tools: Available tools (carry metadata from MCP servers).

    Returns:
        Enriched tool call dict. Original keys are preserved.
    """
    enriched: dict[str, Any] = {**tool_call}
    tool_obj = next((t for t in all_mcp_tools if t.name == tool_call.get("name")), None)
    if not tool_obj:
        return enriched

    tool_metadata = tool_obj.metadata or {}

    server_name = tool_metadata.get("mcp_server")
    if server_name:
        enriched["server_name"] = server_name

    tool_meta = tool_metadata.get("_meta")
    if tool_meta:
        enriched["tool_meta"] = tool_meta

    return enriched


def _build_tool_result_chunks(
    tool_calls: list[dict[str, Any]],
    tool_calls_messages: list[ToolMessage],
    all_mcp_tools: ToolsList,
    round_number: int,
) -> list[StreamedChunk]:
    """Build StreamedChunk objects for tool results with metadata.

    Args:
        tool_calls: LLM-generated tool call dicts (name, args, id).
        tool_calls_messages: Executed ToolMessage results.
        all_mcp_tools: Available tools (carry metadata from MCP servers).
        round_number: Current tool-calling round index.

    Returns:
        List of StreamedChunk objects with type "tool_result".
    """
    tool_id_to_name = {tc.get("id"): tc.get("name") for tc in tool_calls}
    tools_by_name = {t.name: t for t in all_mcp_tools}
    chunks: list[StreamedChunk] = []

    for tool_call_message in tool_calls_messages:
        was_truncated = tool_call_message.additional_kwargs.get("truncated", False)
        tool_status = "truncated" if was_truncated else tool_call_message.status

        tool_name = tool_id_to_name.get(tool_call_message.tool_call_id, "unknown")
        tool_obj = tools_by_name.get(tool_name)
        tool_metadata = (tool_obj.metadata or {}) if tool_obj else {}

        logger.debug(
            json.dumps(
                {
                    "event": "tool_result",
                    "tool_id": tool_call_message.tool_call_id,
                    "tool_name": tool_name,
                    "status": tool_call_message.status,
                    "truncated": was_truncated,
                    "has_meta": "_meta" in tool_metadata,
                    "output_snippet": str(tool_call_message.content)[:1000],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        tool_result_data: dict[str, Any] = {
            "id": tool_call_message.tool_call_id,
            "name": tool_name,
            "status": tool_status,
            "content": tool_call_message.content,
            "type": "tool_result",
            "round": round_number,
        }

        server_name = tool_metadata.get("mcp_server")
        if server_name:
            tool_result_data["server_name"] = server_name

        tool_meta = tool_metadata.get("_meta")
        if tool_meta:
            tool_result_data["tool_meta"] = tool_meta

        structured_content = tool_call_message.additional_kwargs.get(
            "structured_content"
        )
        if structured_content:
            tool_result_data["structured_content"] = structured_content

        chunks.append(
            StreamedChunk(type=StreamChunkType.TOOL_RESULT, data=tool_result_data)
        )

    return chunks


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    def __init__(
        self,
        *args: Any,
        user_token: Optional[str] = None,
        client_headers: ClientHeaders | None = None,
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

    @property
    def _has_mcp_tools(self) -> bool:
        """Check if MCP servers are configured."""
        return config.mcp_servers is not None and len(config.mcp_servers.servers) > 0

    async def _prepare_prompt_context(
        self,
        query: str,
        rag_retriever: Optional[BaseRetriever] = None,
    ) -> tuple[TokenHandler, list[RagChunk], int, int]:
        """Prepare token budget and RAG context for prompt construction.

        Args:
            query: The query to be summarized.
            rag_retriever: The retriever to get RAG data/context.

        Returns:
            A tuple containing token handler, RAG chunks, available token budget,
            and max tokens reserved for tools.
        """
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
            self._has_mcp_tools,
        ).generate_prompt(self.model)
        max_tokens_for_tools = (
            self.model_config.parameters.max_tokens_for_tools
            if self._has_mcp_tools
            else 0
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

        return token_handler, rag_chunks, available_tokens, max_tokens_for_tools

    def _build_final_prompt(
        self,
        query: str,
        history: list[BaseMessage],
        rag_chunks: list[RagChunk],
        token_handler: TokenHandler,
        max_tokens_for_tools: int,
    ) -> tuple[ChatPromptTemplate, dict[str, str]]:
        """Build final prompt from query, context, and prepared history.

        Args:
            query: The query to be summarized.
            history: Prepared conversation history to include in prompt.
            rag_chunks: Retrieved/truncated RAG chunks.
            token_handler: Token helper used for final prompt-size validation.
            max_tokens_for_tools: Reserved tool token budget for final validation.

        Returns:
            Final prompt and input values for LLM invocation.
        """
        rag_context = [rag_chunk.text for rag_chunk in rag_chunks]

        final_prompt, llm_input_values = GeneratePrompt(
            query,
            rag_context,
            history,
            self._system_prompt,
            self._has_mcp_tools,
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

        return final_prompt, llm_input_values

    async def _invoke_llm(
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        tools_map: ToolsList,
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
        llm_start_time = time.monotonic()
        try:
            async for chunk in chain.astream(
                input=llm_input_values,
                config={"callbacks": [token_counter]},
            ):
                yield chunk  # type: ignore [misc]
        except Exception:
            logger.error(
                "LLM invocation failed: provider=%s, model=%s, elapsed=%.2fs",
                self.provider,
                self.model,
                time.monotonic() - llm_start_time,
            )
            raise
        logger.info(
            "LLM invocation completed: provider=%s, model=%s, elapsed=%.2fs",
            self.provider,
            self.model,
            time.monotonic() - llm_start_time,
        )

    async def iterate_with_tools(  # noqa: C901  # pylint: disable=R0912,R0915
        self,
        messages: ChatPromptTemplate,
        max_rounds: int,
        llm_input_values: dict[str, str],
        token_counter: GenericTokenCounter,
        all_mcp_tools: ToolsList,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Iterate through multiple rounds of LLM invocation with tool calling.

        Args:
            messages: The initial messages
            max_rounds: Maximum number of tool calling rounds
            llm_input_values: Input values for the LLM
            token_counter: Counter for tracking token usage
            all_mcp_tools: List of MCP tools to use for tool calling

        Yields:
            StreamedChunk objects representing parts of the response
        """
        async with asyncio.timeout(constants.TOOL_CALL_ROUND_TIMEOUT * max_rounds):
            # Track cumulative token usage for tool outputs
            tool_tokens_used = 0
            max_tokens_for_tools = self.model_config.parameters.max_tokens_for_tools
            max_tokens_per_tool = (
                self.model_config.parameters.max_tokens_per_tool_output
            )
            token_handler = TokenHandler()

            # Account for tool definitions tokens (schemas sent to LLM)
            if all_mcp_tools:
                tool_definitions_text = json.dumps(
                    [
                        {
                            "name": t.name,
                            "description": t.description,
                            "schema": (
                                t.args_schema
                                if isinstance(t.args_schema, dict)
                                else (
                                    t.args_schema.model_json_schema()
                                    if t.args_schema is not None
                                    else {}
                                )
                            ),
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

                is_final_round = (not all_mcp_tools) or (i == max_rounds)
                logger.debug("Tool calling round %s (final: %s)", i, is_final_round)

                tool_call_chunks = []
                chunk_counter = 0
                stop_generation = False
                # invoke LLM and process response chunks
                async for chunk in self._invoke_llm(
                    messages,
                    llm_input_values,
                    tools_map=all_mcp_tools,
                    is_final_round=is_final_round,
                    token_counter=token_counter,
                ):
                    if stop_generation:
                        continue

                    # TODO: Temporary fix for fake-llm (load test) which gives
                    # output as string. Currently every method that we use gives us
                    # proper output, except fake-llm. We need to move to a different
                    # fake-llm (or custom fake-llm) which can handle streaming/non-streaming
                    # & tool calling and gives response not as string. Even below
                    # temp fix will fail for tool calling.
                    # (load test can be run with tool calling set to False till we
                    # have a permanent fix)
                    if isinstance(chunk, str):
                        yield StreamedChunk(type=StreamChunkType.TEXT, text=chunk)
                        break

                    if chunk.response_metadata.get("finish_reason") == "stop":  # type: ignore [attr-defined]
                        stop_generation = True
                        continue

                    # collect tool chunk or yield text
                    if getattr(chunk, "tool_call_chunks", None):
                        tool_call_chunks.append(chunk)
                    else:
                        if not skip_special_chunk(
                            chunk.content, chunk_counter, self.model, is_final_round
                        ):
                            # stream text chunks directly
                            yield StreamedChunk(
                                type=StreamChunkType.TEXT, text=chunk.content
                            )

                    chunk_counter += 1

                if stop_generation:
                    return

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
                        enriched = _enrich_tool_call(tool_call, all_mcp_tools)

                        logger.debug(
                            json.dumps(
                                {
                                    "event": "tool_call",
                                    "tool_name": enriched.get("name", "unknown"),
                                    "arguments": enriched.get("args", {}),
                                    "tool_id": enriched.get("id", "unknown"),
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                        )

                        yield StreamedChunk(
                            type=StreamChunkType.TOOL_CALL, data=enriched
                        )

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
                    messages.extend(tool_calls_messages)

                    # Track tokens used by tool outputs
                    for tool_call_message in tool_calls_messages:
                        content_tokens = token_handler.text_to_tokens(
                            str(tool_call_message.content)
                        )
                        tool_tokens_used += len(content_tokens)

                    for result_chunk in _build_tool_result_chunks(
                        tool_calls, tool_calls_messages, all_mcp_tools, i
                    ):
                        yield result_chunk

    async def generate_response(  # noqa: C901
        self,
        query: str,
        rag_retriever: Optional[BaseRetriever] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        skip_user_id_check: bool = False,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Generate a response for the given query.

        Args:
            query: The query to be answered
            rag_retriever: Retriever for RAG context
            user_id: User ID for retrieving conversation history
            conversation_id: Conversation ID for retrieving history
            skip_user_id_check: Whether to skip user ID validation

        Yields:
            StreamedChunk objects representing parts of the response
        """
        token_handler, rag_chunks, available_tokens, max_tokens_for_tools = (
            await self._prepare_prompt_context(query, rag_retriever)
        )
        history_event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        async def emit_history_event(event: dict[str, object]) -> None:
            await history_event_queue.put(event)

        def to_history_chunk(event: dict[str, object]) -> StreamedChunk | None:
            event_name = event.get("event")
            event_data = {k: v for k, v in event.items() if k != "event"}
            if event_name == StreamChunkType.HISTORY_COMPRESSION_START.value:
                return StreamedChunk(
                    type=StreamChunkType.HISTORY_COMPRESSION_START,
                    data=event_data,
                )
            if event_name == StreamChunkType.HISTORY_COMPRESSION_END.value:
                return StreamedChunk(
                    type=StreamChunkType.HISTORY_COMPRESSION_END,
                    data=event_data,
                )
            logger.warning("Unknown history event emitted: %s", event_name)
            return None

        history_task = asyncio.create_task(
            prepare_history(
                user_id=user_id,
                conversation_id=conversation_id,
                skip_user_id_check=skip_user_id_check,
                available_tokens=available_tokens,
                provider=self.provider,
                model=self.model,
                bare_llm=self.bare_llm,
                token_handler=token_handler,
                emit_event=emit_history_event,
            )
        )

        while not history_task.done():
            try:
                event = await asyncio.wait_for(history_event_queue.get(), timeout=0.05)
            except TimeoutError:
                continue
            chunk = to_history_chunk(event)
            if chunk is not None:
                yield chunk

        history, truncated = await history_task

        while not history_event_queue.empty():
            chunk = to_history_chunk(history_event_queue.get_nowait())
            if chunk is not None:
                yield chunk

        final_prompt, llm_input_values = self._build_final_prompt(
            query=query,
            history=history,
            rag_chunks=rag_chunks,
            token_handler=token_handler,
            max_tokens_for_tools=max_tokens_for_tools,
        )
        messages = final_prompt.model_copy()

        # Get all MCP tools (will handle tools_rag population and filtering)
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
            type=StreamChunkType.END,
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
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        skip_user_id_check: bool = False,
    ) -> SummarizerResponse:
        """Create a synchronous response for the given query.

        This method wraps the asynchronous generate_response method to provide
        a synchronous interface.

        Args:
            query: The query to be answered
            rag_retriever: Retriever for RAG context
            user_id: User ID for retrieving conversation history
            conversation_id: Conversation ID for retrieving history
            skip_user_id_check: Whether to skip user ID validation

        Returns:
            A SummarizerResponse object containing the complete response
        """

        async def drain_generate_response() -> SummarizerResponse:
            """Inner async function to collect all response chunks."""
            chunks = []
            response_end: dict[str, Any] = {}
            tool_calls = []
            tool_results = []
            async for chunk in self.generate_response(
                query, rag_retriever, user_id, conversation_id, skip_user_id_check
            ):
                match chunk.type:
                    case StreamChunkType.END:
                        response_end = chunk.data
                        break
                    case StreamChunkType.TOOL_CALL:
                        tool_calls.append(chunk.data)
                    case StreamChunkType.TOOL_RESULT:
                        tool_results.append(chunk.data)
                    case StreamChunkType.TEXT:
                        chunks.append(chunk.text)
                    case (
                        StreamChunkType.HISTORY_COMPRESSION_START
                        | StreamChunkType.HISTORY_COMPRESSION_END
                    ):
                        # Compression metadata is relevant only for streaming clients.
                        continue
                    case _:
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
