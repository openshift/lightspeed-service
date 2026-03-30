"""Tool calling agent."""

# TODO: Refactor/rename this file
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Coroutine, NamedTuple, Optional, TypeAlias

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
from ols.constants import GenericLLMParameters
from ols.src.auth.k8s import CLUSTER_VERSION_UNAVAILABLE, K8sClientSingleton
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.query_helpers.history_support import prepare_history
from ols.src.query_helpers.query_helper import QueryHelper
from ols.src.tools.tools import enforce_tool_token_budget, execute_tool_calls
from ols.utils.mcp_utils import ClientHeaders, build_mcp_config, get_mcp_tools
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)

MIN_TOOL_EXECUTION_TOKENS = 100
ToolCallDefinition: TypeAlias = tuple[str, dict[str, object], StructuredTool]


@dataclass(slots=True)
class ToolTokenUsage:
    """Mutable holder for cumulative tool-token usage across helper boundaries."""

    used: int


class RoundLLMResult(NamedTuple):
    """Result of one LLM collection round from ``_collect_round_llm_chunks``."""

    tool_call_chunks: list[AIMessageChunk]
    all_chunks: list[AIMessageChunk]
    streamed_chunks: list[StreamedChunk]
    should_stop: bool


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
        self._cluster_version = (
            K8sClientSingleton.get_cluster_version()
            if self._mode == constants.QueryMode.TROUBLESHOOTING
            else CLUSTER_VERSION_UNAVAILABLE
        )

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
            self._tool_calling_enabled,
            self._mode,
            self._cluster_version,
        ).generate_prompt(self.model)
        max_tokens_for_tools = (
            self.model_config.max_tokens_for_tools if self.mcp_servers else 0
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
            self._tool_calling_enabled,
            self._mode,
            self._cluster_version,
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
        if not tools_map:
            llm = self.bare_llm
        elif is_final_round:
            # Responses API dumps tool args as text when tools are unbound;
            # tool_choice="none" prevents this while strict=False avoids
            # langchain-ai/langchain#35837 (Responses API defaults strict=True).
            llm = self.bare_llm.bind_tools(tools_map, tool_choice="none", strict=False)
        else:
            llm = self.bare_llm.bind_tools(tools_map, strict=False)

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

    def _streamed_chunks_from_list_content(
        self,
        content: list[Any],
        chunk_counter: int,
        is_final_round: bool,
    ) -> list[StreamedChunk]:
        """Extract text and reasoning StreamedChunks from list-format content blocks.

        See ``GenericTokenCounter._count_list_content_tokens`` for the block
        schema documentation.
        """
        result: list[StreamedChunk] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            match block.get("type"):
                case "text":
                    text = block.get("text", "")
                    if text and not skip_special_chunk(
                        text, chunk_counter, self.model, is_final_round
                    ):
                        result.append(
                            StreamedChunk(type=StreamChunkType.TEXT, text=text)
                        )
                case "reasoning":
                    for part in block.get("summary", []):
                        if not isinstance(part, dict):
                            continue
                        text = part.get("text", "")
                        if text:
                            result.append(
                                StreamedChunk(type=StreamChunkType.REASONING, text=text)
                            )
        return result

    async def _collect_round_llm_chunks(
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        all_mcp_tools: list[StructuredTool],
        is_final_round: bool,
        token_counter: GenericTokenCounter,
        round_index: int,
    ) -> RoundLLMResult:
        """Collect one round of LLM chunks and streamed text output."""
        tool_call_chunks: list[AIMessageChunk] = []
        all_chunks: list[AIMessageChunk] = []
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
                            StreamedChunk(type=StreamChunkType.TEXT, text=chunk)
                        )
                        break

                    if chunk.response_metadata.get("finish_reason") == "stop":  # type: ignore [attr-defined]
                        return RoundLLMResult(
                            tool_call_chunks,
                            all_chunks,
                            streamed_chunks,
                            should_stop=True,
                        )

                    all_chunks.append(chunk)

                    if getattr(chunk, "tool_call_chunks", None):
                        tool_call_chunks.append(chunk)
                    elif isinstance(chunk.content, str):
                        if chunk.content and not skip_special_chunk(
                            chunk.content, chunk_counter, self.model, is_final_round
                        ):
                            streamed_chunks.append(
                                StreamedChunk(
                                    type=StreamChunkType.TEXT, text=chunk.content
                                )
                            )
                    elif isinstance(chunk.content, list):
                        streamed_chunks.extend(
                            self._streamed_chunks_from_list_content(
                                chunk.content, chunk_counter, is_final_round
                            )
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
                    type=StreamChunkType.TEXT,
                    text=(
                        "I could not complete this request in time. "
                        "Please try again."
                    ),
                )
            )
            return RoundLLMResult(
                tool_call_chunks, all_chunks, streamed_chunks, should_stop=True
            )

        return RoundLLMResult(
            tool_call_chunks, all_chunks, streamed_chunks, should_stop=False
        )

    @staticmethod
    def _enrich_with_tool_metadata(
        data: dict[str, Any],
        tool: Optional[StructuredTool],
    ) -> None:
        """Add MCP server metadata to a tool_call or tool_result dict in-place.

        Adds ``server_name`` and ``tool_meta`` when available so the UI can
        associate events with their originating MCP server and preload
        resources (e.g. MCP Apps iframes).
        """
        if tool is None:
            return
        tool_metadata = tool.metadata or {}
        server_name = tool_metadata.get("mcp_server")
        if server_name:
            data["server_name"] = server_name
        tool_meta = tool_metadata.get("_meta")
        if tool_meta:
            data["tool_meta"] = tool_meta

    def _tool_result_chunk_for_message(
        self,
        *,
        tool_call_message: ToolMessage,
        tool_name: str,
        tool: Optional[StructuredTool],
        token_handler: TokenHandler,
        round_index: int,
    ) -> tuple[int, StreamedChunk]:
        """Convert a ToolMessage into a streamed tool_result chunk.

        Returns:
            A tuple of (token_count_for_tool_content, streamed_tool_result_chunk).
        """
        content_tokens = token_handler.text_to_tokens(str(tool_call_message.content))
        content_token_count = TokenHandler._get_token_count(content_tokens)

        was_truncated = tool_call_message.additional_kwargs.get("truncated", False)
        base_status = tool_call_message.status
        tool_status = "truncated" if was_truncated else base_status
        has_meta = bool(
            isinstance(tool.metadata, dict) and tool.metadata.get("_meta")
            if tool is not None
            else False
        )

        logger.info(
            json.dumps(
                {
                    "event": "tool_result",
                    "tool_id": tool_call_message.tool_call_id,
                    "tool_name": tool_name,
                    "status": tool_status,
                    "truncated": was_truncated,
                    "has_meta": has_meta,
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
            "type": StreamChunkType.TOOL_RESULT.value,
            "round": round_index,
        }
        structured_content = tool_call_message.additional_kwargs.get(
            "structured_content"
        )
        if structured_content:
            tool_result_data["structured_content"] = structured_content
        self._enrich_with_tool_metadata(tool_result_data, tool)

        return content_token_count, StreamedChunk(
            type=StreamChunkType.TOOL_RESULT, data=tool_result_data
        )

    async def _process_tool_calls_for_round(
        self,
        *,
        round_index: int,
        tool_call_chunks: list[AIMessageChunk],
        all_chunks: list[AIMessageChunk],
        all_tools_dict: dict[str, StructuredTool],
        duplicate_tool_names: set[str],
        messages: ChatPromptTemplate,
        token_handler: TokenHandler,
        tool_token_usage: ToolTokenUsage,
        max_tokens_for_tools: int,
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

        # Accumulate the full AI message (reasoning + tool calls) so reasoning
        # context is preserved between rounds per OpenAI's "Keeping reasoning
        # items in context" guidance.
        if all_chunks:
            accumulated = all_chunks[0]
            for c in all_chunks[1:]:
                accumulated += c  # type: ignore [assignment]
            ai_tool_call_message = AIMessage(
                content=accumulated.content,
                tool_calls=tool_calls,
                additional_kwargs=accumulated.additional_kwargs,
            )
        else:
            ai_tool_call_message = AIMessage(
                content="", type="ai", tool_calls=tool_calls
            )
        messages.append(ai_tool_call_message)

        # Charge token budget for the assistant tool-call message itself, so
        # subsequent per-tool limits are computed from the remaining budget.
        ai_content_text = (
            json.dumps(ai_tool_call_message.content)
            if isinstance(ai_tool_call_message.content, list)
            else str(ai_tool_call_message.content)
        )
        ai_message_tokens = TokenHandler._get_token_count(
            token_handler.text_to_tokens(ai_content_text + json.dumps(tool_calls))
        )
        tool_tokens_used += ai_message_tokens

        # Build a mapping from tool_call_id -> tool_name for result enrichment.
        tool_id_to_name: dict[str, str] = {
            str(tc.get("id", "")): str(tc.get("name", "unknown")) for tc in tool_calls
        }

        # Log and emit tool-call intents enriched with MCP metadata so the UI
        # can associate calls with their server and preload resources.
        for tool_call in tool_calls:
            enriched: dict[str, Any] = {**tool_call}
            tool_name = str(tool_call.get("name", "unknown"))
            self._enrich_with_tool_metadata(enriched, all_tools_dict.get(tool_name))
            logger.info(
                json.dumps(
                    {
                        "event": "tool_call",
                        "tool_name": tool_name,
                        "arguments": tool_call.get("args", {}),
                        "tool_id": tool_call.get("id", "unknown"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            yield StreamedChunk(type=StreamChunkType.TOOL_CALL, data=enriched)

        remaining_tool_budget = max_tokens_for_tools - tool_tokens_used
        logger.debug(
            "Tool budget: used=%d, remaining=%d",
            tool_tokens_used,
            remaining_tool_budget,
        )

        tool_calls_messages: list[ToolMessage] = []
        # Execute resolved tool calls and consume streamed execution events
        # (approval prompts + final tool results).
        if tool_call_definitions:
            if remaining_tool_budget < MIN_TOOL_EXECUTION_TOKENS:
                logger.warning(
                    "Skipping %d tool call(s) in round %s due to low remaining tool budget "
                    "(remaining=%d, minimum_required=%d)",
                    len(tool_call_definitions),
                    round_index,
                    remaining_tool_budget,
                    MIN_TOOL_EXECUTION_TOKENS,
                )
                # Emit synthetic tool results for skipped executions so client/UI
                # and conversation state remain consistent (one call -> one outcome).
                for tool_id, _tool_args, tool in tool_call_definitions:
                    tool_calls_messages.append(
                        ToolMessage(
                            content=(
                                f"Tool '{tool.name}' call skipped: remaining tool token budget "
                                f"({remaining_tool_budget}) is below minimum required "
                                f"({MIN_TOOL_EXECUTION_TOKENS}). "
                                "Do not retry this exact tool call."
                            ),
                            status="error",
                            tool_call_id=tool_id,
                        )
                    )
            else:
                tool_calls_dicts = [
                    {"name": tool.name, "args": tool_args, "id": tool_id}
                    for tool_id, tool_args, tool in tool_call_definitions
                ]
                tool_calls_messages = await execute_tool_calls(
                    tool_calls_dicts,
                    list(all_tools_dict.values()),
                    remaining_tool_budget,
                )

        # Merge synthetic skipped outcomes with real execution outcomes and
        # append all of them to conversation state for the next LLM turn.
        all_tool_messages = skipped_tool_messages + tool_calls_messages
        if remaining_tool_budget > 0:
            all_tool_messages = enforce_tool_token_budget(
                all_tool_messages, remaining_tool_budget
            )
        messages.extend(all_tool_messages)

        for tool_call_message in all_tool_messages:
            tool_name = tool_id_to_name.get(tool_call_message.tool_call_id, "unknown")
            content_token_count, tool_result_chunk = (
                self._tool_result_chunk_for_message(
                    tool_call_message=tool_call_message,
                    tool_name=tool_name,
                    tool=all_tools_dict.get(tool_name),
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
        max_tokens_for_tools = self.model_config.max_tokens_for_tools
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
            round_result = await self._collect_round_llm_chunks(
                messages=messages,
                llm_input_values=llm_input_values,
                all_mcp_tools=all_mcp_tools,
                is_final_round=is_final_round,
                token_counter=token_counter,
                round_index=i,
            )
            for streamed_chunk in round_result.streamed_chunks:
                yield streamed_chunk
            if round_result.should_stop:
                return

            # exit if this was the final round
            if is_final_round:
                break

            # tool calling part
            if round_result.tool_call_chunks:
                # Phase 2: resolve and execute tool calls for this round.
                tool_token_usage = ToolTokenUsage(used=tool_tokens_used)
                # No outer timeout here — each MCP server enforces its own
                # configured timeout at the transport layer.
                try:
                    async for streamed_chunk in self._process_tool_calls_for_round(
                        round_index=i,
                        tool_call_chunks=round_result.tool_call_chunks,
                        all_chunks=round_result.all_chunks,
                        all_tools_dict=all_tools_dict,
                        duplicate_tool_names=duplicate_tool_names,
                        messages=messages,
                        token_handler=token_handler,
                        tool_token_usage=tool_token_usage,
                        max_tokens_for_tools=max_tokens_for_tools,
                    ):
                        yield streamed_chunk
                except Exception:
                    logger.exception("Error executing tool calls in round %s", i)
                    yield StreamedChunk(
                        type=StreamChunkType.TEXT,
                        text=(
                            "I could not complete this request. " "Please try again."
                        ),
                    )
                    return
                tool_tokens_used = tool_token_usage.used

    async def generate_response(
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
        history: list[BaseMessage] = []
        truncated = False
        async for item in prepare_history(
            user_id=user_id,
            conversation_id=conversation_id,
            skip_user_id_check=skip_user_id_check,
            available_tokens=available_tokens,
            provider=self.provider,
            model=self.model,
            bare_llm=self.bare_llm,
            token_handler=token_handler,
        ):
            if isinstance(item, StreamedChunk):
                yield item
            else:
                history, truncated = item

        final_prompt, llm_input_values = self._build_final_prompt(
            query=query,
            history=history,
            rag_chunks=rag_chunks,
            token_handler=token_handler,
            max_tokens_for_tools=max_tokens_for_tools,
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
                max_rounds=self._get_max_iterations(),
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

    def _get_max_iterations(self) -> int:
        """Return configured max rounds for tool-calling loop."""
        return config.ols_config.max_iterations

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
                    case StreamChunkType.REASONING:
                        pass
                    case StreamChunkType.TEXT:
                        chunks.append(chunk.text)
                    case (
                        StreamChunkType.HISTORY_COMPRESSION_START
                        | StreamChunkType.HISTORY_COMPRESSION_END
                    ):
                        continue
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
