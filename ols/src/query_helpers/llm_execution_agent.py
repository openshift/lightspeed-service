"""Tool calling agent with iterative tool execution loop."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, TypeAlias

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.structured import StructuredTool

from ols import constants
from ols.app.metrics import TokenMetricUpdater
from ols.app.metrics.token_counter import GenericTokenCounter
from ols.app.models.config import ModelConfig
from ols.app.models.models import RagChunk, StreamChunkType, StreamedChunk
from ols.src.tools.tools import enforce_tool_token_budget, execute_tool_calls_stream
from ols.utils.token_handler import TokenBudgetTracker, TokenCategory

if TYPE_CHECKING:
    from ols.src.tools.offloaded_content import OffloadManager

logger = logging.getLogger(__name__)

MIN_TOOL_EXECUTION_TOKENS = 100
ToolCallDefinition: TypeAlias = tuple[str, dict[str, object], StructuredTool]


@dataclass(slots=True)
class ToolTokenUsage:
    """Mutable holder for cumulative tool-token usage across helper boundaries."""

    used: int


def log_tool_loop_iteration(
    tracker: TokenBudgetTracker,
    round_index: int,
    max_rounds: int,
    outcome: str,
) -> None:
    """Log token budget after one tool-loop LLM iteration (and optional tools)."""
    logger.info(
        "Tool loop iteration %s/%s outcome=%s. %s",
        round_index,
        max_rounds,
        outcome,
        tracker.summary(round_index),
    )


@dataclass
class RoundLLMResult:
    """Mutable accumulator for one LLM collection round.

    Populated by ``_collect_round_llm_chunks`` while it yields streamed chunks.
    """

    tool_call_chunks: list[AIMessageChunk] = field(default_factory=list)
    all_chunks: list[AIMessageChunk] = field(default_factory=list)
    should_stop: bool = False


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


class LLMExecutionAgent:
    """Agent that drives the iterative LLM + tool-calling loop."""

    def __init__(
        self,
        bare_llm: BaseChatModel,
        model: str,
        provider: str,
        provider_type: str,
        model_config: ModelConfig,
        streaming: bool,
        token_budget_tracker: TokenBudgetTracker,
    ) -> None:
        """Initialize the tool calling agent.

        Args:
            bare_llm: The configured LLM instance.
            model: Model name (used for logging and granite workarounds).
            provider: Provider name (used for logging).
            provider_type: Provider type string (used for metrics).
            model_config: Model configuration (token budgets).
            streaming: Whether the request uses the streaming endpoint.
            token_budget_tracker: Shared per-request token budget tracker.
        """
        self.bare_llm = bare_llm
        self.model = model
        self.provider = provider
        self.provider_type = provider_type
        self.model_config = model_config
        self.streaming = streaming
        self._tracker = token_budget_tracker

    async def execute(
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        max_rounds: int,
        all_mcp_tools: list[StructuredTool],
        rag_chunks: list[RagChunk],
        truncated: bool,
        *,
        tool_definitions_tokens: int | None = None,
        offload_manager: "OffloadManager | None" = None,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Run the LLM + tool-calling loop with metrics tracking.

        Args:
            messages: The prepared prompt template.
            llm_input_values: Input values for the prompt.
            max_rounds: Maximum number of tool calling rounds.
            all_mcp_tools: All resolved MCP tools available for the request.
            rag_chunks: RAG chunks used for the response (passed through to END).
            truncated: Whether conversation history was truncated.
            tool_definitions_tokens: When set, charge this value for tool definitions
                without re-tokenizing; must match a prior count of the same payload.
            offload_manager: Optional manager for offloading large tool outputs.

        Yields:
            StreamedChunk objects representing parts of the response,
            ending with a StreamChunkType.END chunk.
        """
        with TokenMetricUpdater(
            llm=self.bare_llm,
            provider=self.provider_type,
            model=self.model,
        ) as token_counter:
            async for chunk in self._iterate_with_tools(
                messages=messages,
                max_rounds=max_rounds,
                token_counter=token_counter,
                llm_input_values=llm_input_values,
                all_mcp_tools=all_mcp_tools,
                tool_definitions_tokens=tool_definitions_tokens,
                offload_manager=offload_manager,
            ):
                yield chunk
        yield StreamedChunk(
            type=StreamChunkType.END,
            data={
                "rag_chunks": rag_chunks,
                "truncated": truncated,
                "token_counter": token_counter.token_counter,
            },
        )

    @staticmethod
    def _dedupe_tools_by_name(
        all_mcp_tools: list[StructuredTool],
    ) -> tuple[dict[str, StructuredTool], set[str]]:
        """Build a name→tool map and disable ambiguous duplicate names."""
        all_tools_dict: dict[str, StructuredTool] = {}
        duplicate_tool_names: set[str] = set()
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
        return all_tools_dict, duplicate_tool_names

    def _charge_tool_definitions_tokens(
        self,
        all_mcp_tools: list[StructuredTool],
        tool_definitions_tokens: int | None,
    ) -> None:
        """Charge ``TOOL_DEFINITIONS`` when the request includes MCP tools."""
        if not all_mcp_tools:
            return
        if tool_definitions_tokens is not None:
            defs_tokens = tool_definitions_tokens
        else:
            tool_definitions_text = json.dumps(
                [
                    {"name": t.name, "description": t.description, "schema": t.args}
                    for t in all_mcp_tools
                ]
            )
            defs_tokens = self._tracker.count_tokens(tool_definitions_text)
        self._tracker.charge(TokenCategory.TOOL_DEFINITIONS, defs_tokens)
        logger.debug("Tool definitions consume %d tokens", defs_tokens)

    async def _iterate_with_tools(
        self,
        messages: ChatPromptTemplate,
        max_rounds: int,
        llm_input_values: dict[str, str],
        token_counter: GenericTokenCounter,
        all_mcp_tools: list[StructuredTool],
        *,
        tool_definitions_tokens: int | None = None,
        offload_manager: "OffloadManager | None" = None,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Iterate through multiple rounds of LLM invocation with tool calling.

        Args:
            messages: The initial messages
            max_rounds: Maximum number of tool calling rounds
            llm_input_values: Input values for the LLM
            token_counter: Counter for tracking token usage
            all_mcp_tools: All resolved MCP tools available for the request.
            tool_definitions_tokens: When set, charge this value for tool definitions
                without re-tokenizing; must match a prior count of the same payload.
            offload_manager: Optional manager for offloading large tool outputs.

        Yields:
            StreamedChunk objects representing parts of the response
        """
        all_tools_dict, duplicate_tool_names = self._dedupe_tools_by_name(all_mcp_tools)
        self._charge_tool_definitions_tokens(all_mcp_tools, tool_definitions_tokens)

        for i in range(1, max_rounds + 1):
            is_final_round = (not all_mcp_tools) or (i == max_rounds)
            logger.debug("Tool calling round %s (final: %s)", i, is_final_round)

            round_result = RoundLLMResult()
            async for chunk in self._collect_round_llm_chunks(
                messages=messages,
                llm_input_values=llm_input_values,
                all_mcp_tools=all_mcp_tools,
                is_final_round=is_final_round,
                token_counter=token_counter,
                round_index=i,
                result=round_result,
            ):
                yield chunk
            if round_result.should_stop:
                log_tool_loop_iteration(self._tracker, i, max_rounds, "llm_stream_stop")
                return

            if is_final_round:
                log_tool_loop_iteration(self._tracker, i, max_rounds, "final_round")
                break

            if not round_result.tool_call_chunks:
                log_tool_loop_iteration(
                    self._tracker, i, max_rounds, "model_finished_without_tools"
                )
                break

            try:
                async for streamed_chunk in self._process_tool_calls_for_round(
                    round_index=i,
                    tool_call_chunks=round_result.tool_call_chunks,
                    all_chunks=round_result.all_chunks,
                    all_tools_dict=all_tools_dict,
                    duplicate_tool_names=duplicate_tool_names,
                    messages=messages,
                    offload_manager=offload_manager,
                ):
                    yield streamed_chunk

                if (
                    offload_manager is not None
                    and offload_manager.has_offloaded_content
                    and not offload_manager.retrieval_tools_registered
                ):
                    retrieval_tools = offload_manager.build_retrieval_tools()
                    for rt in retrieval_tools:
                        all_mcp_tools.append(rt)
                        all_tools_dict[rt.name] = rt
                    offload_manager.mark_retrieval_tools_registered()
                    logger.info(
                        "Registered offload retrieval tools: %s",
                        [rt.name for rt in retrieval_tools],
                    )
            except Exception:
                log_tool_loop_iteration(
                    self._tracker, i, max_rounds, "tool_execution_failed"
                )
                logger.exception("Error executing tool calls in round %s", i)
                yield StreamedChunk(
                    type=StreamChunkType.TEXT,
                    text="I could not complete this request. Please try again.",
                )
                return
            log_tool_loop_iteration(
                self._tracker, i, max_rounds, "after_tool_execution"
            )

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
            llm = self.bare_llm.bind_tools(tools_map, tool_choice="none", strict=False)  # type: ignore [assignment]
        else:
            llm = self.bare_llm.bind_tools(tools_map, strict=False)  # type: ignore [assignment]

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
        logger.debug(
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
                case _:
                    logger.debug(
                        "Ignoring content block with unknown type: %s",
                        block.get("type"),
                    )
        return result

    async def _collect_round_llm_chunks(  # noqa: C901
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        all_mcp_tools: list[StructuredTool],
        is_final_round: bool,
        token_counter: GenericTokenCounter,
        round_index: int,
        result: RoundLLMResult,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Collect one round of LLM chunks, yielding streamed output.

        Populates ``result`` with tool-call chunks, all chunks, and the
        stop flag while yielding ``StreamedChunk`` objects to the caller.
        """
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

                    # Fake-LLM (load tests) returns plain strings; emit and exit.
                    if isinstance(chunk, str):
                        yield StreamedChunk(type=StreamChunkType.TEXT, text=chunk)
                        break

                    # Once finish_reason="stop" is seen we keep draining
                    # the generator instead of breaking out — breaking would
                    # trigger a GeneratorExit traceback inside LangChain's
                    # astream internals.
                    if result.should_stop:
                        continue

                    # LLM signaled completion — mark stop and keep draining.
                    if chunk.response_metadata.get("finish_reason") == "stop":  # type: ignore [attr-defined]
                        result.should_stop = True
                        continue

                    result.all_chunks.append(chunk)

                    # Collect tool-call chunks separately for later assembly.
                    if getattr(chunk, "tool_call_chunks", None):
                        result.tool_call_chunks.append(chunk)
                    else:
                        # Dispatch text and reasoning content to the client.
                        match chunk.content:
                            case str() as text if text and not skip_special_chunk(
                                text, chunk_counter, self.model, is_final_round
                            ):
                                yield StreamedChunk(
                                    type=StreamChunkType.TEXT, text=text
                                )
                            case list() as blocks:
                                for sc in self._streamed_chunks_from_list_content(
                                    blocks, chunk_counter, is_final_round
                                ):
                                    yield sc
                            case str():
                                pass
                            case _:
                                logger.debug(
                                    "Ignoring chunk with unexpected content type: %s",
                                    type(chunk.content).__name__,
                                )

                    chunk_counter += 1
        except TimeoutError:
            logger.error(
                "Timed out waiting for LLM chunks in round %s after %s seconds",
                round_index,
                constants.TOOL_CALL_ROUND_TIMEOUT,
            )
            yield StreamedChunk(
                type=StreamChunkType.TEXT,
                text=(
                    "I could not complete this request in time. " "Please try again."
                ),
            )
            result.should_stop = True

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
        round_index: int,
    ) -> tuple[int, StreamedChunk]:
        """Convert a ToolMessage into a streamed tool_result chunk.

        Returns:
            A tuple of (token_count_for_tool_content, streamed_tool_result_chunk).
        """
        content_token_count = tool_call_message.additional_kwargs.get(
            "token_count"
        ) or self._tracker.count_tokens(str(tool_call_message.content))

        was_truncated = tool_call_message.additional_kwargs.get("truncated", False)
        base_status = tool_call_message.status
        tool_status = "truncated" if was_truncated else base_status
        has_meta = bool(
            isinstance(tool.metadata, dict) and tool.metadata.get("_meta")
            if tool is not None
            else False
        )

        logger.debug(
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

    async def _process_tool_calls_for_round(  # noqa: C901  # pylint: disable=R0912
        self,
        *,
        round_index: int,
        tool_call_chunks: list[AIMessageChunk],
        all_chunks: list[AIMessageChunk],
        all_tools_dict: dict[str, StructuredTool],
        duplicate_tool_names: set[str],
        messages: ChatPromptTemplate,
        offload_manager: "OffloadManager | None" = None,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Resolve, execute, and stream one round of tool calls."""
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
            return

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

        ai_content_text = (
            json.dumps(ai_tool_call_message.content)
            if isinstance(ai_tool_call_message.content, list)
            else str(ai_tool_call_message.content)
        )
        ai_message_tokens = self._tracker.count_tokens(
            ai_content_text + json.dumps(tool_calls)
        )
        self._tracker.charge(TokenCategory.AI_ROUND, ai_message_tokens)

        tool_id_to_name: dict[str, str] = {
            str(tc.get("id", "")): str(tc.get("name", "unknown")) for tc in tool_calls
        }

        for tool_call in tool_calls:
            enriched: dict[str, Any] = {**tool_call}
            tool_name = str(tool_call.get("name", "unknown"))
            self._enrich_with_tool_metadata(enriched, all_tools_dict.get(tool_name))
            logger.debug(
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

        tool_calls_messages: list[ToolMessage] = []
        remaining = self._tracker.tools_round_budget
        if tool_call_definitions:
            if remaining < MIN_TOOL_EXECUTION_TOKENS:
                logger.warning(
                    "Skipping %d tool call(s) in round %s due to low remaining tool budget "
                    "(remaining=%d, minimum_required=%d)",
                    len(tool_call_definitions),
                    round_index,
                    remaining,
                    MIN_TOOL_EXECUTION_TOKENS,
                )
                for tool_id, _tool_args, tool in tool_call_definitions:
                    tool_calls_messages.append(
                        ToolMessage(
                            content=(
                                f"Tool '{tool.name}' call skipped: remaining tool token budget "
                                f"({remaining}) is below minimum required "
                                f"({MIN_TOOL_EXECUTION_TOKENS}). "
                                "Do not retry this exact tool call."
                            ),
                            status="error",
                            tool_call_id=tool_id,
                        )
                    )
            else:
                async for execution_event in execute_tool_calls_stream(
                    tool_call_definitions,
                    remaining,
                    streaming=self.streaming,
                    offload_manager=offload_manager,
                ):
                    match execution_event.event:
                        case StreamChunkType.APPROVAL_REQUIRED:
                            yield StreamedChunk(
                                type=StreamChunkType.APPROVAL_REQUIRED,
                                data=execution_event.data,
                            )
                        case StreamChunkType.TOOL_RESULT:
                            tool_calls_messages.append(execution_event.data)
                        case _:
                            logger.warning(
                                "Ignoring unexpected tool execution event: %s",
                                execution_event,
                            )

        all_tool_messages = skipped_tool_messages + tool_calls_messages
        if remaining > 0:
            all_tool_messages = enforce_tool_token_budget(
                all_tool_messages, remaining, self._tracker.token_handler
            )
        messages.extend(all_tool_messages)

        for tool_call_message in all_tool_messages:
            tool_name = tool_id_to_name.get(tool_call_message.tool_call_id, "unknown")
            content_token_count, tool_result_chunk = (
                self._tool_result_chunk_for_message(
                    tool_call_message=tool_call_message,
                    tool_name=tool_name,
                    tool=all_tools_dict.get(tool_name),
                    round_index=round_index,
                )
            )
            self._tracker.charge(TokenCategory.TOOL_RESULT, content_token_count)
            yield tool_result_chunk
