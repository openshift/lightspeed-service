"""Tool calling agent with iterative tool execution loop."""

import asyncio
import copy
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, TypeAlias

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.structured import StructuredTool
from langchain_openai import ChatOpenAI
from opentelemetry.trace import Span, SpanKind, StatusCode, get_current_span, use_span

from ols import constants
from ols.app.metrics import TokenMetricUpdater
from ols.app.metrics.metrics import gen_ai_client_operation_duration_seconds
from ols.app.metrics.token_counter import GenericTokenCounter
from ols.app.models.config import ModelConfig
from ols.app.models.models import RagChunk, StreamChunkType, StreamedChunk
from ols.src.tools.tool_result_inspection import (
    INJECTION_CATEGORIES,
    ToolResultClassifier,
    ToolResultInspectionError,
    ToolResultRejectedError,
    chunk_text,
)
from ols.src.tools.tools import (
    TOOL_RESULT_BUDGET_EXCEEDED_MESSAGE,
    ToolResultBudgetExceededError,
    _wrap_tool_output,
    enforce_tool_token_budget,
    execute_tool_calls_stream,
)
from ols.utils.audit_logger import AuditContext
from ols.utils.token_handler import TokenBudgetTracker, TokenCategory

if TYPE_CHECKING:
    from ols.src.tools.offloaded_content import OffloadManager

logger = logging.getLogger(__name__)

MIN_TOOL_EXECUTION_TOKENS = 100
FINAL_SYNTHESIS_INSTRUCTION = (
    "Tool investigation is exhausted. Do not call tools again.\n"
    "Summarize available evidence; clearly state what remains uncertain if evidence is partial."
)
FINAL_SYNTHESIS_FAILURE = (
    "I could not complete an answer from the available evidence. Please retry."
)
FINAL_SYNTHESIS_RETRY_INSTRUCTION = (
    "Your previous response was not a usable answer. Answer the user's question "
    "now using only the evidence already gathered. Do not emit tool calls or "
    "tool-call markup."
)
# Providers whose final-round request is sent without tools bound. Vertex was the
# original case; vLLM providers also need unbinding because tool_choice="none"
# still leaves tools bound and some models ignore it and emit raw tool-call XML.
_FINAL_ROUND_TOOL_STRIPPING_PROVIDERS = frozenset(
    {
        constants.PROVIDER_GOOGLE_VERTEX,
        constants.PROVIDER_RHOAI_VLLM,
        constants.PROVIDER_RHELAI_VLLM,
    }
)
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
    collected_text: list[str] = field(default_factory=list)
    collected_thinking: list[str] = field(default_factory=list)
    final_answer_suppressed: bool = False


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


_TOOL_CALL_ANSWER_PREFIXES = ("<tool_call>", "<function=")


class _FinalAnswerToolCallGuard:
    """Hold back a final-round answer until it can be ruled out as a tool call.

    When the final round sets ``tool_choice="none"``, vLLM does not extract
    tool calls and returns the raw model text as content. If the model ignores
    the synthesis instruction and emits a call anyway, the guard suppresses it
    so raw XML never reaches the client.
    """

    def __init__(self) -> None:
        """Initialize an undecided guard with an empty buffer."""
        self._buffer = ""
        self._decided = False
        self._suppressed = False

    @property
    def suppressed(self) -> bool:
        """Return True when a tool call was detected and suppressed."""
        return self._suppressed

    def feed(self, text: str) -> list[str]:
        """Return text that is safe to stream, buffering while undecided."""
        if self._decided:
            return [] if self._suppressed else [text]
        self._buffer += text
        stripped = self._buffer.lstrip()
        if not stripped:
            return []
        if any(stripped.startswith(prefix) for prefix in _TOOL_CALL_ANSWER_PREFIXES):
            self._decided = True
            self._suppressed = True
            self._buffer = ""
            return []
        if any(prefix.startswith(stripped) for prefix in _TOOL_CALL_ANSWER_PREFIXES):
            return []
        self._decided = True
        emitted = [self._buffer]
        self._buffer = ""
        return emitted

    def finish(self) -> list[str]:
        """Return any buffered text once the stream is complete."""
        if self._decided:
            return []
        self._decided = True
        stripped = self._buffer.lstrip()
        if stripped and any(
            prefix.startswith(stripped) for prefix in _TOOL_CALL_ANSWER_PREFIXES
        ):
            self._suppressed = True
            self._buffer = ""
            return []
        emitted = [self._buffer] if self._buffer else []
        self._buffer = ""
        return emitted


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
        audit_ctx: Optional[AuditContext] = None,
        tool_result_classifier: ToolResultClassifier | None = None,
        tool_call_round_timeout: int | None = None,
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
            audit_ctx: Audit context for structured event logging.
            tool_result_classifier: Optional classifier for model-visible tool results.
            tool_call_round_timeout: Optional per-round timeout in seconds. Falls
                back to ``constants.TOOL_CALL_ROUND_TIMEOUT`` when unset.
        """
        self.bare_llm = bare_llm
        self.model = model
        self.provider = provider
        self.provider_type = provider_type
        self.model_config = model_config
        self.streaming = streaming
        self._tracker = token_budget_tracker
        self._audit_ctx = audit_ctx
        self._tool_result_classifier = tool_result_classifier
        self._tool_call_round_timeout = tool_call_round_timeout

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
                if chunk.type == StreamChunkType.TOOL_RESULT:
                    rag_chunks.extend(chunk.data.pop("referenced_documents", []))
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

    def _emit_turn_audit(
        self,
        round_result: "RoundLLMResult",
        turn_index: int,
        token_counter: GenericTokenCounter,
        prev_input_tokens: int,
        prev_output_tokens: int,
    ) -> tuple[int, int]:
        """Emit audit span events for a completed LLM turn."""
        if not self._audit_ctx:
            return prev_input_tokens, prev_output_tokens
        cur_input = token_counter.token_counter.input_tokens
        cur_output = (
            token_counter.token_counter.output_tokens
            + token_counter.token_counter.reasoning_tokens
        )
        capture = self._audit_ctx.capture_content
        thinking = "".join(round_result.collected_thinking)
        text = "".join(round_result.collected_text)
        if thinking:
            self._audit_ctx.logger.llm_thinking(
                content=thinking, capture_content=capture
            )
        if text:
            self._audit_ctx.logger.llm_text(content=text, capture_content=capture)
        self._audit_ctx.logger.llm_turn(
            turn_index=turn_index,
            input_tokens=cur_input - prev_input_tokens,
            output_tokens=cur_output - prev_output_tokens,
        )
        return cur_input, cur_output

    def _prepare_round_request(
        self,
        messages: ChatPromptTemplate,
        all_mcp_tools: list[StructuredTool],
        is_final_round: bool,
    ) -> tuple[ChatPromptTemplate, list[StructuredTool], bool] | None:
        """Prepare round prompt and tools, or return None if synthesis exceeds budget."""
        if not is_final_round or not all_mcp_tools:
            return messages, all_mcp_tools, False

        synthesize = self.provider_type in _FINAL_ROUND_TOOL_STRIPPING_PROVIDERS
        instruction_tokens = self._tracker.count_tokens(FINAL_SYNTHESIS_INSTRUCTION)
        if instruction_tokens > self._tracker.remaining:
            if synthesize:
                logger.warning("Insufficient context budget for final synthesis")
                return None
            logger.warning(
                "Insufficient context budget for final synthesis instruction; "
                "continuing final round without it"
            )
            return messages, all_mcp_tools, False

        self._tracker.charge(TokenCategory.PROMPT, instruction_tokens)
        round_messages = (
            messages.model_copy(deep=True)
            if hasattr(messages, "model_copy")
            else copy.deepcopy(messages)
        )
        round_messages.append(HumanMessage(content=FINAL_SYNTHESIS_INSTRUCTION))
        round_tools = [] if synthesize else all_mcp_tools
        return round_messages, round_tools, synthesize

    async def _iterate_with_tools(  # noqa: C901
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

        prev_input_tokens = 0
        prev_output_tokens = 0

        for i in range(1, max_rounds + 1):
            is_final_round = (not all_mcp_tools) or (i == max_rounds)
            logger.debug("Tool calling round %s (final: %s)", i, is_final_round)

            round_request = self._prepare_round_request(
                messages, all_mcp_tools, is_final_round
            )
            if round_request is None:
                yield StreamedChunk(
                    type=StreamChunkType.TEXT, text=FINAL_SYNTHESIS_FAILURE
                )
                return
            round_messages, round_tools, synthesize = round_request
            round_result = RoundLLMResult()
            turn_span = (
                self._audit_ctx.span(
                    f"chat {self.model}",
                    kind=SpanKind.CLIENT,
                    **{
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": self.model,
                        "gen_ai.provider.name": self.provider,
                    },
                    turn_index=i,
                )
                if self._audit_ctx
                else nullcontext()
            )
            with turn_span:
                async for chunk in self._collect_round_response(
                    messages=round_messages,
                    llm_input_values=llm_input_values,
                    all_mcp_tools=round_tools,
                    is_final_round=is_final_round,
                    token_counter=token_counter,
                    round_index=i,
                    result=round_result,
                    synthesize=synthesize,
                ):
                    yield chunk

                prev_input_tokens, prev_output_tokens = self._emit_turn_audit(
                    round_result,
                    i,
                    token_counter,
                    prev_input_tokens,
                    prev_output_tokens,
                )

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
            except ToolResultInspectionError:
                raise
            except ToolResultBudgetExceededError:
                log_tool_loop_iteration(
                    self._tracker, i, max_rounds, "tool_result_budget_exceeded"
                )
                yield StreamedChunk(
                    type=StreamChunkType.TEXT,
                    text=TOOL_RESULT_BUDGET_EXCEEDED_MESSAGE,
                )
                return
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
        # strict=False is a ChatOpenAI-specific workaround for
        # langchain-ai/langchain#35837 (Responses API defaults strict=True).
        # Passing it to non-OpenAI providers (e.g. ChatAnthropicVertex) causes
        # the kwarg to leak into the underlying SDK call which rejects it.
        strict_kwargs: dict[str, Any] = (
            {"strict": False} if isinstance(self.bare_llm, ChatOpenAI) else {}
        )
        if not tools_map:
            llm = self.bare_llm
        elif is_final_round:
            # Responses API dumps tool args as text when tools are unbound;
            # tool_choice="none" prevents this.
            tool_choice: str | dict[str, str] = "none"
            if self.provider_type in (
                constants.PROVIDER_ANTHROPIC,
                constants.PROVIDER_GOOGLE_VERTEX_ANTHROPIC,
            ):
                tool_choice = {"type": "none"}
            llm = self.bare_llm.bind_tools(
                tools_map, tool_choice=tool_choice, **strict_kwargs
            )  # type: ignore [assignment]
        else:
            llm = self.bare_llm.bind_tools(tools_map, **strict_kwargs)  # type: ignore [assignment]

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
        else:
            logger.debug(
                "LLM invocation completed: provider=%s, model=%s, elapsed=%.2fs",
                self.provider,
                self.model,
                time.monotonic() - llm_start_time,
            )
        finally:
            elapsed = time.monotonic() - llm_start_time
            gen_ai_client_operation_duration_seconds.labels(
                gen_ai_request_model=self.model,
                gen_ai_provider_name=self.provider_type,
                gen_ai_operation_name="chat",
            ).observe(elapsed)

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

    def _build_synthesis_retry_messages(
        self,
        messages: ChatPromptTemplate,
    ) -> ChatPromptTemplate | None:
        """Append the retry instruction when the token budget allows it."""
        retry_tokens = self._tracker.count_tokens(FINAL_SYNTHESIS_RETRY_INSTRUCTION)
        if retry_tokens > self._tracker.remaining:
            logger.warning("Insufficient context budget for final synthesis retry")
            return None
        self._tracker.charge(TokenCategory.PROMPT, retry_tokens)
        retry_messages = (
            messages.model_copy(deep=True)
            if hasattr(messages, "model_copy")
            else copy.deepcopy(messages)
        )
        retry_messages.append(HumanMessage(content=FINAL_SYNTHESIS_RETRY_INSTRUCTION))
        return retry_messages

    def _close_final_answer_guard(
        self,
        guard: _FinalAnswerToolCallGuard,
        result: RoundLLMResult,
        round_index: int,
    ) -> list[str]:
        """Flush buffered guard text and record suppressed tool-call leaks."""
        emitted = guard.finish()
        if guard.suppressed:
            result.final_answer_suppressed = True
            logger.warning(
                "Suppressed final-round tool-call leak in round %s (provider=%s)",
                round_index,
                self.provider_type,
            )
        return emitted

    async def _collect_round_response(
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        all_mcp_tools: list[StructuredTool],
        is_final_round: bool,
        token_counter: GenericTokenCounter,
        round_index: int,
        result: RoundLLMResult,
        synthesize: bool,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Collect a response and retry a failed final synthesis once."""
        has_answer = False
        max_attempts = 2 if synthesize else 1
        round_messages = messages

        for attempt in range(1, max_attempts + 1):
            result.should_stop = False
            result.final_answer_suppressed = False
            apply_final_answer_guard = is_final_round and (
                bool(all_mcp_tools) or synthesize
            )
            try:
                async for chunk in self._collect_round_llm_chunks(
                    messages=round_messages,
                    llm_input_values=llm_input_values,
                    all_mcp_tools=all_mcp_tools,
                    is_final_round=is_final_round,
                    token_counter=token_counter,
                    round_index=round_index,
                    result=result,
                    apply_final_answer_guard=apply_final_answer_guard,
                ):
                    if chunk.type == StreamChunkType.TEXT and chunk.text.strip():
                        has_answer = True
                    yield chunk
            except Exception:
                if not synthesize or has_answer:
                    raise
                logger.exception("Final synthesis invocation failed")
                result.should_stop = True
                break

            if has_answer or not synthesize:
                return

            if attempt < max_attempts and result.final_answer_suppressed:
                retry_messages = self._build_synthesis_retry_messages(messages)
                if retry_messages is not None:
                    logger.warning("Final synthesis emitted a tool call; retrying once")
                    round_messages = retry_messages
                    continue
            break

        if synthesize and not has_answer:
            logger.error("Final synthesis returned no text after retry")
            yield StreamedChunk(type=StreamChunkType.TEXT, text=FINAL_SYNTHESIS_FAILURE)

    async def _collect_round_llm_chunks(  # noqa: C901  # pylint: disable=R0912
        self,
        messages: ChatPromptTemplate,
        llm_input_values: dict[str, str],
        all_mcp_tools: list[StructuredTool],
        is_final_round: bool,
        token_counter: GenericTokenCounter,
        round_index: int,
        result: RoundLLMResult,
        *,
        apply_final_answer_guard: bool = False,
    ) -> AsyncGenerator[StreamedChunk, None]:
        """Collect one round of LLM chunks, yielding streamed output.

        Populates ``result`` with tool-call chunks, all chunks, and the
        stop flag while yielding ``StreamedChunk`` objects to the caller.
        """
        chunk_counter = 0
        guard = _FinalAnswerToolCallGuard() if apply_final_answer_guard else None
        timeout_seconds = (
            self._tool_call_round_timeout or constants.TOOL_CALL_ROUND_TIMEOUT
        )
        try:
            async with asyncio.timeout(timeout_seconds):
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
                        if guard is not None:
                            for emitted in guard.feed(chunk):
                                result.collected_text.append(emitted)
                                yield StreamedChunk(
                                    type=StreamChunkType.TEXT, text=emitted
                                )
                            for emitted in self._close_final_answer_guard(
                                guard, result, round_index
                            ):
                                result.collected_text.append(emitted)
                                yield StreamedChunk(
                                    type=StreamChunkType.TEXT, text=emitted
                                )
                        else:
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
                                for emitted in (guard.feed(text) if guard else [text]):
                                    result.collected_text.append(emitted)
                                    yield StreamedChunk(
                                        type=StreamChunkType.TEXT, text=emitted
                                    )
                            case list() as blocks:
                                for sc in self._streamed_chunks_from_list_content(
                                    blocks, chunk_counter, is_final_round
                                ):
                                    if sc.type == StreamChunkType.TEXT:
                                        for emitted in (
                                            guard.feed(sc.text) if guard else [sc.text]
                                        ):
                                            result.collected_text.append(emitted)
                                            yield StreamedChunk(
                                                type=StreamChunkType.TEXT, text=emitted
                                            )
                                    else:
                                        if sc.type == StreamChunkType.REASONING:
                                            result.collected_thinking.append(sc.text)
                                        yield sc
                            case str():
                                pass
                            case _:
                                logger.debug(
                                    "Ignoring chunk with unexpected content type: %s",
                                    type(chunk.content).__name__,
                                )

                    chunk_counter += 1

                if guard is not None:
                    for emitted in self._close_final_answer_guard(
                        guard, result, round_index
                    ):
                        result.collected_text.append(emitted)
                        yield StreamedChunk(type=StreamChunkType.TEXT, text=emitted)
        except TimeoutError:
            logger.error(
                "Timed out waiting for LLM chunks in round %s after %s seconds",
                round_index,
                timeout_seconds,
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
        raw_tool_call_message: ToolMessage | None = None,
        tool_name: str,
        tool: Optional[StructuredTool],
        round_index: int,
    ) -> tuple[int, StreamedChunk]:
        """Convert a ToolMessage into a streamed tool_result chunk.

        Returns:
            A tuple of (token_count_for_tool_content, streamed_tool_result_chunk).
        """
        content_token_count = tool_call_message.additional_kwargs.get("token_count")
        if content_token_count is None:
            model_content = str(tool_call_message.content)
            if tool_call_message.name is not None:
                model_content = _wrap_tool_output(model_content, tool_call_message.name)
            content_token_count = self._tracker.count_tokens(model_content)

        event_message = raw_tool_call_message or tool_call_message
        was_truncated = event_message.additional_kwargs.get("truncated", False)
        base_status = event_message.status
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
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        tool_result_data: dict[str, Any] = {
            "id": tool_call_message.tool_call_id,
            "name": tool_name,
            "status": tool_status,
            "content": event_message.content,
            "type": StreamChunkType.TOOL_RESULT.value,
            "round": round_index,
        }
        structured_content = event_message.additional_kwargs.get("structured_content")
        if structured_content:
            tool_result_data["structured_content"] = structured_content
        ref_docs = event_message.additional_kwargs.get("referenced_documents")
        if ref_docs:
            tool_result_data["referenced_documents"] = ref_docs
        self._enrich_with_tool_metadata(tool_result_data, tool)

        return content_token_count, StreamedChunk(
            type=StreamChunkType.TOOL_RESULT, data=tool_result_data
        )

    async def _inspect_tool_messages(
        self,
        tool_messages: list[ToolMessage],
        tool_id_to_name: dict[str, str],
        tool_id_to_audit_span: dict[str, Span] | None = None,
        raw_tool_messages_by_id: dict[str, ToolMessage] | None = None,
        tool_id_to_tool: dict[str, StructuredTool | None] | None = None,
    ) -> None:
        """Inspect and audit each result before deciding whether to deliver the round."""
        tool_by_id = tool_id_to_tool or {}
        inspection_messages = [
            message
            for message in tool_messages
            if not self._is_inspection_exempt_tool(
                tool_id_to_name.get(message.tool_call_id, "unknown"),
                tool_by_id.get(message.tool_call_id),
            )
        ]
        max_tokens = 0
        overlap_tokens = 0
        if self._tool_result_classifier is not None and inspection_messages:
            classifier_budget = (
                self.model_config.context_window_size
                - self.model_config.parameters.max_tokens_for_response
                - 512
            )
            if classifier_budget <= 0:
                raise ToolResultInspectionError(
                    "insufficient classifier context budget"
                )
            max_tokens = classifier_budget
            overlap_tokens = min(256, max_tokens - 1)

        first_error: ToolResultInspectionError | None = None
        for message in tool_messages:
            audit_span = (tool_id_to_audit_span or {}).get(message.tool_call_id)
            result_error = await self._inspect_single_tool_message(
                message=message,
                audit_message=(raw_tool_messages_by_id or {}).get(
                    message.tool_call_id, message
                ),
                tool_name=tool_id_to_name.get(message.tool_call_id, "unknown"),
                tool=tool_by_id.get(message.tool_call_id),
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                audit_span=audit_span,
            )
            if result_error is not None:
                if self._audit_ctx:
                    error_type = (
                        "malicious"
                        if isinstance(result_error, ToolResultRejectedError)
                        else "classifier_error"
                    )
                    get_current_span().set_status(StatusCode.ERROR, error_type)
                if first_error is None:
                    first_error = result_error

        if first_error is not None:
            raise first_error

    async def _inspect_single_tool_message(
        self,
        *,
        message: ToolMessage,
        audit_message: ToolMessage,
        tool_name: str,
        tool: StructuredTool | None,
        max_tokens: int,
        overlap_tokens: int,
        audit_span: Span | None,
    ) -> ToolResultInspectionError | None:
        """Inspect one result and audit it only after a passing decision."""
        content = (
            message.content
            if isinstance(message.content, str)
            else json.dumps(message.content, ensure_ascii=False)
        )
        audit_content = (
            audit_message.content
            if isinstance(audit_message.content, str)
            else json.dumps(audit_message.content, ensure_ascii=False)
        )
        if self._is_inspection_exempt_tool(tool_name, tool):
            self._audit_tool_result(audit_message, audit_span, audit_content)
            return None
        if self._tool_result_classifier is None:
            self._audit_tool_result(audit_message, audit_span, audit_content)
            return None

        result_span_context = (
            use_span(audit_span, end_on_exit=False)
            if audit_span is not None
            else nullcontext()
        )
        with result_span_context:
            chunk_count = len(
                chunk_text(
                    content,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    token_handler=self._tracker.token_handler,
                )
            )
            result_type = "error" if message.status == "error" else "result"
            parent_span = get_current_span()
            span_context = (
                self._audit_ctx.span(
                    "tool_result.inspection",
                    runtime="classic",
                    **{
                        "inspection.runtime": "classic",
                        "inspection.result_type": result_type,
                        "inspection.chunk_count": chunk_count,
                        "inspection.enabled": True,
                        "tool.name": tool_name,
                        "llm.provider": self.provider_type,
                        "llm.model": self.model,
                    },
                )
                if self._audit_ctx
                else nullcontext()
            )
            inspection_error: ToolResultInspectionError | None = None
            with span_context as inspection_span:
                try:
                    await self._tool_result_classifier.inspect(
                        tool_name,
                        result_type,
                        content,
                        max_tokens=max_tokens,
                        token_handler=self._tracker.token_handler,
                        overlap_tokens=overlap_tokens,
                    )
                except ToolResultRejectedError as error:
                    inspection_error = error
                    category = (
                        error.category
                        if error.category in INJECTION_CATEGORIES
                        else "unknown"
                    )
                    logger.debug(
                        "Tool result inspection outcome=malicious result_type=%s "
                        "category=%s",
                        result_type,
                        category,
                    )
                    if self._audit_ctx:
                        inspection_span.set_attribute("inspection.outcome", "malicious")
                        inspection_span.set_attribute(
                            "inspection.category", error.category
                        )
                        inspection_span.set_status(StatusCode.ERROR, "malicious")
                        parent_span.set_status(StatusCode.ERROR, "malicious")
                except ToolResultInspectionError as error:
                    inspection_error = error
                    logger.debug(
                        "Tool result inspection outcome=classifier_error result_type=%s",
                        result_type,
                    )
                    if self._audit_ctx:
                        inspection_span.set_attribute(
                            "inspection.outcome", "classifier_error"
                        )
                        inspection_span.set_status(StatusCode.ERROR, "classifier_error")
                        parent_span.set_status(StatusCode.ERROR, "classifier_error")
                else:
                    logger.debug(
                        "Tool result inspection outcome=benign result_type=%s",
                        result_type,
                    )
                    if self._audit_ctx:
                        inspection_span.set_attribute("inspection.outcome", "benign")

            if inspection_error is not None:
                return inspection_error

        self._audit_tool_result(audit_message, audit_span, audit_content)
        return None

    @staticmethod
    def _is_inspection_exempt_tool(tool_name: str, tool: StructuredTool | None) -> bool:
        """Return whether a registered first-party tool is exempt from inspection."""
        metadata = tool.metadata or {} if tool is not None else {}
        return (
            tool_name == constants.OPENSHIFT_DOCS_TOOL_NAME
            and metadata.get(constants.OLS_INTERNAL_TOOL_METADATA_KEY)
            == constants.OPENSHIFT_DOCS_TOOL_NAME
            and not metadata.get("mcp_server")
        )

    def _audit_tool_result(
        self,
        message: ToolMessage,
        audit_span: Span | None,
        content: str,
    ) -> None:
        """Record a result only after its inspection succeeds."""
        if self._audit_ctx is None or audit_span is None:
            return
        span_context = use_span(audit_span, end_on_exit=False)
        with span_context:  # pylint: disable=not-context-manager
            self._audit_ctx.logger.tool_result(
                output_length=len(content),
                success=message.status == "success",
                duration_ms=message.additional_kwargs.get("duration_ms"),
                output_content=content,
            )

    async def _process_tool_calls_for_round(  # noqa: C901  # pylint: disable=R0912,R0915
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
        tool_id_to_tool: dict[str, StructuredTool | None] = {
            str(tc.get("id", "")): all_tools_dict.get(str(tc.get("name", "unknown")))
            for tc in tool_calls
        }

        for tool_call in tool_calls:
            enriched: dict[str, Any] = {**tool_call}
            tool_name = str(tool_call.get("name", "unknown"))
            resolved_tool = all_tools_dict.get(tool_name)
            self._enrich_with_tool_metadata(enriched, resolved_tool)
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
        tool_id_to_audit_span: dict[str, Span] = {}
        remaining = self._tracker.tools_round_budget
        try:
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
                        audit_ctx=self._audit_ctx,
                    ):
                        match execution_event.event:
                            case StreamChunkType.APPROVAL_REQUIRED:
                                yield StreamedChunk(
                                    type=StreamChunkType.APPROVAL_REQUIRED,
                                    data=execution_event.data,
                                )
                            case StreamChunkType.TOOL_RESULT:
                                tool_calls_messages.append(execution_event.data)
                                if execution_event.audit_span is not None:
                                    tool_id_to_audit_span[
                                        execution_event.data.tool_call_id
                                    ] = execution_event.audit_span
                            case _:
                                logger.warning(
                                    "Ignoring unexpected tool execution event: %s",
                                    execution_event,
                                )

            raw_tool_messages_by_id = {
                message.tool_call_id: message
                for message in skipped_tool_messages + tool_calls_messages
            }
            all_tool_messages = skipped_tool_messages + tool_calls_messages
            if remaining > 0:
                all_tool_messages = enforce_tool_token_budget(
                    all_tool_messages, remaining, self._tracker.token_handler
                )
            await self._inspect_tool_messages(
                all_tool_messages,
                tool_id_to_name,
                tool_id_to_audit_span,
                raw_tool_messages_by_id,
                tool_id_to_tool,
            )
        finally:
            for audit_span in tool_id_to_audit_span.values():
                if audit_span.is_recording():
                    audit_span.end()

        model_tool_messages = [
            (
                message.model_copy(
                    update={
                        "content": _wrap_tool_output(str(message.content), message.name)
                    }
                )
                if message.name is not None
                else message
            )
            for message in all_tool_messages
        ]
        messages.extend(model_tool_messages)

        for tool_call_message in all_tool_messages:
            tool_name = tool_id_to_name.get(tool_call_message.tool_call_id, "unknown")
            content_token_count, tool_result_chunk = (
                self._tool_result_chunk_for_message(
                    tool_call_message=tool_call_message,
                    raw_tool_call_message=raw_tool_messages_by_id.get(
                        tool_call_message.tool_call_id
                    ),
                    tool_name=tool_name,
                    tool=all_tools_dict.get(tool_name),
                    round_index=round_index,
                )
            )
            self._tracker.charge(TokenCategory.TOOL_RESULT, content_token_count)
            yield tool_result_chunk
