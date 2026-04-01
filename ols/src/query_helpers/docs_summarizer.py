"""Documentation summarizer with tool-calling support."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Coroutine, Optional

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.retrievers import BaseRetriever

from ols import config, constants
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
from ols.src.query_helpers.llm_execution_agent import (
    LLMExecutionAgent,
    RoundLLMResult,
    ToolTokenUsage,
)
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils.mcp_utils import ClientHeaders, build_mcp_config, get_mcp_tools
from ols.utils.token_handler import TokenHandler

__all__ = ["DocsSummarizer", "RoundLLMResult", "ToolTokenUsage", "run_async_safely"]

logger = logging.getLogger(__name__)


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

        # Configure MCP tool-calling servers (if any).
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

        # Create the LLM execution agent for streaming and tool-calling loops.
        self._llm_agent = LLMExecutionAgent(
            bare_llm=self.bare_llm,
            model=self.model,
            provider=self.provider,
            provider_type=self.provider_config.type,
            model_config=self.model_config,
            streaming=self.streaming,
        )

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
            query: The query to be answered.
            rag_retriever: The retriever to get RAG data/context.

        Returns:
            A tuple containing token handler, RAG chunks, available token budget,
            and max tokens reserved for tools.
        """
        # Build a temporary prompt to measure baseline token usage and
        # determine how many tokens remain for RAG, skills, and history.
        token_handler = TokenHandler()

        temp_prompt, temp_prompt_input = GeneratePrompt(
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

        # Retrieve and truncate RAG context to fit the remaining budget.
        if rag_retriever:
            retrieved_nodes = rag_retriever.retrieve(query)
            logger.info("Retrieved %d documents from indexes", len(retrieved_nodes))

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

        return token_handler, rag_chunks, available_tokens, max_tokens_for_tools

    def _build_final_prompt(
        self,
        query: str,
        history: list[BaseMessage],
        rag_chunks: list[RagChunk],
        token_handler: TokenHandler,
        max_tokens_for_tools: int,
        skill_content: Optional[str] = None,
    ) -> tuple[ChatPromptTemplate, dict[str, str]]:
        """Build the final LLM prompt from collected context.

        Args:
            query: The user query.
            history: Truncated conversation history.
            rag_chunks: Retrieved RAG chunks.
            token_handler: Token handler for budget checking.
            max_tokens_for_tools: Token budget reserved for tools.
            skill_content: Optional skill body to inject into the prompt.

        Returns:
            Tuple of (prompt_template, llm_input_values).
        """
        rag_context = [rag_chunk.text for rag_chunk in rag_chunks]
        if len(rag_context) == 0:
            logger.debug("Using llm to answer the query without reference content")

        final_prompt, llm_input_values = GeneratePrompt(
            query,
            rag_context,
            history,
            self._system_prompt,
            self._tool_calling_enabled,
            self._mode,
            self._cluster_version,
            skill_content=skill_content,
        ).generate_prompt(self.model)

        # Final budget validation — the full prompt (with all context) must
        # still fit within the context window.
        token_handler.calculate_and_check_available_tokens(
            final_prompt.format(**llm_input_values),
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
            max_tokens_for_tools,
        )

        return final_prompt, llm_input_values

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
        # Compute initial token budget and retrieve RAG context.
        token_handler, rag_chunks, available_tokens, max_tokens_for_tools = (
            await self._prepare_prompt_context(query, rag_retriever)
        )

        # Resolve skill before history so that skill tokens reduce the
        # budget available for conversation history.
        skill_content: Optional[str] = None
        async for skill_item in self._resolve_skill(
            query, token_handler, available_tokens
        ):
            if isinstance(skill_item, StreamedChunk):
                yield skill_item
            else:
                skill_content = skill_item

        if skill_content is not None:
            skill_tokens = TokenHandler._get_token_count(
                token_handler.text_to_tokens(skill_content)
            )
            available_tokens -= skill_tokens

        # Prepare conversation history within the remaining token budget.
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

        # Build the final prompt with all collected context and validate
        # that it fits within the context window.
        final_prompt, llm_input_values = self._build_final_prompt(
            query=query,
            history=history,
            rag_chunks=rag_chunks,
            token_handler=token_handler,
            max_tokens_for_tools=max_tokens_for_tools,
            skill_content=skill_content,
        )

        # Invoke the LLM agent to execute the query with iterative tool calling.
        messages = final_prompt.model_copy()
        all_mcp_tools = await get_mcp_tools(query, self.user_token, self.client_headers)
        async for response in self._llm_agent.execute(
            messages=messages,
            llm_input_values=llm_input_values,
            max_rounds=self._get_max_iterations(),
            all_mcp_tools=all_mcp_tools,
            rag_chunks=rag_chunks,
            truncated=truncated,
        ):
            yield response

    async def _resolve_skill(
        self,
        query: str,
        token_handler: TokenHandler,
        available_tokens: int,
    ) -> AsyncGenerator[StreamedChunk | str | None, None]:
        """Resolve a matching skill for the query and check its token budget.

        Yields:
            StreamedChunk (SKILL_SELECTED events) and, as the final item,
            the skill content string or None.
        """
        # No skills configured — nothing to resolve.
        skills_rag = config.skills_rag
        if skills_rag is None:
            yield None
            return

        # Find the best-matching skill via RAG retrieval.
        skill_content: Optional[str] = None
        skill, confidence = skills_rag.retrieve_skill(query)

        # Load the skill file content; fall back gracefully on I/O errors.
        if skill is not None:
            try:
                skill_content = skill.load_content()
            except OSError:
                logger.exception(
                    "Failed to load skill '%s'; falling back to no skill",
                    skill.name,
                )
                skill = None

        # No skill matched or loading failed — proceed without one.
        if skill is None:
            yield None
            return

        # Check whether the skill fits within the available token budget.
        skill_tokens = TokenHandler._get_token_count(
            token_handler.text_to_tokens(skill_content)
        )

        # Skip the skill if it would consume more than 80% of the budget.
        if skill_tokens > available_tokens * 0.8:
            logger.warning(
                "Skill '%s' requires %d tokens but only %d available; skipping",
                skill.name,
                skill_tokens,
                available_tokens,
            )
            yield StreamedChunk(
                type=StreamChunkType.SKILL_SELECTED,
                data={
                    "name": skill.name,
                    "confidence": confidence,
                    "skipped": True,
                    "reason": "exceeds token budget",
                },
            )
            yield None
            return

        # Warn if the skill uses more than half the budget but still fits.
        if skill_tokens > available_tokens * 0.5:
            logger.warning(
                "Skill '%s' uses %d tokens (%.0f%% of available budget)",
                skill.name,
                skill_tokens,
                skill_tokens / available_tokens * 100,
            )

        # Emit the skill selection event and yield the content.
        yield StreamedChunk(
            type=StreamChunkType.SKILL_SELECTED,
            data={
                "name": skill.name,
                "confidence": confidence,
            },
        )
        yield skill_content

    def _get_max_iterations(self) -> int:
        """Return configured max rounds for tool-calling loop.

        An explicit ``max_iterations`` value in the OLS config overrides the
        mode-specific default for all modes.  When the config value is None
        (not set in YAML), the default is chosen based on the active query mode.
        """
        explicit = config.ols_config.max_iterations
        if explicit is not None:
            return explicit
        return constants.MAX_ITERATIONS_BY_MODE.get(
            self._mode, constants.DEFAULT_MAX_ITERATIONS
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
                    case StreamChunkType.SKILL_SELECTED:
                        continue
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
