"""Documentation summarizer with tool-calling support."""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Coroutine, Optional

from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.structured import StructuredTool
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
    log_tool_loop_iteration,
)
from ols.src.query_helpers.query_helper import QueryHelper
from ols.src.skills.skills_rag import create_skill_support_tool
from ols.src.tools.offloaded_content import OffloadManager
from ols.utils.mcp_utils import (
    ClientHeaders,
    build_mcp_config,
    get_mcp_tools,
    retrieve_from_knowledge_mcps,
)
from ols.utils.token_handler import (
    PromptTooLongError,
    TokenBudgetTracker,
    TokenCategory,
    TokenHandler,
)

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

        self._tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=self.model_config.context_window_size,
            max_response_tokens=self.model_config.parameters.max_tokens_for_response,
            max_tool_tokens=self.model_config.max_tokens_for_tools,
            round_cap_fraction=config.ols_config.tool_round_cap_fraction,
        )
        self._tracker.set_tool_loop_max_rounds(self._get_max_iterations())

        set_debug(self.verbose)

        self._llm_agent = LLMExecutionAgent(
            bare_llm=self.bare_llm,
            model=self.model,
            provider=self.provider,
            provider_type=self.provider_config.type,
            model_config=self.model_config,
            streaming=self.streaming,
            token_budget_tracker=self._tracker,
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
    ) -> list[RagChunk]:
        """Prepare RAG context for prompt construction.

        Args:
            query: The query to be answered.
            rag_retriever: The retriever to get RAG data/context.

        Returns:
            RAG chunks truncated to fit the prompt budget.
        """
        temp_prompt, temp_prompt_input = GeneratePrompt(
            query,
            ["sample"],
            [AIMessage("sample")],
            self._system_prompt,
            self._tool_calling_enabled,
            self._mode,
            self._cluster_version,
        ).generate_prompt(self.model)
        prompt_tokens = self._tracker.count_tokens(
            temp_prompt.format(**temp_prompt_input)
        )
        if prompt_tokens > self._tracker.prompt_budget:
            raise PromptTooLongError(
                f"Prompt length {prompt_tokens} exceeds "
                f"LLM available context window limit "
                f"{self._tracker.prompt_budget} tokens"
            )
        self._tracker.charge(TokenCategory.PROMPT, prompt_tokens)

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

            rag_chunks = self._tracker.token_handler.truncate_rag_context(
                retrieved_nodes, self._tracker.history_budget
            )
            rag_tokens = sum(
                self._tracker.count_tokens(chunk.text) for chunk in rag_chunks
            )
            self._tracker.charge(TokenCategory.RAG, rag_tokens)
        else:
            logger.warning("Proceeding without RAG content. Check start up messages.")
            rag_chunks = []

        return rag_chunks

    def _serialized_tool_definitions_text(
        self, all_mcp_tools: list[StructuredTool]
    ) -> str:
        """Return JSON serialization of MCP tool definitions for token counting."""
        if not all_mcp_tools:
            return ""
        return json.dumps(
            [
                {"name": t.name, "description": t.description, "schema": t.args}
                for t in all_mcp_tools
            ]
        )

    def _build_final_prompt(
        self,
        query: str,
        history: list[BaseMessage],
        rag_chunks: list[RagChunk],
        skill_content: Optional[str] = None,
        *,
        tool_definitions_tokens: int = 0,
    ) -> tuple[ChatPromptTemplate, dict[str, str]]:
        """Build the final LLM prompt and charge the token budget.

        Args:
            query: The user query.
            history: Truncated conversation history.
            rag_chunks: Retrieved RAG chunks.
            skill_content: Optional skill body to inject into the prompt.
            tool_definitions_tokens: Token count for MCP tool schemas (not in the
                formatted prompt string); included in the prompt-budget check.

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

        log_tool_loop_iteration(
            self._tracker, 0, self._get_max_iterations(), "prompt_built"
        )
        if (
            self._tracker.total_used + tool_definitions_tokens
            > self._tracker.prompt_budget
        ):
            if tool_definitions_tokens > 0:
                raise PromptTooLongError(
                    f"Tool definitions ({tool_definitions_tokens} tokens) with current "
                    f"request ({self._tracker.total_used} tokens) exceed prompt budget "
                    f"({self._tracker.prompt_budget} tokens)"
                )
            raise PromptTooLongError(
                f"Prompt ({self._tracker.total_used} tokens) exceeds "
                f"budget ({self._tracker.prompt_budget} tokens)"
            )

        return final_prompt, llm_input_values

    def _create_offload_manager(self) -> Optional[OffloadManager]:
        """Create an OffloadManager if tool calling is enabled, else None."""
        if not self._tool_calling_enabled:
            return None
        return OffloadManager(
            storage_path=config.ols_config.offload_storage_path,
        )

    async def generate_response(  # noqa: C901  # pylint: disable=too-many-branches,too-many-statements
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
        rag_chunks = await self._prepare_prompt_context(query, rag_retriever)

        knowledge_results = await retrieve_from_knowledge_mcps(
            query, self.user_token, self.client_headers
        )
        if knowledge_results:
            for kr in knowledge_results:
                byok_chunk = RagChunk(
                    text=kr["text"],
                    doc_url=kr.get("source", ""),
                    doc_title=kr.get("title", "BYOK"),
                )
                rag_chunks.insert(0, byok_chunk)
            byok_tokens = sum(
                self._tracker.count_tokens(kr["text"]) for kr in knowledge_results
            )
            self._tracker.charge(TokenCategory.RAG, byok_tokens)
            logger.info(
                "Injected %d BYOK knowledge chunks (%d tokens) from %d MCP sources",
                len(knowledge_results),
                byok_tokens,
                len(knowledge_results),
            )

        skill_content: Optional[str] = None
        has_support_files = False
        skill = None
        skills_rag = config.skills_rag
        if skills_rag is not None:
            skill, confidence = skills_rag.retrieve_skill(query)
            if skill is not None:
                loaded = skill.load_skill()
                if not loaded.ok:
                    skill = None
                else:
                    skill_content = loaded.content
                    has_support_files = loaded.has_support_files
                    skill_tokens = self._tracker.count_tokens(skill_content)
                    shared_tail_budget = self._tracker.prompt_budget_remaining
                    if skill_tokens > shared_tail_budget * 0.8:
                        logger.warning(
                            "Skill '%s' requires %d tokens but only %d available "
                            "in prompt tail (skill + history); skipping",
                            skill.name,
                            skill_tokens,
                            shared_tail_budget,
                        )
                        skill_content = None
                        has_support_files = False
                        yield StreamedChunk(
                            type=StreamChunkType.SKILL_SELECTED,
                            data={
                                "name": skill.name,
                                "confidence": confidence,
                                "skipped": True,
                                "reason": "exceeds token budget",
                            },
                        )
                    else:
                        self._tracker.charge(TokenCategory.SKILL, skill_tokens)
                        if skill_tokens > shared_tail_budget * 0.5:
                            logger.warning(
                                "Skill '%s' uses %d tokens (%.0f%% of prompt tail budget)",
                                skill.name,
                                skill_tokens,
                                skill_tokens / shared_tail_budget * 100,
                            )
                        yield StreamedChunk(
                            type=StreamChunkType.SKILL_SELECTED,
                            data={
                                "name": skill.name,
                                "confidence": confidence,
                            },
                        )

        history: list[BaseMessage] = []
        truncated = False
        available_tokens = self._tracker.history_budget
        async for item in prepare_history(
            user_id=user_id,
            conversation_id=conversation_id,
            skip_user_id_check=skip_user_id_check,
            available_tokens=available_tokens,
            provider=self.provider,
            model=self.model,
            bare_llm=self.bare_llm,
            token_handler=self._tracker.token_handler,
        ):
            if isinstance(item, StreamedChunk):
                yield item
            else:
                history, truncated = item

        for msg in history:
            if isinstance(msg.content, str):
                self._tracker.charge(
                    TokenCategory.HISTORY,
                    self._tracker.count_tokens(msg.content),
                )

        final_prompt, llm_input_values = self._build_final_prompt(
            query=query,
            history=history,
            rag_chunks=rag_chunks,
            skill_content=skill_content,
            tool_definitions_tokens=0,
        )

        messages = final_prompt.model_copy()
        mcp_tools_query = f"{skill_content}\n\n{query}" if skill_content else query
        all_mcp_tools = await get_mcp_tools(
            mcp_tools_query, self.user_token, self.client_headers
        )
        if skill is not None and skill_content is not None and has_support_files:
            all_mcp_tools.append(create_skill_support_tool(skill))
        tool_definitions_text = self._serialized_tool_definitions_text(all_mcp_tools)
        tool_definitions_tokens = (
            self._tracker.count_tokens(tool_definitions_text)
            if tool_definitions_text
            else 0
        )
        if (
            self._tracker.total_used + tool_definitions_tokens
            > self._tracker.prompt_budget
        ):
            if tool_definitions_tokens > 0:
                raise PromptTooLongError(
                    f"Tool definitions ({tool_definitions_tokens} tokens) with current "
                    f"request ({self._tracker.total_used} tokens) exceed prompt budget "
                    f"({self._tracker.prompt_budget} tokens)"
                )
            raise PromptTooLongError(
                f"Prompt ({self._tracker.total_used} tokens) exceeds "
                f"budget ({self._tracker.prompt_budget} tokens)"
            )

        offload_manager = self._create_offload_manager()
        try:
            async for response in self._llm_agent.execute(
                messages=messages,
                llm_input_values=llm_input_values,
                max_rounds=self._get_max_iterations(),
                all_mcp_tools=all_mcp_tools,
                rag_chunks=rag_chunks,
                truncated=truncated,
                tool_definitions_tokens=tool_definitions_tokens,
                offload_manager=offload_manager,
            ):
                yield response
        finally:
            if offload_manager is not None:
                offload_manager.cleanup()

    def _get_max_iterations(self) -> int:
        """Return configured max rounds for tool-calling loop.

        The result is the greater of the explicit ``max_iterations`` config
        value and the mode-specific default.  This ensures the config can raise
        the cap but never lower it below the mode's built-in minimum.  When the
        config value is None (not set in YAML), the mode default is used as-is.
        """
        mode_default = constants.MAX_ITERATIONS_BY_MODE.get(
            self._mode, constants.DEFAULT_MAX_ITERATIONS
        )
        explicit = config.ols_config.max_iterations
        if explicit is not None:
            return max(explicit, mode_default)
        return mode_default

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
