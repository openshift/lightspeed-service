"""Documentation summarizer with tool-calling support."""

import asyncio
import json
import logging
from contextlib import nullcontext
from typing import Any, AsyncGenerator, Coroutine, Optional

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
from ols.src.rag_index.solr_support import RetrievedChunk, get_openshift_docs_tool
from ols.src.skills.skills_rag import create_skill_support_tool
from ols.src.tools.offloaded_content import OffloadManager
from ols.src.tools.tool_result_inspection import (
    ToolResultClassifier,
    ToolResultInspectionError,
)
from ols.utils.audit_logger import AuditContext
from ols.utils.mcp_utils import ClientHeaders, build_mcp_config, get_mcp_tools
from ols.utils.token_handler import (
    PromptTooLongError,
    TokenBudgetTracker,
    TokenCategory,
    TokenHandler,
)

logger = logging.getLogger(__name__)


def _disable_classifier_reasoning(model: Any) -> None:
    """Strip the main request's reasoning configuration from the classifier.

    The classifier produces one small structured decision, so reasoning
    offers no benefit and its tokens count against the same output limit,
    which can truncate the decision before it is produced. Keep the main
    agent's reasoning configuration untouched.
    """
    if getattr(model, "thinking", None) is not None:
        model.thinking = None
    model_kwargs = getattr(model, "model_kwargs", None)
    if isinstance(model_kwargs, dict):
        model_kwargs.pop("thinking", None)
    extra_fields = getattr(model, "additional_model_request_fields", None)
    if isinstance(extra_fields, dict):
        extra_fields.pop("thinking", None)


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
        audit_ctx: Optional[AuditContext] = None,
        pre_retrieval_query: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the DocsSummarizer.

        Args:
            user_token: Optional user authentication token for tool access
            client_headers: Optional client-provided MCP headers for authentication
            streaming: Whether this summarizer is used for the streaming endpoint
            audit_ctx: Audit context for structured event logging
            pre_retrieval_query: The user question without attachment bodies, used
                for the pre-retrieval search. The prompt still receives the full
                query, including attachments.
            *args: Additional positional arguments passed to the parent class
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(*args, **kwargs)
        self._audit_ctx = audit_ctx
        self._prepare_llm()
        self.streaming = streaming
        self._pre_retrieval_query = pre_retrieval_query
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
        self._solr_hybrid = config.ols_config.solr_hybrid
        self._solr_client = config.solr_hybrid_search
        solr_docs_tool_active = (
            self._solr_hybrid is not None and self._solr_client is not None
        )
        self._solr_docs_tool_prompt_guidance = solr_docs_tool_active
        # Pre-retrieval and the docs tool are mutually exclusive. Running both
        # would duplicate the product-docs surface and keep the round
        # pre-retrieval exists to save, a combination that has not been
        # measured, so MCP requests keep the tool path. Gate on the configured
        # server list, not the resolved map: a server whose headers were not
        # supplied is still a configured MCP request.
        self._pre_retrieval_enabled = bool(
            solr_docs_tool_active
            and self._solr_hybrid is not None
            and self._solr_hybrid.pre_retrieval
            and self._mode == constants.QueryMode.ASK
            and not config.mcp_servers.servers
        )
        # Per-request state. DocsSummarizer is constructed per request in
        # ols/app/endpoints/ols.py; if it ever becomes a shared instance this
        # flag becomes a cross-request data leak.
        self._docs_tool_suppressed = False
        ref = config.config.ols_config.reference_content
        self._byok_active = ref is not None and bool(ref.indexes)
        self._tool_calling_enabled = bool(self.mcp_servers) or solr_docs_tool_active
        tool_result_inspection_enabled = (
            config.ols_config.guardrails.tool_result_inspection.enabled
        )
        logger.info(
            "Tool-result inspection enabled=%s, tool_calling_enabled=%s",
            tool_result_inspection_enabled,
            self._tool_calling_enabled,
        )
        if self.mcp_servers:
            logger.info("MCP servers provided: %s", list(self.mcp_servers.keys()))
        elif self._tool_calling_enabled:
            logger.info(
                "Tool calling enabled for Solr hybrid OpenShift documentation search"
            )
        else:
            logger.debug("No MCP servers provided, tool calling is disabled")

        self._tracker = TokenBudgetTracker(
            token_handler=TokenHandler(),
            context_window_size=self.model_config.context_window_size,
            max_response_tokens=self.model_config.parameters.max_tokens_for_response,
            max_tool_tokens=self.model_config.max_tokens_for_tools,
            round_cap_fraction=config.ols_config.tool_round_cap_fraction,
        )
        self._tracker.set_tool_loop_max_rounds(self._get_max_iterations())

        tool_result_classifier = None
        if self._tool_calling_enabled and tool_result_inspection_enabled:
            # No classifier-specific output-token cap: thinking models count
            # reasoning tokens against max_output_tokens, so a small cap can
            # truncate the structured decision before it is produced (OLS-4653).
            # The classifier uses the provider/model default response budget.
            classifier_llm = self.llm_loader(self.provider, self.model, {})
            if not hasattr(classifier_llm, "with_structured_output"):
                raise ToolResultInspectionError(
                    "tool-result classifier requires structured output support"
                )
            _disable_classifier_reasoning(classifier_llm)
            tool_result_classifier = ToolResultClassifier(
                classifier_llm, provider_type=self.provider_config.type
            )
            logger.info(
                "Tool-result classifier initialized for provider=%s, model=%s",
                self.provider,
                self.model,
            )

        self._tool_result_classifier = tool_result_classifier
        self._llm_agent = LLMExecutionAgent(
            bare_llm=self.bare_llm,
            model=self.model,
            provider=self.provider,
            provider_type=self.provider_config.type,
            model_config=self.model_config,
            streaming=self.streaming,
            token_budget_tracker=self._tracker,
            audit_ctx=self._audit_ctx,
            tool_result_classifier=tool_result_classifier,
        )

    async def _resolve_tools_for_request(
        self, mcp_tools_query: str
    ) -> list[StructuredTool]:
        """Load MCP tools and append built-in Solr docs search when configured for tool-only."""
        tools = await get_mcp_tools(
            mcp_tools_query, self.user_token, self.client_headers
        )
        if (
            self._solr_hybrid is not None
            and self._solr_client is not None
            and not self._docs_tool_suppressed
        ):
            tools.append(get_openshift_docs_tool(self._solr_client))
        return tools

    def _has_conversation_history(
        self,
        user_id: Optional[str],
        conversation_id: Optional[str],
        skip_user_id_check: bool,
    ) -> bool:
        """Return whether the conversation already has cached turns.

        Pre-retrieval searches with the current turn's text only, so a follow-up
        that depends on earlier turns would be grounded on unrelated passages
        and lose the docs tool. Those requests keep the tool path, which sees
        the history and can build a history-aware query.
        """
        if not (user_id and conversation_id):
            return False
        return bool(
            config.conversation_cache.get(
                user_id,
                conversation_id,
                skip_user_id_check,
            )
        )

    async def _pre_retrieve_docs(self, query: str) -> list[RetrievedChunk]:
        """Retrieve documentation passages before the first LLM call.

        Only used when ``solr_hybrid.pre_retrieval`` is enabled in ask mode and
        no other RAG retriever supplies context. The search is sized with the
        per-round tool budget the docs tool would receive, so the amount of
        documentation fetched is comparable to a tool round.

        Returns:
            Ranked passages, or an empty list when retrieval fails or has no hits.
        """
        if self._solr_client is None:
            return []
        chunks = await self._solr_client.search(
            query, token_budget=self._tracker.tools_round_budget
        )
        logger.info(
            "Pre-retrieval returned %d documentation passages for query: %.200s",
            len(chunks),
            query,
        )
        return chunks

    def _pre_retrieved_to_rag_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[RagChunk]:
        """Convert retrieved passages to prompt chunks within the prompt budget.

        Each passage keeps its source title and URL, matching how the docs tool
        presents results so the model can cite them. Passages with no text are
        skipped, and a missing URL is left out of the citation rather than
        rendered as a placeholder. A passage that does not fit the remaining
        prompt budget is skipped so a later, smaller passage can still be used.

        Search was sized with the per-round tool budget so the volume matches a
        tool round, but these tokens become prompt input, so they are charged to
        the RAG pool and capped by the prompt-side history budget. That is the
        budget that actually binds; the tool pool is left untouched because no
        tool round runs when the docs tool is suppressed.
        """
        rag_chunks: list[RagChunk] = []
        kept: list[RetrievedChunk] = []
        used = 0
        budget = self._tracker.history_budget
        for chunk in chunks:
            if not chunk.text.strip():
                logger.warning(
                    "Skipping pre-retrieved passage with no text (title=%r, url=%r)",
                    chunk.metadata.get("title"),
                    chunk.metadata.get("docs_url"),
                )
                continue
            title = str(chunk.metadata.get("title") or "unknown")
            doc_url = str(chunk.metadata.get("docs_url") or "")
            source = f'From document "{title}"'
            if doc_url:
                source += f" ({doc_url})"
            text = f"{source}\n{chunk.text}"
            tokens = self._tracker.count_tokens(text)
            if used + tokens > budget:
                continue
            rag_chunks.append(RagChunk(text=text, doc_url=doc_url, doc_title=title))
            kept.append(chunk)
            used += tokens
        self._tracker.charge(TokenCategory.RAG, used)
        logger.info(
            "Pre-retrieved context: %d/%d passages, %d tokens",
            len(rag_chunks),
            len(chunks),
            used,
        )
        if self._audit_ctx:
            self._audit_ctx.logger.rag_retrieved(
                chunk_count=len(rag_chunks),
                scores=[
                    chunk.score if chunk.score is not None else 0.0 for chunk in kept
                ],
                source_documents=[chunk.doc_url or "unknown" for chunk in rag_chunks],
            )
        return rag_chunks

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

    def _count_prompt_tokens(
        self, query: str, *, docs_tool_suppressed: bool | None = None
    ) -> int:
        """Count prompt tokens, optionally for a pre-retrieval state."""
        suppressed = (
            self._docs_tool_suppressed
            if docs_tool_suppressed is None
            else docs_tool_suppressed
        )
        temp_prompt, temp_prompt_input = GeneratePrompt(
            query,
            ["sample"],
            [AIMessage("sample")],
            self._system_prompt,
            self._tool_calling_enabled,
            self._mode,
            self._cluster_version,
            solr_docs_tool_guidance=(
                self._solr_docs_tool_prompt_guidance and not suppressed
            ),
            byok_active=self._byok_active,
            pre_retrieved_context=suppressed,
        ).generate_prompt(self.model)
        return self._tracker.count_tokens(temp_prompt.format(**temp_prompt_input))

    def _ensure_prompt_budget_for_pre_retrieval(self, query: str) -> None:
        """Reject over-budget prompts before paying for a documentation search.

        A request can fit only with the pre-retrieved supplement, so the search
        proceeds when either prompt variant fits. The fallback after a search
        that yields no usable passage keeps its own check, because switching
        back to the docs-tool prompt may not fit.
        """
        tool_tokens = self._count_prompt_tokens(query, docs_tool_suppressed=False)
        pre_retrieved_tokens = self._count_prompt_tokens(
            query, docs_tool_suppressed=True
        )
        smallest = min(tool_tokens, pre_retrieved_tokens)
        if smallest > self._tracker.prompt_budget:
            raise PromptTooLongError(
                f"Prompt length {smallest} exceeds "
                f"LLM available context window limit "
                f"{self._tracker.prompt_budget} tokens"
            )

    async def _prepare_prompt_context(
        self,
        query: str,
        rag_retriever: Optional[BaseRetriever] = None,
        pre_retrieved: Optional[list[RetrievedChunk]] = None,
    ) -> list[RagChunk]:
        """Prepare RAG context for prompt construction.

        Args:
            query: The query to be answered.
            rag_retriever: The retriever to get RAG data/context.
            pre_retrieved: Documentation passages retrieved before the LLM call.
                When they are used, the docs tool stays suppressed for the rest
                of the request unless no passage fits the prompt budget.

        Returns:
            RAG chunks truncated to fit the prompt budget.
        """
        if pre_retrieved and rag_retriever is None:
            # Optimistic: passages were found, so the prompt should use the
            # pre-retrieved supplement. Cleared below if none fit the budget.
            self._docs_tool_suppressed = True

        prompt_tokens = self._count_prompt_tokens(query)
        if prompt_tokens > self._tracker.prompt_budget:
            raise PromptTooLongError(
                f"Prompt length {prompt_tokens} exceeds "
                f"LLM available context window limit "
                f"{self._tracker.prompt_budget} tokens"
            )
        self._tracker.charge(TokenCategory.PROMPT, prompt_tokens)

        if rag_retriever:
            rag_span = (
                self._audit_ctx.span("request.rag")
                if self._audit_ctx
                else nullcontext()
            )
            with rag_span as span:
                retrieved_nodes = rag_retriever.retrieve(query)
                logger.info(
                    "Retrieved %d document nodes for RAG context",
                    len(retrieved_nodes),
                )

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

                if self._audit_ctx and span:
                    self._audit_ctx.logger.rag_retrieved(
                        chunk_count=len(rag_chunks),
                        scores=[
                            node.get_score(raise_error=False)
                            for node in retrieved_nodes[: len(rag_chunks)]
                        ],
                        source_documents=[
                            chunk.doc_url or "unknown" for chunk in rag_chunks
                        ],
                    )
        else:
            if pre_retrieved:
                rag_chunks = self._pre_retrieved_to_rag_chunks(pre_retrieved)
                if not rag_chunks:
                    # None of the passages fit the prompt budget, so the
                    # fallback contract requires the docs tool to stay bound.
                    # Recount the prompt: the tool supplement it now uses is
                    # longer than the pre-retrieved one charged above.
                    self._docs_tool_suppressed = False
                    tool_prompt_tokens = self._count_prompt_tokens(query)
                    if tool_prompt_tokens > self._tracker.prompt_budget:
                        raise PromptTooLongError(
                            f"Prompt length {tool_prompt_tokens} with docs tool "
                            f"guidance exceeds LLM available context window "
                            f"limit {self._tracker.prompt_budget} tokens"
                        )
                    self._tracker.charge(
                        TokenCategory.PROMPT, tool_prompt_tokens - prompt_tokens
                    )
                    logger.warning(
                        "Pre-retrieved documentation did not fit the prompt "
                        "budget; keeping the docs tool bound."
                    )
            else:
                logger.warning(
                    "Proceeding without RAG content. Check start up messages."
                )
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
            solr_docs_tool_guidance=(
                self._solr_docs_tool_prompt_guidance and not self._docs_tool_suppressed
            ),
            byok_active=self._byok_active,
            pre_retrieved_context=self._docs_tool_suppressed,
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
        if self._tool_result_classifier is not None:
            self._tool_result_classifier.set_quota_context(
                config.quota_limiters,
                user_id or "",
            )

        pre_retrieved: list[RetrievedChunk] = []
        if (
            self._pre_retrieval_enabled
            and rag_retriever is None
            and not self._has_conversation_history(
                user_id, conversation_id, skip_user_id_check
            )
        ):
            self._ensure_prompt_budget_for_pre_retrieval(query)
            pre_retrieved = await self._pre_retrieve_docs(
                self._pre_retrieval_query or query
            )

        rag_chunks = await self._prepare_prompt_context(
            query, rag_retriever, pre_retrieved
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
        compressed = False
        available_tokens = self._tracker.history_budget
        history_span = (
            self._audit_ctx.span("request.history")
            if self._audit_ctx
            else nullcontext()
        )
        with history_span as span:
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
                    if item.type == StreamChunkType.HISTORY_COMPRESSION_END:
                        compressed = True
                    yield item
                else:
                    history, truncated = item

            if self._audit_ctx and span:
                self._audit_ctx.logger.history_retrieved(
                    turn_count=len(history) // 2,
                    compressed=compressed,
                    truncated=truncated,
                )

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
        all_mcp_tools = await self._resolve_tools_for_request(mcp_tools_query)
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
