"""Unit tests for DocsSummarizer PR2 class."""

import logging
from contextlib import contextmanager
from typing import Callable, ClassVar
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from langchain_core.globals import get_debug, set_debug
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from ols import config
from ols.app.models.config import (
    LoggingConfig,
    MCPServerConfig,
    MCPServers,
    SolrHybridSettings,
)
from ols.app.models.models import StreamChunkType, StreamedChunk
from ols.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_ITERATIONS_TROUBLESHOOTING,
    DEFAULT_MAX_TOKENS_FOR_RESPONSE,
    QueryMode,
)
from ols.utils.config import AppConfig

# needs to be setup before importing docs_summarizer
config.ols_config.authentication_config.module = "k8s"

from ols.app.models.models import (  # noqa: E402
    CacheEntry,
    ReferencedDocument,
)
from ols.src.llms.llm_loader import load_llm  # noqa: E402
from ols.src.prompts.prompt_generator import GeneratePrompt  # noqa: E402
from ols.src.query_helpers.docs_summarizer import (  # noqa: E402
    DocsSummarizer,
    QueryHelper,
)
from ols.src.query_helpers.llm_execution_agent import (  # noqa: E402
    LLMExecutionAgent,
)
from ols.src.rag_index.solr_support import RetrievedChunk  # noqa: E402
from ols.src.tools.tool_result_inspection import (  # noqa: E402
    ToolResultInspectionError,
)
from ols.utils.logging_configurator import configure_logging  # noqa: E402
from ols.utils.mcp_utils import build_mcp_config, gather_mcp_tools  # noqa: E402
from ols.utils.token_handler import (  # noqa: E402
    PromptTooLongError,
    TokenBudgetTracker,
    TokenCategory,
    TokenHandler,
)
from tests import constants  # noqa: E402
from tests.mock_classes.mock_langchain_interface import (  # noqa: E402
    mock_langchain_interface,
)
from tests.mock_classes.mock_llm_loader import (  # noqa: E402
    MockLLMLoader,
    MockStructuredOutputLLM,
    mock_llm_loader,
)
from tests.mock_classes.mock_retrievers import MockRetriever  # noqa: E402
from tests.mock_classes.mock_tools import mock_tools_map  # noqa: E402


def check_summary_result(summary, question: str) -> None:
    """Check result produced by DocsSummarizer.create_response method."""
    assert question in summary.response
    assert isinstance(summary.rag_chunks, list)
    assert len(summary.rag_chunks) == 1
    assert (
        f"{constants.OCP_DOCS_ROOT_URL}/{constants.OCP_DOCS_VERSION}/docs/test.html"
        in summary.rag_chunks[0].doc_url
    )
    assert summary.history_truncated is False
    assert summary.tool_calls == []
    assert summary.tool_results == []


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Set up config for tests."""
    config.reload_from_yaml_file("tests/config/valid_config_without_mcp.yaml")


def test_is_query_helper_subclass():
    """Test that DocsSummarizer is a subclass of QueryHelper."""
    assert issubclass(DocsSummarizer, QueryHelper)


def test_if_system_prompt_was_updated():
    """Test if system prompt was overridden from the configuration."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert summarizer._system_prompt == config.ols_config.system_prompt


def test_tool_calling_enabled_when_solr_docs_tool_active_without_mcp():
    """Enable tool loop when Solr hybrid docs tool is active, without MCP servers."""
    hybrid = SolrHybridSettings()
    mock_client = MagicMock()
    with (
        patch.object(config.ols_config, "solr_hybrid", hybrid),
        patch.object(
            AppConfig,
            "solr_hybrid_search",
            PropertyMock(return_value=mock_client),
        ),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert summarizer._tool_calling_enabled is True


def test_tool_calling_disabled_without_mcp_and_without_solr_docs_tool():
    """Tool calling stays off when neither MCP nor Solr docs tool is active."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert summarizer._tool_calling_enabled is False


def test_debug_app_logging_does_not_enable_global_langchain_prompt_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not dump classifier prompts through LangChain's global DEBUG callbacks."""
    set_debug(False)
    monkeypatch.setattr(
        config.ols_config.logging_config, "app_log_level", logging.DEBUG
    )
    try:
        DocsSummarizer(llm_loader=mock_llm_loader(None))
        assert get_debug() is False
    finally:
        set_debug(False)


def test_tool_result_inspection_logging(caplog):
    """Log the effective tool-result inspection and tool-calling state."""
    caplog.set_level(logging.INFO)

    with patch.object(
        config.ols_config.guardrails.tool_result_inspection,
        "enabled",
        True,
    ):
        DocsSummarizer(llm_loader=mock_llm_loader(None))

    assert (
        "Tool-result inspection enabled=True, tool_calling_enabled=False" in caplog.text
    )


def test_tool_result_classifier_logging(caplog):
    """Log classifier initialization when tool inspection is active."""
    caplog.set_level(logging.INFO)

    with (
        patch.object(
            config.ols_config.guardrails.tool_result_inspection,
            "enabled",
            True,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value={"test_server": {}},
        ),
    ):
        DocsSummarizer(llm_loader=mock_llm_loader(None))

    assert "Tool-result classifier initialized" in caplog.text


def test_tool_result_classifier_does_not_force_temperature() -> None:
    """Initialize the classifier without overriding model sampling parameters."""
    loader = MagicMock(side_effect=mock_llm_loader(None))
    with (
        patch.object(
            config.ols_config.guardrails.tool_result_inspection,
            "enabled",
            True,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value={"test_server": {}},
        ),
    ):
        summarizer = DocsSummarizer(llm_loader=loader)

    assert summarizer._tool_result_classifier is not None
    # The classifier LLM must not carry a classifier-specific token cap:
    # a small max_output_tokens truncates thinking models' decisions (OLS-4653).
    loader.assert_called_with(summarizer.provider, summarizer.model, {})


@pytest.mark.parametrize(
    ("provider_type", "expect_one_of"),
    [("watsonx", True), ("openai", False)],
)
def test_tool_result_classifier_uses_configured_provider_type(
    provider_type: str, expect_one_of: bool
) -> None:
    """Select the schema by configured provider type, not by model name."""
    schemas: list[dict] = []

    def capture_schema(_llm, schema, **_kwargs):
        schemas.append(schema.model_json_schema())
        return MockStructuredOutputLLM()

    with (
        patch.object(config.llm_config.providers["p1"], "type", provider_type),
        patch.object(
            config.ols_config.guardrails.tool_result_inspection, "enabled", True
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value={"test_server": {}},
        ),
        patch.object(
            MockLLMLoader,
            "with_structured_output",
            autospec=True,
            side_effect=capture_schema,
        ),
    ):
        DocsSummarizer(llm_loader=mock_llm_loader(None))

    assert len(schemas) == 1
    assert ("oneOf" in schemas[0]) is expect_one_of


@pytest.mark.parametrize("temperature", [None, 0.0, 0.7])
def test_tool_result_classifier_uses_configured_temperature(
    temperature: float | None,
) -> None:
    """Honor model temperature without injecting a classifier-specific default."""
    provider_config = config.llm_config.providers["p1"]
    provider_config.type = "anthropic"
    provider_config.models["m1"].parameters.temperature = temperature
    with (
        patch.object(
            config.ols_config.guardrails.tool_result_inspection,
            "enabled",
            True,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value={"test_server": {}},
        ),
        patch(
            "ols.src.llms.providers.anthropic.ChatAnthropic",
            return_value=mock_llm_loader(None)(),
        ) as mock_chat_anthropic,
    ):
        summarizer = DocsSummarizer(llm_loader=load_llm)

    assert summarizer._tool_result_classifier is not None
    assert mock_chat_anthropic.call_count == 2
    # No classifier-specific cap: the provider default response budget applies
    # to the classifier like any other LLM (OLS-4653).
    assert (
        mock_chat_anthropic.call_args.kwargs["max_tokens"]
        == DEFAULT_MAX_TOKENS_FOR_RESPONSE
    )
    for call in mock_chat_anthropic.call_args_list:
        kwargs = call.kwargs
        if temperature is None:
            assert "temperature" not in kwargs
        else:
            assert kwargs["temperature"] == temperature
        assert "top_p" not in kwargs


def test_tool_result_inspection_fails_closed_without_structured_output():
    """Reject tool-loop initialization when the classifier lacks structured output."""

    def loader(*args, **kwargs):
        return type("UnsupportedLLM", (), {"provider": "mock", "model": "mock"})()

    with (
        patch.object(
            config.ols_config.guardrails.tool_result_inspection,
            "enabled",
            True,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value={"test_server": {}},
        ),
    ):
        with pytest.raises(ToolResultInspectionError, match="structured output"):
            DocsSummarizer(llm_loader=loader)


def test_tool_result_inspection_disabled_skips_classifier():
    """Skip classifier construction when inspection is explicitly disabled."""

    def loader(*args, **kwargs):
        return type("UnsupportedLLM", (), {"provider": "mock", "model": "mock"})()

    with (
        patch.object(
            config.ols_config.guardrails.tool_result_inspection,
            "enabled",
            False,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value={"test_server": {}},
        ),
    ):
        summarizer = DocsSummarizer(llm_loader=loader)

    assert summarizer._tool_result_classifier is None


def test_summarize_empty_history():
    """Basic test for DocsSummarizer using mocked retriever and empty history."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 1),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        summary = summarizer.create_response(question, MockRetriever(), [])
        check_summary_result(summary, question)


def test_summarize_no_history():
    """Basic test for DocsSummarizer without explicit history argument."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        summary = summarizer.create_response(question, MockRetriever())
        check_summary_result(summary, question)


def test_summarize_history_provided():
    """Basic test with explicit history vs default history paths."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
        patch("ols.config.conversation_cache.get") as mock_cache_get,
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()

        mock_cache_get.return_value = [
            CacheEntry(query=HumanMessage("What is Kubernetes?"))
        ]
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary1 = summarizer.create_response(
                question, rag_retriever, "user-id", "conv-id"
            )
            # Non-overflow path returns early from prepare_history (no second limit pass).
            token_handler.assert_not_called()
            check_summary_result(summary1, "What is Kubernetes?")

        mock_cache_get.return_value = []
        with patch(
            "ols.src.query_helpers.docs_summarizer.TokenHandler.limit_conversation_history",
            return_value=([], False),
        ) as token_handler:
            summary2 = summarizer.create_response(
                question, rag_retriever, "user-id", "conv-id2"
            )
            token_handler.assert_not_called()
            check_summary_result(summary2, question)


def test_summarize_truncation():
    """Basic test for DocsSummarizer to check compression avoids truncation."""
    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.config.conversation_cache.get") as mock_cache_get,
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        rag_retriever = MockRetriever()

        history = [
            CacheEntry(
                query=HumanMessage("What is Kubernetes?" * 100),
                response=AIMessage(
                    "Kubernetes is a container orchestration system." * 100
                ),
            )
        ] * 100
        mock_cache_get.return_value = history

        summary = summarizer.create_response(
            question, rag_retriever, "user-id", "conv-id"
        )

        assert not summary.history_truncated


def test_summarize_no_reference_content():
    """Basic test when no retriever is provided."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    summary = summarizer.create_response(question)
    assert question in summary.response
    assert summary.rag_chunks == []
    assert not summary.history_truncated


def test_byok_active_true_when_reference_content_configured():
    """Verify _byok_active is True when BYOK indexes are configured."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("resp")())
    )
    assert summarizer._byok_active


def test_byok_active_false_when_no_reference_content():
    """Verify _byok_active is False when no BYOK indexes are configured."""
    original = config.config.ols_config.reference_content
    try:
        config.config.ols_config.reference_content = None
        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(mock_langchain_interface("resp")())
        )
        assert not summarizer._byok_active
    finally:
        config.config.ols_config.reference_content = original


def test_summarize_retrieval_logging(caplog):
    """Basic test to ensure retrieval details are visible in logs."""
    logging_config = LoggingConfig(app_log_level="debug")
    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]

    with (
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        question = "What's the ultimate question with answer 42?"
        summary = summarizer.create_response(question, MockRetriever())
        check_summary_result(summary, question)
        assert "Retrieved 1 document nodes for RAG context" in caplog.text


@pytest.mark.asyncio
async def test_resolve_tools_appends_openshift_docs_tool_when_solr_configured_no_mcp():
    """Append docs tool for Solr hybrid when MCP returns no tools."""
    with patch(
        "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
        new_callable=AsyncMock,
    ) as m_get:
        m_get.return_value = []
        prev = config.ols_config.solr_hybrid
        config.ols_config.solr_hybrid = SolrHybridSettings()
        config._cached_solr_hybrid_search = MagicMock()
        config._solr_hybrid_initialized = True
        try:
            summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
            tools = await summarizer._resolve_tools_for_request("query")
        finally:
            config.ols_config.solr_hybrid = prev
            config._cached_solr_hybrid_search = None
            config._solr_hybrid_initialized = False
    assert [t.name for t in tools] == ["search_openshift_documentation"]
    m_get.assert_awaited_once()


@contextmanager
def _solr_pre_retrieval_source(
    search: AsyncMock | None = None, pre_retrieval: bool = True
):
    """Patch Solr hybrid config and search client for pre-retrieval tests."""
    client = MagicMock()
    client.search = search or AsyncMock(return_value=[])
    hybrid = SolrHybridSettings(pre_retrieval=pre_retrieval)
    with (
        patch.object(config.ols_config, "solr_hybrid", hybrid),
        patch.object(
            AppConfig, "solr_hybrid_search", PropertyMock(return_value=client)
        ),
    ):
        yield client


@contextmanager
def _spy_generate_prompt():
    """Record GeneratePrompt calls so tests can inspect the built prompt."""
    calls: list[tuple] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return GeneratePrompt(*args, **kwargs)

    with patch("ols.src.query_helpers.docs_summarizer.GeneratePrompt", side_effect=spy):
        yield calls


@contextmanager
def _capture_agent_execute():
    """Record the arguments the summarizer passes to the tool loop."""
    executions: list[dict] = []
    original = LLMExecutionAgent.execute

    async def capture(self, *args, **kwargs):
        executions.append(kwargs)
        async for chunk in original(self, *args, **kwargs):
            yield chunk

    with patch.object(LLMExecutionAgent, "execute", new=capture):
        yield executions


def _count_tokens_by_variant(
    summarizer: DocsSummarizer, *, tool_tokens: int, pre_tokens: int
) -> Callable[..., int]:
    """Return a token counter with distinct values per prompt variant."""

    def count(_query: str, *, docs_tool_suppressed: bool | None = None) -> int:
        suppressed = (
            summarizer._docs_tool_suppressed
            if docs_tool_suppressed is None
            else docs_tool_suppressed
        )
        return pre_tokens if suppressed else tool_tokens

    return count


def _render_system_prompt(call, model: str = "mock_model") -> str:
    """Render the system message template captured from a GeneratePrompt call."""
    args, kwargs = call
    prompt, _ = GeneratePrompt(*args, **kwargs).generate_prompt(model)
    return prompt.messages[0].prompt.template


def _pre_retrieval_chunk(
    text: str = "passage",
    title: str = "Install",
    url: str = "https://docs/1",
    score: float = 1.0,
) -> RetrievedChunk:
    """Build a retrieved chunk with the metadata the docs tool would return."""
    return RetrievedChunk(
        text=text, score=score, metadata={"title": title, "docs_url": url}
    )


@pytest.mark.asyncio
async def test_pre_retrieval_disabled_by_default_keeps_docs_tool():
    """With the flag off, search is skipped and the docs tool stays bound."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search, pre_retrieval=False),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        _spy_generate_prompt() as calls,
        _capture_agent_execute() as executions,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response("install a cluster"):
            pass

    search.assert_not_awaited()
    assert summarizer._docs_tool_suppressed is False
    assert len(executions) == 1
    assert [t.name for t in executions[0]["all_mcp_tools"]] == [
        "search_openshift_documentation"
    ]
    template = _render_system_prompt(calls[-1])
    assert "Solr docs tool:" in template
    assert "Documentation context (already retrieved):" not in template


def test_pre_retrieval_disabled_by_default():
    """Pre-retrieval stays off unless explicitly configured."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert summarizer._pre_retrieval_enabled is False


def test_pre_retrieval_enabled_in_ask_mode():
    """Pre-retrieval is active in ask mode when Solr hybrid enables it."""
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert summarizer._pre_retrieval_enabled is True


def test_pre_retrieval_disabled_when_mcp_configured_even_if_headers_missing():
    """Configured MCP servers keep the tool path even when headers are absent."""
    configured = MCPServers(
        servers=[MCPServerConfig(name="test_server", url="http://mcp.example/mcp")]
    )
    with (
        _solr_pre_retrieval_source(),
        patch.object(config.config, "mcp_servers", configured),
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value={},
        ),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert summarizer.mcp_servers == {}
    assert summarizer._pre_retrieval_enabled is False


@pytest.mark.asyncio
async def test_pre_retrieval_inert_outside_ask_mode():
    """Troubleshooting mode never pre-retrieves product documentation."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
        )
        assert summarizer._pre_retrieval_enabled is False
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response("install a cluster"):
            pass
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_retrieve_docs_uses_round_budget():
    """Retrieval receives the configured query and the per-round tool budget."""
    chunk = _pre_retrieval_chunk(text="passage one", score=1.5)
    search = AsyncMock(return_value=[chunk])
    with _solr_pre_retrieval_source(search):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    chunks = await summarizer._pre_retrieve_docs("install a cluster")
    assert chunks == [chunk]
    assert search.await_args.args[0] == "install a cluster"
    assert (
        search.await_args.kwargs["token_budget"]
        == summarizer._tracker.tools_round_budget
    )


def test_pre_retrieved_chunks_carry_source_and_url():
    """Passages keep the docs tool's title and URL so the model can cite them."""
    chunk = _pre_retrieval_chunk(score=2.0)
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    rag_chunks = summarizer._pre_retrieved_to_rag_chunks([chunk, chunk])
    assert len(rag_chunks) == 2
    assert rag_chunks[0].text.startswith('From document "Install" (https://docs/1)')
    assert rag_chunks[0].doc_title == "Install"
    assert rag_chunks[0].doc_url == "https://docs/1"
    assert summarizer._tracker.usage(TokenCategory.RAG) == 2 * (
        summarizer._tracker.count_tokens(rag_chunks[0].text)
    )


@pytest.mark.asyncio
async def test_pre_retrieve_docs_returns_empty_without_client():
    """The retrieval helper is inert when no Solr client exists."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    assert await summarizer._pre_retrieve_docs("install a cluster") == []


def test_pre_retrieved_chunks_log_audit_event():
    """Injected passages are reported through the audit logger."""
    audit_ctx = MagicMock()
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(None), audit_ctx=audit_ctx
        )
    summarizer._pre_retrieved_to_rag_chunks(
        [_pre_retrieval_chunk(score=0.75), _pre_retrieval_chunk(score=None)]
    )
    audit_ctx.logger.rag_retrieved.assert_called_once_with(
        chunk_count=2,
        scores=[0.75, 0.0],
        source_documents=["https://docs/1", "https://docs/1"],
    )


def test_pre_retrieved_chunks_drop_passages_over_budget():
    """Passages that no longer fit are dropped; earlier passages are kept."""
    first = _pre_retrieval_chunk(text="first passage", url="https://docs/1")
    second = _pre_retrieval_chunk(text="second passage", url="https://docs/2")
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    first_tokens = summarizer._tracker.count_tokens(
        f'From document "Install" (https://docs/1)\n{first.text}'
    )
    summarizer._tracker.charge(
        TokenCategory.PROMPT,
        summarizer._tracker.prompt_budget - first_tokens - 1,
    )

    rag_chunks = summarizer._pre_retrieved_to_rag_chunks([first, second])
    assert len(rag_chunks) == 1
    assert rag_chunks[0].doc_url == "https://docs/1"
    assert "first passage" in rag_chunks[0].text


def test_pre_retrieved_chunks_skip_oversized_passage_and_keep_later_one():
    """A passage that does not fit is skipped so later passages can be used."""
    oversized = _pre_retrieval_chunk(text="big " * 100, url="https://docs/big")
    later = _pre_retrieval_chunk(text="small", url="https://docs/small")
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    later_tokens = summarizer._tracker.count_tokens(
        'From document "Install" (https://docs/small)\nsmall'
    )
    with patch.object(
        TokenBudgetTracker,
        "history_budget",
        new_callable=PropertyMock,
        return_value=later_tokens + 5,
    ):
        rag_chunks = summarizer._pre_retrieved_to_rag_chunks([oversized, later])

    assert len(rag_chunks) == 1
    assert rag_chunks[0].doc_url == "https://docs/small"


@pytest.mark.asyncio
async def test_pre_retrieval_suppresses_docs_tool_and_injects_passages():
    """Passages replace the docs tool and reach the prompt with title and URL."""
    chunk = _pre_retrieval_chunk(
        text="Run the installer.",
        title="Installing a cluster",
        url="https://docs.example/install",
    )
    search = AsyncMock(return_value=[chunk])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        patch(
            "ols.src.query_helpers.docs_summarizer.get_openshift_docs_tool"
        ) as mock_get_docs_tool,
        _spy_generate_prompt() as calls,
        _capture_agent_execute() as executions,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response("install a cluster"):
            pass
        assert summarizer._docs_tool_suppressed is True

    search.assert_awaited_once()
    assert search.await_args.args[0] == "install a cluster"
    assert len(executions) == 1
    assert executions[0]["all_mcp_tools"] == []
    mock_get_docs_tool.assert_not_called()

    template = _render_system_prompt(calls[-1])
    assert "Documentation context (already retrieved):" in template
    assert "Solr docs tool:" not in template
    assert calls[-1][0][1] == [
        'From document "Installing a cluster" '
        "(https://docs.example/install)\nRun the installer."
    ]
    _, final_kwargs = calls[-1]
    assert final_kwargs["pre_retrieved_context"] is True
    assert final_kwargs["solr_docs_tool_guidance"] is False


@pytest.mark.asyncio
async def test_pre_retrieval_falls_back_to_docs_tool_without_hits():
    """An empty retrieval keeps the docs tool so the model can still search."""
    search = AsyncMock(return_value=[])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        _spy_generate_prompt() as calls,
        _capture_agent_execute() as executions,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response("install a cluster"):
            pass
        assert summarizer._docs_tool_suppressed is False

    search.assert_awaited_once()
    assert len(executions) == 1
    assert [t.name for t in executions[0]["all_mcp_tools"]] == [
        "search_openshift_documentation"
    ]
    template = _render_system_prompt(calls[-1])
    assert "Solr docs tool:" in template
    assert "Documentation context (already retrieved):" not in template
    _, final_kwargs = calls[-1]
    assert final_kwargs["pre_retrieved_context"] is False
    assert final_kwargs["solr_docs_tool_guidance"] is True


@pytest.mark.asyncio
async def test_pre_retrieval_falls_back_when_no_passage_fits_budget():
    """Hits that do not fit the prompt budget restore the docs tool."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        patch.object(
            TokenBudgetTracker,
            "history_budget",
            new_callable=PropertyMock,
            return_value=0,
        ),
        _spy_generate_prompt() as calls,
        _capture_agent_execute() as executions,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response("install a cluster"):
            pass
        assert summarizer._docs_tool_suppressed is False

    search.assert_awaited_once()
    assert len(executions) == 1
    assert [t.name for t in executions[0]["all_mcp_tools"]] == [
        "search_openshift_documentation"
    ]
    template = _render_system_prompt(calls[-1])
    assert "Solr docs tool:" in template
    assert "Documentation context (already retrieved):" not in template


@pytest.mark.asyncio
async def test_pre_retrieval_budget_fallback_recharges_prompt_for_tool_guidance():
    """The prompt charge follows the supplement actually used after fallback."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        patch.object(
            TokenBudgetTracker,
            "history_budget",
            new_callable=PropertyMock,
            return_value=0,
        ),
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536

        with patch.object(
            summarizer,
            "_count_prompt_tokens",
            side_effect=_count_tokens_by_variant(
                summarizer, tool_tokens=250, pre_tokens=100
            ),
        ) as count_prompt_tokens_mock:
            async for _ in summarizer.generate_response("install a cluster"):
                pass

    assert count_prompt_tokens_mock.call_count == 4
    assert summarizer._tracker.usage(TokenCategory.PROMPT) == 250


@pytest.mark.asyncio
async def test_pre_retrieval_rejects_over_budget_prompt_before_search():
    """No search is paid for when neither prompt variant fits the budget."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536
        with patch.object(summarizer, "_count_prompt_tokens", return_value=100_000):
            with pytest.raises(PromptTooLongError, match="exceeds"):
                async for _ in summarizer.generate_response("install a cluster"):
                    pass

    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_retrieval_runs_when_only_pre_retrieved_prompt_fits():
    """A prompt that fits only without the tool guidance still searches."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        _capture_agent_execute() as executions,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536

        with patch.object(
            summarizer,
            "_count_prompt_tokens",
            side_effect=_count_tokens_by_variant(
                summarizer, tool_tokens=100_000, pre_tokens=100
            ),
        ):
            async for _ in summarizer.generate_response("install a cluster"):
                pass
        assert summarizer._docs_tool_suppressed is True

    search.assert_awaited_once()
    assert len(executions) == 1
    assert executions[0]["all_mcp_tools"] == []


@pytest.mark.asyncio
async def test_pre_retrieval_fallback_rejects_tool_prompt_over_budget():
    """A request that fits only with passages fails if none can be injected."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        patch.object(
            TokenBudgetTracker,
            "history_budget",
            new_callable=PropertyMock,
            return_value=0,
        ),
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536

        with patch.object(
            summarizer,
            "_count_prompt_tokens",
            side_effect=_count_tokens_by_variant(
                summarizer, tool_tokens=100_000, pre_tokens=100
            ),
        ):
            with pytest.raises(PromptTooLongError, match="with docs tool guidance"):
                async for _ in summarizer.generate_response("install a cluster"):
                    pass

    search.assert_awaited_once()


@pytest.mark.asyncio
async def test_pre_retrieval_searches_question_without_attachments():
    """Solr gets the attachment-free question; the answer prompt gets both."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    question = "why is this pod failing?"
    attachment = "\n\n```yaml\nkind: Pod\nmetadata:\n  name: internal-id\n```"
    full_query = question + attachment
    with (
        _solr_pre_retrieval_source(search),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        _spy_generate_prompt() as calls,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(None),
            pre_retrieval_query=question,
        )
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response(full_query):
            pass

    search.assert_awaited_once()
    assert search.await_args.args[0] == question
    assert calls[-1][0][0] == full_query


@pytest.mark.asyncio
async def test_pre_retrieval_skips_follow_up_turns():
    """Follow-ups keep the docs tool so search can use conversation history."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        patch.object(
            DocsSummarizer, "_has_conversation_history", return_value=True
        ) as has_history,
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new_callable=AsyncMock,
        ) as mock_get_mcp_tools,
        _spy_generate_prompt() as calls,
    ):
        mock_get_mcp_tools.side_effect = lambda *args, **kwargs: []
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response("how do I do that?"):
            pass

    has_history.assert_called_once()
    search.assert_not_awaited()
    assert summarizer._docs_tool_suppressed is False
    assert "Solr docs tool:" in _render_system_prompt(calls[-1])


def test_has_conversation_history_reads_cache():
    """Cached turns mark the request as a follow-up."""
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    with patch(
        "ols.config.conversation_cache.get",
        return_value=[CacheEntry(query=HumanMessage("earlier turn"))],
    ) as cache_get:
        assert summarizer._has_conversation_history("user", "conv", False) is True
    cache_get.assert_called_once_with("user", "conv", False)

    with patch("ols.config.conversation_cache.get", return_value=[]):
        assert summarizer._has_conversation_history("user", "conv", False) is False
    assert summarizer._has_conversation_history(None, "conv", False) is False


def test_pre_retrieved_chunks_skip_blank_passages():
    """Hits without passage text never reach the prompt or suppress the tool."""
    blank = _pre_retrieval_chunk(text="   ")
    usable = _pre_retrieval_chunk(text="real content", url="https://docs/2")
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    rag_chunks = summarizer._pre_retrieved_to_rag_chunks([blank, usable])
    assert len(rag_chunks) == 1
    assert "real content" in rag_chunks[0].text
    assert rag_chunks[0].doc_url == "https://docs/2"


def test_pre_retrieved_chunks_omit_missing_docs_url():
    """A passage without a URL still grounds the answer but is not cited."""
    chunk = _pre_retrieval_chunk(text="real content", url="")
    with _solr_pre_retrieval_source():
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    rag_chunks = summarizer._pre_retrieved_to_rag_chunks([chunk])
    assert len(rag_chunks) == 1
    assert "real content" in rag_chunks[0].text
    assert "unknown" not in rag_chunks[0].text
    assert rag_chunks[0].doc_url == ""
    assert ReferencedDocument.from_rag_chunks(rag_chunks) == []


@pytest.mark.asyncio
async def test_pre_retrieval_skipped_when_retriever_supplied():
    """A configured RAG retriever takes precedence over Solr pre-retrieval."""
    search = AsyncMock(return_value=[_pre_retrieval_chunk()])
    with (
        _solr_pre_retrieval_source(search),
        _spy_generate_prompt() as calls,
        patch("ols.utils.token_handler.RAG_SIMILARITY_CUTOFF", 0.4),
        patch("ols.utils.token_handler.MINIMUM_CONTEXT_TOKEN_LIMIT", 3),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tracker.context_window_size = 65536
        async for _ in summarizer.generate_response(
            "install a cluster", rag_retriever=MockRetriever()
        ):
            pass

    search.assert_not_awaited()
    assert summarizer._docs_tool_suppressed is False
    rag_context = calls[-1][0][1]
    assert any("a text text text text" in text for text in rag_context)


@pytest.mark.asyncio
async def test_response_generator():
    """Test response generator method."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"
    generated_content = ""

    async for item in summarizer.generate_response(question):
        generated_content += item.text

    assert generated_content == question


@pytest.mark.asyncio
async def test_response_generator_emits_history_events_before_tokens():
    """Test history compression events are emitted in order before text chunks."""
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(mock_langchain_interface("test response")())
    )
    question = "What's the ultimate question with answer 42?"

    async def mock_prepare_history(**kwargs):
        yield StreamedChunk(
            type=StreamChunkType.HISTORY_COMPRESSION_START,
            data={"status": "started"},
        )
        yield StreamedChunk(
            type=StreamChunkType.HISTORY_COMPRESSION_END,
            data={"status": "completed", "duration_ms": 1.0},
        )
        yield ([], False)

    chunk_types: list[StreamChunkType] = []

    with patch(
        "ols.src.query_helpers.docs_summarizer.prepare_history",
        side_effect=mock_prepare_history,
    ):
        chunk_types.extend(
            [item.type async for item in summarizer.generate_response(question)]
        )

    assert StreamChunkType.HISTORY_COMPRESSION_START in chunk_types
    assert StreamChunkType.HISTORY_COMPRESSION_END in chunk_types
    assert StreamChunkType.TEXT in chunk_types
    assert chunk_types.index(
        StreamChunkType.HISTORY_COMPRESSION_START
    ) < chunk_types.index(StreamChunkType.HISTORY_COMPRESSION_END)
    assert chunk_types.index(
        StreamChunkType.HISTORY_COMPRESSION_END
    ) < chunk_types.index(StreamChunkType.TEXT)


async def async_mock_invoke(yield_values):
    """Mock async invoke_llm function to simulate LLM behavior."""
    for value in yield_values:
        yield value


def test_tool_calling_one_iteration():
    """Test tool calling - stops after one iteration."""
    with patch(
        "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
    ) as mock_invoke:
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [AIMessageChunk(content="XYZ", response_metadata={"finish_reason": "stop"})]
        )
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summarizer.create_response("How many namespaces are there in my cluster?")
        assert mock_invoke.call_count == 1


def test_tool_calling_drains_chunks_after_stop():
    """Test that chunks after finish_reason=stop are consumed but not forwarded."""
    question = "How many namespaces are there in my cluster?"

    with patch(
        "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
    ) as mock_invoke:
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [
                AIMessageChunk(content="Hello", response_metadata={}),
                AIMessageChunk(content="", response_metadata={"finish_reason": "stop"}),
                AIMessageChunk(content="trailing1", response_metadata={}),
                AIMessageChunk(content="trailing2", response_metadata={}),
            ]
        )
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer._tool_calling_enabled = True
        summary = summarizer.create_response(question)
        assert mock_invoke.call_count == 1
        assert "Hello" in summary.response
        assert "trailing" not in summary.response


async def fake_invoke_llm(*args, **kwargs):
    """Fake invoke_llm function to simulate two-turn LLM behavior."""
    if not hasattr(fake_invoke_llm, "call_count"):
        fake_invoke_llm.call_count = 0
    fake_invoke_llm.call_count += 1

    if fake_invoke_llm.call_count == 1:
        yield AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_call_chunks=[
                {
                    "name": "get_namespaces_mock",
                    "args": "{}",
                    "id": "call_1",
                    "index": 0,
                }
            ],
        )
    elif fake_invoke_llm.call_count == 2:
        yield AIMessageChunk(content="XYZ", response_metadata={"finish_reason": "stop"})


def test_tool_calling_two_iteration():
    """Test tool calling - stops after two iterations."""
    with (
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm",
            new=fake_invoke_llm,
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
    ):
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.model_config.context_window_size = 1000
        summarizer._tracker.context_window_size = 1000
        summarizer._tool_calling_enabled = True
        summarizer.create_response("How many namespaces are there in my cluster?")
        assert mock_invoke.call_count == 2


def test_tool_calling_force_stop():
    """Test tool calling - force stop by max rounds."""
    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._get_max_iterations",
            return_value=3,
        ),
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
    ):
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [
                AIMessageChunk(
                    content="",
                    response_metadata={"finish_reason": "tool_calls"},
                    tool_call_chunks=[
                        {
                            "name": "get_namespaces_mock",
                            "args": "{}",
                            "id": "call_1",
                            "index": 0,
                        }
                    ],
                )
            ]
        )
        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.model_config.context_window_size = 1000
        summarizer._tracker.context_window_size = 1000
        summarizer._tool_calling_enabled = True
        summarizer.create_response("How many namespaces are there in my cluster?")
        assert mock_invoke.call_count == 3


def test_tool_calling_tool_execution(caplog):
    """Test tool execution path with one valid and one invalid tool call."""
    caplog.set_level(10)
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._get_max_iterations",
            return_value=2,
        ),
        patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.TokenBudgetTracker.tools_round_budget",
            new_callable=PropertyMock,
            return_value=1000,
        ),
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
        ) as mock_invoke,
        patch(
            "ols.src.query_helpers.docs_summarizer.build_mcp_config",
            return_value=mcp_servers_config,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
            new=AsyncMock(return_value=mock_tools_map),
        ),
    ):
        mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
            [
                AIMessageChunk(
                    content="",
                    response_metadata={"finish_reason": "tool_calls"},
                    tool_calls=[
                        {"name": "get_namespaces_mock", "args": {}, "id": "call_id1"},
                        {"name": "invalid_function_name", "args": {}, "id": "call_id2"},
                    ],
                )
            ]
        )

        mock_mcp_client_instance = AsyncMock()
        mock_mcp_client_instance.get_tools.return_value = mock_tools_map
        mock_mcp_client_cls.return_value = mock_mcp_client_instance

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.model_config.max_tokens_for_tools = 100
        summarizer.model_config.context_window_size = 1000
        summarizer.model_config.parameters.max_tokens_for_response = 100
        summarizer._tracker.context_window_size = 1000
        summarizer.create_response("How many namespaces are there in my cluster?")

        assert "get_namespaces_mock" in caplog.text
        assert "invalid_function_name" in caplog.text
        assert mock_invoke.call_count == 2


def test_build_final_prompt_raises_when_tool_definitions_exceed_prompt_budget() -> None:
    """Tool definitions token estimate is validated with the final prompt budget."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    summarizer._tracker = TokenBudgetTracker(
        token_handler=TokenHandler(),
        context_window_size=1000,
        max_response_tokens=100,
        max_tool_tokens=200,
        round_cap_fraction=0.6,
    )
    summarizer._tracker.set_tool_loop_max_rounds(5)
    summarizer._tracker.charge(TokenCategory.PROMPT, 650)
    with pytest.raises(PromptTooLongError, match="Tool definitions"):
        summarizer._build_final_prompt(
            query="q",
            history=[],
            rag_chunks=[],
            skill_content=None,
            tool_definitions_tokens=100,
        )


def test_tool_output_token_tracking_uses_buffer_weight(caplog):
    """Test that tool output tokens are counted with TOKEN_BUFFER_WEIGHT like other budget items.

    Before this fix, raw len(tokens) was used for tool outputs while tool definitions
    and AIMessage tokens used _get_token_count() (which applies a 1.1x buffer).
    This test asserts _get_token_count() is called for tool output tokens by spying on
    it: with one tool call in one round it must be called at least 3 times
    (tool definitions, AIMessage, tool output).
    """
    mcp_servers_config = {
        "test_server": {
            "transport": "streamable_http",
            "url": "http://test-server:8080/mcp",
        },
    }

    original_get_token_count = TokenHandler._get_token_count
    call_count = 0

    def counting_get_token_count(tokens: list) -> int:
        nonlocal call_count
        call_count += 1
        return original_get_token_count(tokens)

    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._get_max_iterations",
            return_value=2,
        ),
        patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.llm_execution_agent.LLMExecutionAgent._invoke_llm"
        ) as mock_invoke,
        patch("ols.utils.mcp_utils.config") as mock_config,
        patch.object(
            TokenHandler, "_get_token_count", staticmethod(counting_get_token_count)
        ),
    ):
        mock_config.tools_rag = None
        mock_config.mcp_servers.servers = [MagicMock()]

        with patch(
            "ols.utils.mcp_utils._gather_and_populate_tools",
            new=AsyncMock(return_value=(mcp_servers_config, mock_tools_map)),
        ):
            mock_invoke.side_effect = lambda *args, **kwargs: async_mock_invoke(
                [
                    AIMessageChunk(
                        content="",
                        response_metadata={"finish_reason": "tool_calls"},
                        tool_calls=[
                            {
                                "name": "get_namespaces_mock",
                                "args": {},
                                "id": "call_id1",
                            },
                        ],
                    )
                ]
            )

        mock_mcp_client_instance = AsyncMock()
        mock_mcp_client_instance.get_tools.return_value = mock_tools_map
        mock_mcp_client_cls.return_value = mock_mcp_client_instance

        summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
        summarizer.model_config.max_tokens_for_tools = 50000
        summarizer.create_response("How many namespaces?")

    # _get_token_count must be called for:
    #   1. tool definitions (once at the start of the loop)
    #   2. AIMessage with tool_calls
    #   3. tool output (the change introduced by this fix)
    assert call_count >= 3, (
        f"Expected _get_token_count to be called at least 3 times "
        f"(definitions + AIMessage + tool output), got {call_count}"
    )


@pytest.mark.asyncio
async def test_gather_mcp_tools_failure_isolation(caplog):
    """Test gather_mcp_tools isolates failures from individual MCP servers."""
    caplog.set_level(10)
    mcp_servers = {
        "working_server": {
            "transport": "streamable_http",
            "url": "http://working-server:8080/mcp",
        },
        "broken_server": {
            "transport": "streamable_http",
            "url": "http://non-exist:8888/mcp",
        },
    }

    async def mock_get_tools(server_name=None):
        if server_name == "working_server":
            return mock_tools_map
        raise ConnectionError("Failed to connect to http://non-exist:8888/mcp")

    with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
        mock_client_instance = AsyncMock()
        mock_client_instance.get_tools.side_effect = mock_get_tools
        mock_client_cls.return_value = mock_client_instance

        tools = await gather_mcp_tools(mcp_servers)
        assert len(tools) == 1
        assert tools[0].name == "get_namespaces_mock"
        assert "Loaded 1 tools from MCP server 'working_server'" in caplog.text
        assert "Failed to get tools from MCP server 'broken_server'" in caplog.text


def test_build_mcp_config_transport_is_streamable_http():
    """Test build_mcp_config sets transport to streamable_http for all servers."""
    server1 = MCPServerConfig(name="server1", url="http://server1:8080/mcp")
    server1._resolved_headers = {}
    server2 = MCPServerConfig(name="server2", url="http://server2:9090/mcp", timeout=30)
    server2._resolved_headers = {}

    mcp_config = build_mcp_config([server1, server2], None, None)

    assert mcp_config["server1"]["transport"] == "streamable_http"
    assert mcp_config["server2"]["transport"] == "streamable_http"


def test_get_max_iterations_ask_mode_no_override():
    """Test _get_max_iterations returns ASK default when config has no override."""
    config.ols_config.max_iterations = None
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None), mode=QueryMode.ASK)
    assert summarizer._get_max_iterations() == DEFAULT_MAX_ITERATIONS


def test_get_max_iterations_troubleshooting_mode_no_override():
    """Test _get_max_iterations returns TROUBLESHOOTING default when config has no override."""
    config.ols_config.max_iterations = None
    summarizer = DocsSummarizer(
        llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
    )
    assert summarizer._get_max_iterations() == DEFAULT_MAX_ITERATIONS_TROUBLESHOOTING


def test_get_max_iterations_config_override_above_default():
    """Test _get_max_iterations uses config value when it exceeds the mode default."""
    config.ols_config.max_iterations = 40
    try:
        for mode in (QueryMode.ASK, QueryMode.TROUBLESHOOTING):
            summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None), mode=mode)
            assert summarizer._get_max_iterations() == 40
    finally:
        config.ols_config.max_iterations = None


def test_get_max_iterations_config_override_below_default():
    """Test _get_max_iterations uses mode default when it exceeds the config value."""
    config.ols_config.max_iterations = 10
    try:
        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(None), mode=QueryMode.ASK
        )
        assert summarizer._get_max_iterations() == 10

        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(None), mode=QueryMode.TROUBLESHOOTING
        )
        assert (
            summarizer._get_max_iterations() == DEFAULT_MAX_ITERATIONS_TROUBLESHOOTING
        )
    finally:
        config.ols_config.max_iterations = None


def test_create_response_raises_on_unknown_chunk_type():
    """Test create_response raises ValueError on unsupported chunk type."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))

    class UnknownChunk:
        type = "unsupported"
        text = ""
        data: ClassVar[dict[str, str]] = {}

    async def _fake_generate(self, *args, **kwargs):
        yield UnknownChunk()

    with patch.object(DocsSummarizer, "generate_response", _fake_generate):
        with pytest.raises(ValueError, match="Unknown chunk type"):
            summarizer.create_response("q")


def test_create_response_ignores_reasoning_chunks():
    """Test create_response skips reasoning chunks without error."""
    summarizer = DocsSummarizer(llm_loader=mock_llm_loader(None))
    from ols.app.models.models import StreamedChunk

    async def _fake_generate(self, *args, **kwargs):
        yield StreamedChunk(type=StreamChunkType.REASONING, text="thinking")
        yield StreamedChunk(type=StreamChunkType.TEXT, text="answer")
        yield StreamedChunk(
            type=StreamChunkType.END,
            data={"rag_chunks": [], "truncated": False, "token_counter": None},
        )

    with patch.object(DocsSummarizer, "generate_response", _fake_generate):
        result = summarizer.create_response("q")

    assert result.response == "answer"


@pytest.mark.asyncio
async def test_generate_response_creates_and_cleans_offload_manager():
    """Test OffloadManager lifecycle in generate_response for both streaming modes."""
    for streaming in (False, True):
        created_managers = []
        cleanup_calls = []

        original_init = __import__(
            "ols.src.tools.offloaded_content", fromlist=["OffloadManager"]
        ).OffloadManager.__init__

        def _tracking_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            created_managers.append(self)
            original_cleanup = self.cleanup

            def _tracked_cleanup():
                cleanup_calls.append(self)
                original_cleanup()

            self.cleanup = _tracked_cleanup

        summarizer = DocsSummarizer(
            llm_loader=mock_llm_loader(mock_langchain_interface("test response")()),
            streaming=streaming,
        )
        summarizer._tool_calling_enabled = True

        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.get_mcp_tools",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "ols.src.tools.offloaded_content.OffloadManager.__init__",
                _tracking_init,
            ),
        ):
            chunks = [
                chunk async for chunk in summarizer.generate_response("test query")
            ]

        assert len(created_managers) == 1, (
            f"Expected 1 OffloadManager created (streaming={streaming}), "
            f"got {len(created_managers)}"
        )
        assert (
            len(cleanup_calls) == 1
        ), f"Expected cleanup called once (streaming={streaming}), got {len(cleanup_calls)}"
        assert any(
            c.type == StreamChunkType.END for c in chunks
        ), f"Expected END chunk (streaming={streaming})"


def test_classifier_reasoning_configuration_is_stripped() -> None:
    """The classifier model must not carry the main request's reasoning config."""
    from types import SimpleNamespace

    from ols.src.query_helpers.docs_summarizer import _disable_classifier_reasoning

    direct_model = SimpleNamespace(thinking={"type": "enabled"})
    vertex_model = SimpleNamespace(model_kwargs={"thinking": {"type": "enabled"}})
    bedrock_model = SimpleNamespace(
        additional_model_request_fields={"thinking": {"type": "enabled"}}
    )
    untouched = SimpleNamespace(temperature=0.7)

    _disable_classifier_reasoning(direct_model)
    _disable_classifier_reasoning(vertex_model)
    _disable_classifier_reasoning(bedrock_model)
    _disable_classifier_reasoning(untouched)

    assert direct_model.thinking is None
    assert vertex_model.model_kwargs == {}
    assert bedrock_model.additional_model_request_fields == {}
    assert untouched.temperature == 0.7
