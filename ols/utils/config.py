"""Configuration loader."""

from __future__ import annotations

import logging
import traceback
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

import ols.app.models.config as config_model
from ols.src.cache.cache_factory import CacheFactory
from ols.src.quota.quota_limiter_factory import QuotaLimiterFactory
from ols.src.quota.token_usage_history import TokenUsageHistory

# as the index_loader.py is excluded from type checks, it confuses
# mypy a bit, hence the [attr-defined] bellow
from ols.src.rag_index.index_loader import IndexLoader  # type: ignore [attr-defined]
from ols.src.skills.skills_rag import SkillsRAG, load_skills_from_directory
from ols.src.tools.tools_rag.hybrid_tools_rag import ToolsRAG
from ols.utils.redactor import Redactor

if TYPE_CHECKING:
    from ols.src.a2a.client import AgentsRAG

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from io import TextIOBase

    from ols.src.cache.cache import Cache
    from ols.src.quota.quota_limiter import QuotaLimiter
    from ols.src.tools.approval import PendingApprovalStoreBase


class AppConfig:  # pylint: disable=too-many-public-methods
    """Singleton class to load and store the configuration."""

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "AppConfig":
        """Create a new instance of the class."""
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the class instance."""
        self.config = config_model.Config()
        self._query_filters: Optional[Redactor] = None
        self._rag_index_loader: Optional[IndexLoader] = None
        self._conversation_cache: Optional[Cache] = None
        self._quota_limiters: Optional[list[QuotaLimiter]] = None
        self._token_usage_history: Optional[TokenUsageHistory] = None
        self.k8s_tools_resolved = False
        self.k8s_a2a_agents_resolved = False
        self._tools_approval: Optional[config_model.ToolsApprovalConfig] = None
        self._pending_approval_store: Optional["PendingApprovalStoreBase"] = None

    @property
    def llm_config(self) -> config_model.LLMProviders:
        """Return the LLM providers configuration."""
        return self.config.llm_providers

    @property
    def ols_config(self) -> config_model.OLSConfig:
        """Return the OLS configuration."""
        return self.config.ols_config

    @property
    def mcp_servers(self) -> config_model.MCPServers:
        """Return the MCP servers configuration."""
        return self.config.mcp_servers

    @cached_property
    def mcp_servers_dict(self) -> dict[str, config_model.MCPServerConfig]:
        """Return dictionary mapping MCP server names to their configuration."""
        return {server.name: server for server in self.config.mcp_servers.servers}

    @property
    def a2a_agents(self) -> config_model.A2AAgents:
        """Return the A2A agents configuration."""
        return self.config.a2a_agents

    @cached_property
    def a2a_agents_dict(self) -> dict[str, config_model.A2AAgentConfig]:
        """Return dictionary mapping A2A agent names to their configuration."""
        return {agent.name: agent for agent in self.config.a2a_agents.agents}

    @property
    def dev_config(self) -> config_model.DevConfig:
        """Return the dev configuration."""
        return self.config.dev_config

    @property
    def tools_approval(self) -> config_model.ToolsApprovalConfig:
        """Return the tools approval configuration."""
        if self._tools_approval is None:
            if self.config.ols_config.tools_approval is not None:
                self._tools_approval = self.config.ols_config.tools_approval
            else:
                self._tools_approval = config_model.ToolsApprovalConfig()
        return self._tools_approval

    @property
    def conversation_cache(self) -> Cache:
        """Return the conversation cache."""
        if self._conversation_cache is None:
            self._conversation_cache = CacheFactory.conversation_cache(
                self.ols_config.conversation_cache
            )
        return self._conversation_cache

    @property
    def pending_approval_store(self) -> "PendingApprovalStoreBase":
        """Return the pending approval store for tool approval flow."""
        if self._pending_approval_store is None:
            from ols.src.tools.approval import (  # pylint: disable=import-outside-toplevel
                create_pending_approval_store,
            )

            self._pending_approval_store = create_pending_approval_store()
        return self._pending_approval_store

    @property
    def quota_limiters(self) -> list[QuotaLimiter]:
        """Return all quota limiters."""
        if self._quota_limiters is None:
            self._quota_limiters = QuotaLimiterFactory.quota_limiters(
                self.ols_config.quota_handlers
            )
        return self._quota_limiters

    @property
    def token_usage_history(self) -> Optional[TokenUsageHistory]:
        """Return token usage history object."""
        if (
            self._token_usage_history is None
            and self.ols_config.quota_handlers is not None
            and self.ols_config.quota_handlers.enable_token_history
        ):
            self._token_usage_history = TokenUsageHistory(
                self.ols_config.quota_handlers.storage
            )
        return self._token_usage_history

    @property
    def query_redactor(self) -> Redactor:
        """Return the query redactor."""
        # TODO: OLS-380 Config object mirrors configuration
        if self._query_filters is None:
            self._query_filters = Redactor(self.ols_config.query_filters)
        return self._query_filters

    @property
    def rag_index(self) -> Optional[list[Any]]:
        """Return the RAG index.

        Returns a list of LlamaIndex BaseIndex objects, but we use Any because
        the index_loader module is excluded from type checking.
        """
        # TODO: OLS-380 Config object mirrors configuration
        return self.rag_index_loader.vector_indexes

    @property
    def rag_index_loader(self) -> IndexLoader:
        """Return the RAG index loader."""
        if self._rag_index_loader is None:
            self._rag_index_loader = IndexLoader(self.ols_config.reference_content)
        return self._rag_index_loader

    def _resolve_embed_model(self, embed_model_path: Optional[str] = None) -> Any:
        """Resolve the embedding model for hybrid RAG (tools and skills).

        Uses the RAG index loader's model when available (production/operator path).
        Falls back to creating a HuggingFaceEmbedding from embed_model_path or the
        default sentence-transformers model (local testing only).
        """
        from llama_index.embeddings.huggingface import (  # pylint: disable=import-outside-toplevel
            HuggingFaceEmbedding,
        )

        # Suppress noisy progress bars and model load reports from HuggingFace.
        for name in ("sentence_transformers", "transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)

        # Local testing override -- not exposed by the operator.
        if embed_model_path:
            return HuggingFaceEmbedding(model_name=embed_model_path)

        # Production path -- reuse the model from the RAG index loader.
        embed_model = self.rag_index_loader.embed_model
        if embed_model is not None and not isinstance(embed_model, str):
            return embed_model

        # Fallback when no RAG index is configured.
        fallback_model = "sentence-transformers/all-mpnet-base-v2"
        logger.warning(
            "No embedding model from RAG index or config; "
            "downloading '%s' from HuggingFace Hub",
            fallback_model,
        )
        return HuggingFaceEmbedding(model_name=fallback_model)

    @cached_property
    def tools_rag(self) -> Optional[ToolsRAG]:
        """Return the ToolsRAG instance for tool filtering.

        Only creates the instance if tool_filtering configuration exists in the config
        and there are MCP servers configured.
        """
        if (
            self.config.ols_config.tool_filtering is not None
            and self.config.mcp_servers
            and len(self.config.mcp_servers.servers) > 0
        ):
            tool_config = self.config.ols_config.tool_filtering
            try:
                embed_model = self._resolve_embed_model(tool_config.embed_model_path)
            except Exception:
                logger.exception(
                    "Failed to load embedding model for tool filtering; "
                    "tool filtering disabled"
                )
                return None
            return ToolsRAG(
                encode_fn=embed_model.get_text_embedding,
                alpha=tool_config.alpha,
                top_k=tool_config.top_k,
                threshold=tool_config.threshold,
            )
        return None

    @cached_property
    def agents_rag(self) -> Optional["AgentsRAG"]:
        """Return the AgentsRAG instance for A2A agent filtering.

        Only creates the instance if agent_filtering configuration exists
        and there are A2A agents configured.
        """
        if (
            self.config.ols_config.tool_filtering is not None
            and self.config.a2a_agents
            and len(self.config.a2a_agents.agents) > 0
        ):
            from ols.src.a2a.client import (  # pylint: disable=import-outside-toplevel
                AgentsRAG,
            )

            agent_config = self.config.ols_config.tool_filtering
            try:
                embed_model = self._resolve_embed_model(agent_config.embed_model_path)
            except Exception:
                logger.exception(
                    "Failed to load embedding model for A2A agent filtering; "
                    "agent filtering disabled"
                )
                return None
            return AgentsRAG(
                encode_fn=embed_model.get_text_embedding,
                alpha=agent_config.alpha,
                top_k=agent_config.top_k,
                threshold=agent_config.threshold,
            )
        return None

    @cached_property
    def skills_rag(self) -> Optional[SkillsRAG]:
        """Return the SkillsRAG instance for skill selection.

        Only creates the instance if skills configuration exists. Loads skills
        from the configured directory and populates the index eagerly.
        """
        skills_config = self.config.ols_config.skills
        if skills_config is None:
            return None

        skills_dir = Path(skills_config.skills_dir)
        if not skills_dir.is_dir():
            logger.warning("Skills directory does not exist: %s", skills_dir)
            return None

        skills = load_skills_from_directory(skills_dir)
        if not skills:
            logger.warning("No skills found in %s", skills_dir)
            return None

        try:
            embed_model = self._resolve_embed_model(skills_config.embed_model_path)
        except Exception:
            logger.exception(
                "Failed to load embedding model for skills; skills disabled"
            )
            return None

        rag = SkillsRAG(
            encode_fn=embed_model.get_text_embedding,
            alpha=skills_config.alpha,
            threshold=skills_config.threshold,
        )
        rag.populate_skills(skills)

        return rag

    @property
    def proxy_config(self) -> Optional[config_model.ProxyConfig]:
        """Return the proxy configuration."""
        return self.config.proxy_config  # type: ignore[attr-defined]

    def reload_empty(self) -> None:
        """Reload the configuration with empty values."""
        self.config = config_model.Config()

    @staticmethod
    def _load_config_from_yaml_stream(
        stream: TextIOBase,
        ignore_llm_secrets: bool = False,
        ignore_missing_certs: bool = False,
    ) -> config_model.Config:
        """Load configuration from a YAML stream."""
        data = yaml.safe_load(stream)
        config = config_model.Config(data, ignore_llm_secrets, ignore_missing_certs)
        config.validate_yaml()
        return config

    def reload_from_yaml_file(
        self,
        config_file: str,
        ignore_llm_secrets: bool = False,
        ignore_missing_certs: bool = False,
    ) -> None:
        """Reload the configuration from the YAML file."""
        try:
            with open(config_file, encoding="utf-8") as f:
                self.config = self._load_config_from_yaml_stream(
                    f, ignore_llm_secrets, ignore_missing_certs
                )
            # reset the query filters and rag index to not use cached
            # values
            self._query_filters = None
            self._rag_index_loader = None
            self._tools_approval = None
            self._pending_approval_store = None
            # Clear cached_property if it exists
            if "mcp_servers_dict" in self.__dict__:
                del self.__dict__["mcp_servers_dict"]
            if "a2a_agents_dict" in self.__dict__:
                del self.__dict__["a2a_agents_dict"]
            if "tools_rag" in self.__dict__:
                del self.__dict__["tools_rag"]
            if "agents_rag" in self.__dict__:
                del self.__dict__["agents_rag"]
            if "skills_rag" in self.__dict__:
                del self.__dict__["skills_rag"]
        except Exception as e:
            print(f"Failed to load config file {config_file}: {e!s}")
            print(traceback.format_exc())
            raise


config: AppConfig = AppConfig()
