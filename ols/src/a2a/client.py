"""A2A client: discover remote agent skills, wrap them as LangChain tools, and AgentsRAG."""

import json
import logging
import re
from collections.abc import Callable
from typing import Any, Optional
from uuid import uuid4

import httpx
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.client.middleware import ClientCallContext, ClientCallInterceptor
from a2a.types import (
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TextPart,
)
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

from ols import config
from ols.src.rag.hybrid_rag import HybridRAGBase
from ols.utils.mcp_utils import ClientHeaders, resolve_server_headers

logger = logging.getLogger(__name__)


class A2AToolInput(BaseModel):
    """Input schema for A2A tool invocations."""

    query: str = Field(description="The text query to send to the remote agent.")


class _HeaderInterceptor(ClientCallInterceptor):
    """Inject authorization headers into every outbound A2A RPC call."""

    def __init__(self, headers: dict[str, str]) -> None:
        self._headers = headers

    async def intercept(
        self,
        method_name: str,
        request_payload: dict[str, Any],
        http_kwargs: dict[str, Any],
        agent_card: AgentCard | None,
        context: ClientCallContext | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Add stored headers to the HTTP request kwargs."""
        existing = http_kwargs.get("headers", {})
        existing.update(self._headers)
        http_kwargs["headers"] = existing
        return request_payload, http_kwargs


def _sanitize_tool_name(name: str) -> str:
    """Convert a string into a valid LangChain tool name (alphanumeric + underscores)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


async def _fetch_agent_card(
    base_url: str,
    headers: dict[str, str],
    request_timeout: int,
) -> AgentCard:
    """Fetch an A2A agent card from its well-known endpoint."""
    async with httpx.AsyncClient(
        headers=headers, timeout=httpx.Timeout(request_timeout)
    ) as http:
        resolver = A2ACardResolver(http, base_url)
        return await resolver.get_agent_card()


def _extract_task_text(task: Task) -> str:
    """Extract text content from a completed A2A Task.

    Prefers artifact text; falls back to the status message.
    """
    parts: list[str] = []

    if task.artifacts:
        for artifact in task.artifacts:
            parts.extend(
                part.root.text
                for part in artifact.parts
                if isinstance(part.root, TextPart)
            )

    if parts:
        return "\n".join(parts)

    if task.status and task.status.message:
        parts.extend(
            part.root.text
            for part in task.status.message.parts
            if isinstance(part.root, TextPart)
        )

    return "\n".join(parts) if parts else ""


def _extract_message_text(message: Message) -> str:
    """Extract text content from an A2A Message response."""
    return "\n".join(
        part.root.text for part in message.parts if isinstance(part.root, TextPart)
    )


def _build_agent_tool(
    agent_name: str,
    skill: AgentSkill,
    card: AgentCard,
    headers: dict[str, str],
    timeout: int,
) -> StructuredTool:
    """Wrap a remote A2A agent skill as a LangChain StructuredTool.

    The returned tool sends a text query to the remote A2A agent and
    collects the full (non-streaming) response.
    """
    tool_name = _sanitize_tool_name(f"{agent_name}_{skill.id}")

    async def _invoke(query: str) -> str:
        message = Message(
            message_id=uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(text=query))],
        )

        request_metadata: dict[str, str] = {}
        if skill.id:
            request_metadata["skill_id"] = skill.id

        client_config = ClientConfig(streaming=False)
        interceptor = _HeaderInterceptor(headers)
        factory = ClientFactory(client_config)
        client = factory.create(card, interceptors=[interceptor])

        try:
            async for event in client.send_message(
                message, request_metadata=request_metadata or None
            ):
                if isinstance(event, tuple):
                    task, _ = event
                    if task.status.state == TaskState.failed:
                        error_text = (
                            _extract_task_text(task)
                            or "Remote agent returned an error."
                        )
                        raise RuntimeError(error_text)
                    return _extract_task_text(task)
                if isinstance(event, Message):
                    return _extract_message_text(event)
        finally:
            await client.close()  # type: ignore [attr-defined]

        return ""

    return StructuredTool(
        name=tool_name,
        description=skill.description,
        coroutine=_invoke,
        args_schema=A2AToolInput,
        metadata={"a2a_agent": agent_name, "a2a_skill_id": skill.id},
    )


async def _gather_a2a_tools(
    agents: list[Any],
    user_token: Optional[str] = None,
    client_headers: ClientHeaders | None = None,
    populate_to_rag: bool = False,
) -> tuple[dict[str, Any], list[StructuredTool]]:
    """Discover tools from a list of A2A agents.

    Args:
        agents: Agent config objects to query.
        user_token: Kubernetes bearer token for resolving auth headers.
        client_headers: Client-provided headers keyed by agent name.
        populate_to_rag: Whether to populate discovered tools into AgentsRAG.

    Returns:
        Tuple of (agent-name-to-config dict, list of StructuredTools).
    """
    agents_config: dict[str, Any] = {}
    all_tools: list[StructuredTool] = []

    for agent_cfg in agents:
        try:
            headers = resolve_server_headers(agent_cfg, user_token, client_headers)
            if headers is None:
                continue

            timeout = agent_cfg.timeout or 30
            card = await _fetch_agent_card(agent_cfg.url, headers, timeout)

            for skill in card.skills:
                tool = _build_agent_tool(agent_cfg.name, skill, card, headers, timeout)
                all_tools.append(tool)

            agents_config[agent_cfg.name] = agent_cfg
            logger.info(
                "Discovered %d tools from A2A agent '%s'",
                len(card.skills),
                agent_cfg.name,
            )
        except Exception:
            logger.exception(
                "Failed to discover tools from A2A agent '%s'",
                agent_cfg.name,
            )

    if populate_to_rag and all_tools and config.agents_rag:
        config.agents_rag.populate_agents(all_tools)

    return agents_config, all_tools


async def _populate_agents_rag(
    user_token: Optional[str],
    client_headers: ClientHeaders | None,
) -> None:
    """Populate AgentsRAG with tools from k8s-auth and client-auth A2A agents.

    Args:
        user_token: Optional user authentication token.
        client_headers: Optional client-provided headers keyed by agent name.
    """
    if not config.k8s_a2a_agents_resolved:
        k8s_agents_config, k8s_tools = await _gather_a2a_tools(
            config.a2a_agents.agents,
            user_token=user_token,
            client_headers=None,
            populate_to_rag=True,
        )

        if k8s_tools:
            logger.info(
                "Populated AgentsRAG with %d tools from %d k8s-auth A2A agents",
                len(k8s_tools),
                len(k8s_agents_config),
            )
            config.agents_rag.set_default_agents(list(k8s_agents_config.keys()))

        config.k8s_a2a_agents_resolved = True

    if client_headers:
        client_agents_list = [
            config.a2a_agents_dict[name]
            for name in client_headers
            if name in config.a2a_agents_dict
        ]

        if client_agents_list:
            client_agents_config, client_tools = await _gather_a2a_tools(
                client_agents_list,
                user_token,
                client_headers,
                populate_to_rag=True,
            )

            if client_tools:
                logger.info(
                    "Added %d tools from %d client-auth A2A agents to AgentsRAG",
                    len(client_tools),
                    len(client_agents_config),
                )


async def get_a2a_tools(  # pylint: disable=too-many-return-statements
    query: str,
    user_token: Optional[str] = None,
    client_headers: ClientHeaders | None = None,
) -> list[StructuredTool]:
    """Get A2A agent tools, using AgentsRAG filtering when configured.

    Args:
        query: The user's query for filtering tools.
        user_token: Optional user authentication token.
        client_headers: Optional client-provided headers keyed by agent name.

    Returns:
        List of StructuredTools from A2A agents (filtered if AgentsRAG configured).
    """
    if not config.a2a_agents.agents:
        return []

    if not config.agents_rag:
        agents_config, all_tools = await _gather_a2a_tools(
            config.a2a_agents.agents, user_token, client_headers
        )
        if not agents_config:
            logger.debug("No A2A agents reachable, agent calling is disabled")
            return []
        logger.info("A2A agents provided: %s", list(agents_config.keys()))
        return all_tools

    await _populate_agents_rag(user_token, client_headers)

    try:
        client_agent_names = list(client_headers.keys()) if client_headers else None
        filtered_result = config.agents_rag.retrieve_hybrid(
            query, client_agents=client_agent_names
        )
    except Exception as e:
        logger.error(
            "Failed to filter tools using AgentsRAG: %s, falling back to all tools",
            e,
        )
        _, all_tools = await _gather_a2a_tools(
            config.a2a_agents.agents, user_token, client_headers
        )
        return all_tools

    if filtered_result:
        filtered_tool_names: set[str] = set()
        agent_names: set[str] = set()
        for agent_name, tools_list in filtered_result.items():
            agent_names.add(agent_name)
            for tool in tools_list:
                if "name" in tool:
                    filtered_tool_names.add(tool["name"])

        filtered_agents_list = [
            config.a2a_agents_dict[name]
            for name in agent_names
            if name in config.a2a_agents_dict
        ]

        if not filtered_agents_list:
            logger.warning(
                "No matching agents found in config for filtered tools. "
                "Filtered tools referenced agents: %s",
                agent_names,
            )
            return []

        _, filtered_tools = await _gather_a2a_tools(
            filtered_agents_list, user_token, client_headers
        )
        filtered_tools = [t for t in filtered_tools if t.name in filtered_tool_names]

        logger.info(
            "Filtered to %d tools from %d A2A agents based on query",
            len(filtered_tools),
            len(filtered_agents_list),
        )
        return filtered_tools

    logger.warning("No A2A agent tools matched the query filter")
    return []


# ---------------------------------------------------------------------------
# AgentsRAG
# ---------------------------------------------------------------------------


class AgentsRAG(HybridRAGBase):
    """Hybrid RAG system for A2A agent tool retrieval.

    Mirrors ToolsRAG but indexes tools discovered from remote A2A agents.
    Internally stores the agent name under the Qdrant ``server`` metadata
    key so the base-class dense-search filter works unchanged.
    """

    _COLLECTION = "a2a_agents"

    def __init__(
        self,
        encode_fn: Callable[[str], list[float]],
        alpha: float = 0.8,
        top_k: int = 10,
        threshold: float = 0.01,
    ) -> None:
        """Initialize the AgentsRAG system.

        Args:
            encode_fn: Function that encodes text into an embedding vector.
            alpha: Weight for dense vs sparse (1.0 = full dense, 0.0 = full sparse).
            top_k: Number of tools to retrieve.
            threshold: Minimum similarity threshold for filtering results.
        """
        super().__init__(
            collection=self._COLLECTION,
            encode_fn=encode_fn,
            alpha=alpha,
            top_k=top_k,
            threshold=threshold,
        )
        self.default_allowed_agents: set[str] = set()

    def set_default_agents(self, agents: list[str]) -> None:
        """Set the default agents that are always included in retrieval.

        Args:
            agents: List of agent names to use as defaults.
        """
        self.default_allowed_agents = set(agents)

    def populate_agents(self, tools_list: list[StructuredTool]) -> None:
        """Populate the RAG with tools discovered from A2A agents.

        Args:
            tools_list: LangChain tools returned by ``get_a2a_tools``.
        """
        ids: list[str] = []
        dense_docs: list[str] = []
        vectors: list[list[float]] = []
        metadatas: list[dict[str, str]] = []

        for tool in tools_list:
            tool_dict = self._convert_agent_to_dict(tool)
            text = self._build_text(tool_dict)
            agent_name = tool_dict.get("server", "")

            ids.append(f"{agent_name}::{tool_dict['name']}")
            dense_docs.append(text)
            vectors.append(self._encode(text))
            metadatas.append(
                {
                    "tool_json": json.dumps(tool_dict),
                    "server": agent_name,
                }
            )

        self._index_documents(ids, dense_docs, vectors, metadatas=metadatas)

    def retrieve_hybrid(
        self,
        query: str,
        client_agents: list[str] | None = None,
        k: int | None = None,
        alpha: float | None = None,
        threshold: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve agent tools using hybrid (dense + sparse) search.

        Args:
            query: Search query describing what tools are needed.
            client_agents: Optional list of client-specific agent names to include.
            k: Number of results to return (default from instance).
            alpha: Weight for dense vs sparse (1.0=dense, 0.0=sparse).
            threshold: Minimum similarity score (default from instance).

        Returns:
            Dictionary mapping agent names to lists of matching tool dicts.
        """
        k = k if k is not None else self.top_k
        alpha = alpha if alpha is not None else self.alpha
        threshold = threshold if threshold is not None else self.threshold

        allowed = self.default_allowed_agents
        if client_agents:
            allowed = allowed | set(client_agents)

        q_vec = self._encode(query)

        dense, dense_ids, dense_metas = self._dense_scores(
            q_vec, k, allowed_servers=allowed
        )
        metadata_lookup = {
            tool_id: json.loads(meta["tool_json"])
            for tool_id, meta in zip(dense_ids, dense_metas)
        }

        sparse, sparse_metadata = self._retrieve_sparse_scores(
            query, allowed_agents=allowed
        )
        for tool_id, tool_dict in sparse_metadata.items():
            if tool_id not in metadata_lookup:
                metadata_lookup[tool_id] = tool_dict

        fused = self._fuse_scores(dense, sparse, alpha, k)

        agent_tools: dict[str, list[dict[str, Any]]] = {}
        for name, score in fused.items():
            if score < threshold:
                continue
            if name not in metadata_lookup:
                continue
            tool = metadata_lookup[name]
            agent = tool.pop("server", None)
            if agent:
                agent_tools.setdefault(agent, []).append(tool)

        return agent_tools

    def _convert_agent_to_dict(self, tool: StructuredTool) -> dict[str, Any]:
        """Convert an A2A agent tool to a dict for indexing.

        Maps the ``a2a_agent`` metadata key to ``server`` so the base-class
        Qdrant filter works unchanged.
        """
        schema = getattr(tool, "args_schema", None)
        if schema is not None and not isinstance(schema, dict):
            schema = schema.model_json_schema()
        return {
            "name": tool.name,
            "desc": tool.description or "",
            "params": schema,
            "server": (
                tool.metadata.get("a2a_agent") if hasattr(tool, "metadata") else None
            ),
        }

    def _build_text(self, t: dict[str, Any]) -> str:
        """Build text representation for embedding: name + description."""
        return f"{t['name']} {t['desc']}"

    def _retrieve_sparse_scores(
        self, query: str, allowed_agents: set[str] | None = None
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """Retrieve BM25 scores with optional agent filtering.

        Args:
            query: The query string.
            allowed_agents: Optional set of agent names to filter by.

        Returns:
            Tuple of (scores dict, metadata dict).
        """
        base_scores, meta_by_id = self._sparse_scores(query)
        if not base_scores:
            return {}, {}

        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for name, score in base_scores.items():
            raw_meta = meta_by_id.get(name)
            if raw_meta is None:
                continue
            tool_dict = json.loads(raw_meta["tool_json"])
            if allowed_agents and tool_dict.get("server", "") not in allowed_agents:
                continue
            scores[name] = score
            metadata[name] = tool_dict

        return scores, metadata
