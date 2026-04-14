"""Fake OCP tools for agent loop integration testing.

Provides six StructuredTool instances that simulate real OpenShift operations
without requiring an OCP cluster. Each tool carries a ``policy`` annotation in
its metadata so the approval-gate tests can verify ALLOW / DENY / CONFIRM
classification without parsing descriptions or hard-coding names.

Tools
-----
get_cluster_info   - safe read, always allowed
list_pods          - safe read, always allowed
describe_pod       - safe read, returns oversized output to trigger compaction
drain_node         - destructive, needs human confirmation
scale_deployment   - mutating, needs human confirmation
delete_namespace   - dangerous, always denied
"""

from typing import Any

from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Policy constants - used in tool metadata and by test assertions
# ---------------------------------------------------------------------------
POLICY_ALLOW = "allow"
POLICY_CONFIRM = "confirm"
POLICY_DENY = "deny"


# ---------------------------------------------------------------------------
# Arg schemas (pydantic v2)
# ---------------------------------------------------------------------------
class _Empty(BaseModel):
    """Schema for tools that take no arguments."""


class _NamespaceArgs(BaseModel):
    namespace: str = Field(default="default", description="Kubernetes namespace")


class _NodeArgs(BaseModel):
    node_name: str = Field(description="Name of the node to drain")


class _ScaleArgs(BaseModel):
    name: str = Field(description="Deployment name")
    replicas: int = Field(description="Target replica count")


class _PodArgs(BaseModel):
    name: str = Field(description="Pod name")
    namespace: str = Field(default="default", description="Kubernetes namespace")


# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------
CLUSTER_INFO_RESPONSE = (
    '{"version": "4.15.2", "nodes": 6, "status": "healthy", '
    '"platform": "AWS", "channel": "stable-4.15"}'
)

POD_LIST_RESPONSE = (
    "NAME                      READY   STATUS             RESTARTS   AGE\n"
    "nginx-abc123              1/1     Running            0          2d\n"
    "postgres-def456           0/1     CrashLoopBackOff   47         5h\n"
    "redis-ghi789              1/1     Running            0          12h\n"
)

DRAIN_RESPONSE_TPL = '{{"drained": "{node_name}", "pods_evicted": 12, "status": "completed"}}'

SCALE_RESPONSE_TPL = '{{"scaled": "{name}", "replicas": {replicas}, "previous_replicas": 1}}'

_DESCRIBE_POD_YAML_LINE = "  container: nginx\n  image: nginx:1.25\n  ports:\n    - 80/TCP\n"
DESCRIBE_POD_RESPONSE = (
    "apiVersion: v1\nkind: Pod\nmetadata:\n  name: {name}\n"
    + _DESCRIBE_POD_YAML_LINE * 800
)


# ---------------------------------------------------------------------------
# Async coroutines — each returns (text, artifact_dict)
# ---------------------------------------------------------------------------
async def _get_cluster_info(**_kwargs: Any) -> tuple[str, dict]:
    return CLUSTER_INFO_RESPONSE, {}


async def _list_pods(**kwargs: Any) -> tuple[str, dict]:
    ns = kwargs.get("namespace", "default")
    header = f"# Pods in namespace: {ns}\n"
    return header + POD_LIST_RESPONSE, {
        "structured_content": {
            "pods": [
                {"name": "nginx-abc123", "status": "Running", "restarts": 0},
                {"name": "postgres-def456", "status": "CrashLoopBackOff", "restarts": 47},
                {"name": "redis-ghi789", "status": "Running", "restarts": 0},
            ]
        }
    }


async def _drain_node(**kwargs: Any) -> tuple[str, dict]:
    node_name = kwargs.get("node_name", "unknown")
    return DRAIN_RESPONSE_TPL.format(node_name=node_name), {}


async def _delete_namespace(**_kwargs: Any) -> tuple[str, dict]:
    raise RuntimeError(
        "delete_namespace should never execute — policy is DENY"
    )


async def _scale_deployment(**kwargs: Any) -> tuple[str, dict]:
    name = kwargs.get("name", "unknown")
    replicas = kwargs.get("replicas", 1)
    return SCALE_RESPONSE_TPL.format(name=name, replicas=replicas), {}


async def _describe_pod(**kwargs: Any) -> tuple[str, dict]:
    name = kwargs.get("name", "unknown")
    return DESCRIBE_POD_RESPONSE.format(name=name), {}


# ---------------------------------------------------------------------------
# StructuredTool definitions
# ---------------------------------------------------------------------------
def _build_tool(
    name: str,
    description: str,
    coroutine: Any,
    args_schema: type[BaseModel],
    policy: str,
) -> StructuredTool:
    """Build a StructuredTool with policy metadata."""
    return StructuredTool(
        name=name,
        description=description,
        func=lambda **kw: None,
        coroutine=coroutine,
        response_format="content_and_artifact",
        args_schema=args_schema,
        metadata={"policy": policy, "mcp_server": "fake-ocp"},
    )


get_cluster_info = _build_tool(
    name="get_cluster_info",
    description="Return OpenShift cluster version, node count, and health status.",
    coroutine=_get_cluster_info,
    args_schema=_Empty,
    policy=POLICY_ALLOW,
)

list_pods = _build_tool(
    name="list_pods",
    description="List pods in a namespace with status and restart counts.",
    coroutine=_list_pods,
    args_schema=_NamespaceArgs,
    policy=POLICY_ALLOW,
)

describe_pod = _build_tool(
    name="describe_pod",
    description="Return the full YAML manifest for a pod (may be very large).",
    coroutine=_describe_pod,
    args_schema=_PodArgs,
    policy=POLICY_ALLOW,
)

drain_node = _build_tool(
    name="drain_node",
    description="Cordon and drain a node, evicting all pods. Destructive operation.",
    coroutine=_drain_node,
    args_schema=_NodeArgs,
    policy=POLICY_CONFIRM,
)

scale_deployment = _build_tool(
    name="scale_deployment",
    description="Scale a deployment to the specified replica count.",
    coroutine=_scale_deployment,
    args_schema=_ScaleArgs,
    policy=POLICY_CONFIRM,
)

delete_namespace = _build_tool(
    name="delete_namespace",
    description="Delete a Kubernetes namespace and all its resources. DANGEROUS.",
    coroutine=_delete_namespace,
    args_schema=_NamespaceArgs,
    policy=POLICY_DENY,
)


# ---------------------------------------------------------------------------
# Convenience collections
# ---------------------------------------------------------------------------
ALL_FAKE_TOOLS: list[StructuredTool] = [
    get_cluster_info,
    list_pods,
    describe_pod,
    drain_node,
    scale_deployment,
    delete_namespace,
]

SAFE_TOOLS: list[StructuredTool] = [
    get_cluster_info,
    list_pods,
    describe_pod,
]

DESTRUCTIVE_TOOLS: list[StructuredTool] = [
    drain_node,
    scale_deployment,
    delete_namespace,
]

POLICY_MAP: dict[str, str] = {
    t.name: t.metadata["policy"] for t in ALL_FAKE_TOOLS if t.metadata is not None
}
