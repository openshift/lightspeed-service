"""Tests for the fake OCP tool definitions.

Validates that every fake tool is a well-formed StructuredTool, returns
the expected output shape, and carries the correct policy annotation.
"""

import pytest

from tests.harness.fake_tools import (
    ALL_FAKE_TOOLS,
    DESTRUCTIVE_TOOLS,
    POLICY_ALLOW,
    POLICY_CONFIRM,
    POLICY_DENY,
    POLICY_MAP,
    SAFE_TOOLS,
    delete_namespace,
    describe_pod,
    drain_node,
    get_cluster_info,
    list_pods,
    scale_deployment,
)


class TestToolDefinitions:
    """Verify structural properties of the fake tool set."""

    @staticmethod
    def test_all_tools_count():
        """Six tools should be registered."""
        assert len(ALL_FAKE_TOOLS) == 6

    @staticmethod
    def test_safe_tools_are_allow_policy():
        """Every safe tool must carry the ALLOW policy."""
        for tool in SAFE_TOOLS:
            assert tool.metadata["policy"] == POLICY_ALLOW, tool.name

    @staticmethod
    def test_destructive_tools_are_not_allow():
        """No destructive tool should have the ALLOW policy."""
        for tool in DESTRUCTIVE_TOOLS:
            assert tool.metadata["policy"] != POLICY_ALLOW, tool.name

    @staticmethod
    def test_policy_map_covers_all_tools():
        """The POLICY_MAP must have an entry for every tool."""
        tool_names = {t.name for t in ALL_FAKE_TOOLS}
        assert set(POLICY_MAP.keys()) == tool_names

    @staticmethod
    def test_all_tools_have_coroutine():
        """Every tool must define an async coroutine."""
        for tool in ALL_FAKE_TOOLS:
            assert tool.coroutine is not None, f"{tool.name} missing coroutine"

    @staticmethod
    def test_all_tools_have_description():
        """Every tool must have a non-empty description."""
        for tool in ALL_FAKE_TOOLS:
            assert tool.description, f"{tool.name} missing description"

    @staticmethod
    def test_unique_tool_names():
        """Tool names must be unique."""
        names = [t.name for t in ALL_FAKE_TOOLS]
        assert len(names) == len(set(names))


class TestToolExecution:
    """Verify the coroutines return well-formed (text, artifact) tuples."""

    @staticmethod
    @pytest.mark.asyncio
    async def test_get_cluster_info_returns_json():
        """get_cluster_info should return parseable JSON text."""
        text, artifact = await get_cluster_info.coroutine()
        assert '"version"' in text
        assert '"4.15.2"' in text
        assert isinstance(artifact, dict)

    @staticmethod
    @pytest.mark.asyncio
    async def test_list_pods_returns_structured_content():
        """list_pods should return structured content in the artifact."""
        text, artifact = await list_pods.coroutine(namespace="kube-system")
        assert "kube-system" in text
        assert "structured_content" in artifact
        pods = artifact["structured_content"]["pods"]
        assert len(pods) == 3

    @staticmethod
    @pytest.mark.asyncio
    async def test_drain_node_uses_node_name():
        """drain_node should include the node name in its response."""
        text, _artifact = await drain_node.coroutine(node_name="worker-3")
        assert "worker-3" in text
        assert "pods_evicted" in text

    @staticmethod
    @pytest.mark.asyncio
    async def test_scale_deployment_uses_args():
        """scale_deployment should reflect the requested name and replicas."""
        text, _ = await scale_deployment.coroutine(name="nginx", replicas=10)
        assert "nginx" in text
        assert "10" in text

    @staticmethod
    @pytest.mark.asyncio
    async def test_describe_pod_produces_large_output():
        """describe_pod should return output large enough to trigger compaction."""
        text, _ = await describe_pod.coroutine(name="nginx-abc123")
        assert len(text) > 40_000, "Output should be >40KB to trigger compaction"
        assert "nginx-abc123" in text

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_namespace_raises():
        """delete_namespace should raise — it must never actually execute."""
        with pytest.raises(RuntimeError, match="should never execute"):
            await delete_namespace.coroutine()


class TestPolicyClassification:
    """Verify the policy annotations match the playbook spec."""

    @staticmethod
    def test_get_cluster_info_is_allow():
        """Read-only cluster info should be allowed."""
        assert POLICY_MAP["get_cluster_info"] == POLICY_ALLOW

    @staticmethod
    def test_list_pods_is_allow():
        """Pod listing is a safe read."""
        assert POLICY_MAP["list_pods"] == POLICY_ALLOW

    @staticmethod
    def test_describe_pod_is_allow():
        """Describe is read-only despite large output."""
        assert POLICY_MAP["describe_pod"] == POLICY_ALLOW

    @staticmethod
    def test_drain_node_is_confirm():
        """Draining a node is destructive and requires confirmation."""
        assert POLICY_MAP["drain_node"] == POLICY_CONFIRM

    @staticmethod
    def test_scale_deployment_is_confirm():
        """Scaling is a mutation that requires confirmation."""
        assert POLICY_MAP["scale_deployment"] == POLICY_CONFIRM

    @staticmethod
    def test_delete_namespace_is_deny():
        """Deleting a namespace is always denied."""
        assert POLICY_MAP["delete_namespace"] == POLICY_DENY
