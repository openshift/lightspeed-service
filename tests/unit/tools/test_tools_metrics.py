"""Unit tests for gen_ai tool duration metrics."""

import pytest

from ols import config

config.ols_config.authentication_config.module = "k8s"

from ols.app.metrics.metrics import gen_ai_execute_tool_duration_seconds  # noqa: E402
from ols.src.tools.tools import execute_tool_calls_stream  # noqa: E402


@pytest.mark.asyncio
async def test_tool_execution_observes_duration_histogram():
    """Tool execution records gen_ai_execute_tool_duration_seconds histogram."""
    from tests.unit.tools.test_tools import FakeTool

    tool = FakeTool("test_metric_tool", metadata={"mcp_server": ""})
    tool_calls = [("call_1", {}, tool)]

    labeled_metric = gen_ai_execute_tool_duration_seconds.labels(
        gen_ai_tool_name="test_metric_tool",
    )
    before_samples = [s for s in labeled_metric._samples() if s.name == "_count"]
    before = before_samples[0].value if before_samples else 0.0

    async for _ in execute_tool_calls_stream(tool_calls, 100_000):
        pass

    after_samples = [s for s in labeled_metric._samples() if s.name == "_count"]
    after = after_samples[0].value if after_samples else 0.0

    assert after - before == 1
