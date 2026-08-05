"""Unit tests for ols.utils.otel module."""

from unittest.mock import patch

import pytest
from opentelemetry import trace

from ols.utils.otel import init_tracer


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Reset the global tracer provider after each test."""
    yield
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()
    trace._TRACER_PROVIDER = None


class TestInitTracer:
    """Tests for init_tracer function."""

    @patch("ols.utils.otel.BatchSpanProcessor")
    def test_no_endpoint_creates_noop(self, mock_processor):
        """No endpoint and no audit means no exporter is created."""
        tracer = init_tracer()
        assert tracer is not None
        mock_processor.assert_not_called()

    @patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter")
    @patch("ols.utils.otel.BatchSpanProcessor")
    def test_endpoint_creates_otlp_exporter(self, mock_processor, mock_exporter):
        """Providing an endpoint creates an OTLP gRPC exporter."""
        init_tracer(otel_endpoint="localhost:4317")
        mock_exporter.assert_called_once_with(endpoint="localhost:4317")
        mock_processor.assert_called_once()

    @patch("ols.utils.otel.SimpleSpanProcessor")
    def test_audit_enabled_adds_stdout_exporter(self, mock_simple_processor):
        """audit_enabled=True adds OTLPJsonStdoutExporter via SimpleSpanProcessor."""
        init_tracer(audit_enabled=True)
        mock_simple_processor.assert_called_once()
        exporter = mock_simple_processor.call_args[0][0]
        from ols.utils.otel import OTLPJsonStdoutExporter

        assert isinstance(exporter, OTLPJsonStdoutExporter)

    @patch("ols.utils.otel.SimpleSpanProcessor")
    def test_audit_disabled_no_stdout_exporter(self, mock_simple_processor):
        """audit_enabled=False does not add stdout exporter."""
        init_tracer(audit_enabled=False)
        mock_simple_processor.assert_not_called()

    @patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter")
    @patch("ols.utils.otel.BatchSpanProcessor")
    @patch("ols.utils.otel.SimpleSpanProcessor")
    def test_audit_and_endpoint_adds_both(
        self, mock_simple_processor, mock_batch_processor, mock_exporter
    ):
        """Both audit and endpoint add stdout + OTLP exporters."""
        init_tracer(otel_endpoint="localhost:4317", audit_enabled=True)
        mock_simple_processor.assert_called_once()
        mock_batch_processor.assert_called_once()
        mock_exporter.assert_called_once()
