"""Configuration for unit tests."""

from typing import Sequence

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from ols import config
from ols.utils.audit_logger import AuditContext, AuditLogger


class CollectingExporter(SpanExporter):
    """Span exporter that collects finished spans in a list."""

    def __init__(self):
        """Initialize the collecting exporter."""
        self.spans = []

    def export(self, spans: Sequence) -> SpanExportResult:
        """Export spans by appending to the internal list."""
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        """Shut down the exporter."""


@pytest.fixture()
def otel_setup():
    """Set up an OTel provider with a collecting exporter."""
    exporter = CollectingExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("ols.audit.test")
    yield exporter, tracer
    provider.shutdown()


def make_audit_ctx(otel_setup, conversation_id="conv-test", user_id="user-test"):
    """Create an AuditContext wired to the test OTel provider."""
    _, tracer = otel_setup
    return AuditContext(
        conversation_id=conversation_id,
        user_id=user_id,
        logger=AuditLogger(enabled=True),
        tracer=tracer,
    )


@pytest.fixture(scope="function", autouse=True)
def ensure_empty_config_for_each_unit_test_by_default():
    """Set up fixture for all unit tests."""
    config.reload_empty()
