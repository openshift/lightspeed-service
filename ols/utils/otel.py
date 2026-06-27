"""OpenTelemetry tracing setup for audit spans."""

import atexit
import contextvars
import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator

logger = logging.getLogger(__name__)

_TRACER_NAME = "ols.audit"

_trace_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "_trace_id_var", default=None
)


class _ConversationIdGenerator(IdGenerator):
    """Id generator that uses a context-local trace_id override when set."""

    def __init__(self) -> None:
        self._random = RandomIdGenerator()

    def generate_span_id(self) -> int:
        return self._random.generate_span_id()

    def generate_trace_id(self) -> int:
        override = _trace_id_var.get()
        if override is not None:
            return override
        return self._random.generate_trace_id()


_id_generator = _ConversationIdGenerator()


def init_tracer(
    otel_endpoint: Optional[str] = None, insecure: bool = False
) -> trace.Tracer:
    """Initialize the OTEL tracer with OTLP exporter or no-op."""
    resource = Resource.create({"service.name": "lightspeed-service"})
    provider = TracerProvider(resource=resource, id_generator=_id_generator)
    if otel_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # pylint: disable=C0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import (  # pylint: disable=C0415
            BatchSpanProcessor,
        )

        exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=insecure)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("OTEL tracer configured with endpoint: %s", otel_endpoint)
    else:
        logger.info("OTEL tracer configured with no-op exporter (no endpoint)")

    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)
    return trace.get_tracer(_TRACER_NAME)


def set_conversation_trace_id(trace_id_hex: str) -> None:
    """Set the trace_id override for the current context."""
    _trace_id_var.set(int(trace_id_hex, 16))


def clear_conversation_trace_id() -> None:
    """Clear the trace_id override for the current context."""
    _trace_id_var.set(None)
