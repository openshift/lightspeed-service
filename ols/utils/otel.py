"""OpenTelemetry tracing setup for audit spans."""

import atexit
import base64
import json
import logging
import sys
import threading
from typing import Any, Optional, Sequence

from google.protobuf.json_format import MessageToDict
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ReadableSpan,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

logger = logging.getLogger(__name__)

_TRACER_NAME = "ols.audit"


def _b64_to_hex(val: str) -> str:
    """Decode a base64-encoded protobuf bytes field to lowercase hex."""
    return base64.b64decode(val).hex()


_ID_KEYS = ("traceId", "spanId", "parentSpanId")


def _hex_ids(obj: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Convert base64-encoded ID fields to hex in-place."""
    for k in keys:
        if k in obj:
            obj[k] = _b64_to_hex(obj[k])


def _proto_to_otlp_json(d: dict[str, Any]) -> dict[str, Any]:
    """Fix ProtoJSON → OTLP JSON: hex traceId/spanId, already int enums."""
    for rs in d.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                _hex_ids(span, _ID_KEYS)
                for link in span.get("links", []):
                    _hex_ids(link, ("traceId", "spanId"))
    return d


_stdout_lock = threading.Lock()


class OTLPJsonStdoutExporter(SpanExporter):
    """Write OTLP-encoded spans as single-line JSON to stdout."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Encode spans to OTLP proto, convert to dict, and write as single-line JSON."""
        try:
            pb = encode_spans(spans)
            d = MessageToDict(pb, use_integers_for_enums=True)
            line = json.dumps(_proto_to_otlp_json(d))
            with _stdout_lock:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
        except Exception:
            logger.warning("Failed to export audit spans to stdout", exc_info=True)
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS


def init_tracer(
    otel_endpoint: Optional[str] = None,
    insecure: bool = False,
    certificate_file: Optional[str] = None,
    audit_enabled: bool = False,
) -> trace.Tracer:
    """Initialize the OTEL tracer with exporters based on configuration."""
    resource = Resource.create({"service.name": "lightspeed-service"})
    provider = TracerProvider(resource=resource)

    if audit_enabled:
        provider.add_span_processor(SimpleSpanProcessor(OTLPJsonStdoutExporter()))
        logger.info("OTEL stdout JSON exporter enabled for audit compliance")

    if otel_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # pylint: disable=import-outside-toplevel
            OTLPSpanExporter,
        )

        credentials = None
        if not insecure and certificate_file:
            import grpc  # pylint: disable=C0415

            with open(certificate_file, "rb") as f:
                credentials = grpc.ssl_channel_credentials(root_certificates=f.read())

        exporter = OTLPSpanExporter(
            endpoint=otel_endpoint, insecure=insecure, credentials=credentials
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("OTEL tracer configured with endpoint: %s", otel_endpoint)
    elif not audit_enabled:
        logger.info("OTEL tracer configured with no-op exporter (no endpoint)")

    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)
    return trace.get_tracer(_TRACER_NAME)
