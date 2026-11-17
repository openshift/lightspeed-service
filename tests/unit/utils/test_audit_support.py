"""Exercise metadata-only audit lifecycles and native transports without external providers."""

# ruff: noqa: S106

import asyncio
import importlib
import json
import logging
import queue
import socket
import ssl
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpcore
import httpx
import httpx2
import pytest
from fastapi import HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI

from ols import config, constants
from ols.app.models.config import (
    AuditLoggingMode,
    LLMProviders,
    MCPServerConfig,
    MCPServers,
)
from ols.app.models.models import MCPAppResourceRequest, MCPAppToolCallRequest
from ols.src.llms.llm_loader import load_llm
from ols.utils import audit_support as support
from ols.utils.mcp_utils import gather_mcp_tools


@pytest.fixture()
def audit_records(caplog, monkeypatch):
    """Capture public log lines and private lifecycle state independently."""
    logger = logging.getLogger("ols.connection_audit")
    monkeypatch.setattr(logger, "propagate", False)
    logger.addHandler(caplog.handler)
    caplog._audit_internal_records = []
    original_send = support.RequestAttempt._send

    def capture_internal(attempt, event, **fields):
        internal = {
            key: value
            for key, value in {
                **attempt._record,
                **fields,
                "event": event,
                "timestamp": datetime.now().astimezone().isoformat(),
            }.items()
            if value is not None
        }
        previous_count = len(wire_records(caplog))
        original_send(attempt, event, **fields)
        if len(wire_records(caplog)) > previous_count:
            caplog._audit_internal_records.append(internal)

    monkeypatch.setattr(support.RequestAttempt, "_send", capture_internal)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            yield caplog
    finally:
        for attempt in tuple(support._attempts.values()):
            attempt.finish(response_complete=False, cancelled=True)
        logger.removeHandler(caplog.handler)


def records(captured):
    """Return private attempt state for lifecycle-focused unit assertions."""
    count = len(wire_records(captured))
    return captured._audit_internal_records[-count:] if count else []


def wire_records(captured):
    """Decode public records emitted to the dedicated audit logger."""
    return [
        json.loads(item.getMessage())
        for item in captured.records
        if item.name == "ols.connection_audit"
    ]


@contextmanager
def dispatch():
    """Activate a native model dispatch."""
    activate = support.llm_call_scope()
    with activate():
        yield


def test_started_record_uses_shared_public_schema(audit_records):
    """Keep internal retry metadata out of the shared outbound record contract."""
    with dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443),
            method="POST",
        )
    attempt.response_headers(200)
    attempt.finish(response_complete=True)

    started, finished = wire_records(audit_records)
    assert started["event"] == "outbound.request.started"
    assert finished["event"] == "outbound.request.finished"
    assert started["schema_version"] == 1
    assert started["service.name"] == "lightspeed-service"
    assert started["request.kind"] == "llm"
    assert started["server.address"] == "model.invalid"
    assert started["server.port"] == 443
    assert started["network.transport"] == "tcp"
    assert started["http.request.method"] == "POST"
    assert started["attempt_id"] == finished["attempt_id"]
    assert UUID(started["attempt_id"])
    assert set(started) == {
        "schema_version",
        "event",
        "timestamp",
        "service.name",
        "attempt_id",
        "request.kind",
        "server.address",
        "server.port",
        "network.transport",
        "http.request.method",
    }
    assert set(finished) == {
        *set(started),
        "outcome",
        "http.response.status_code",
        "duration_ms",
    }
    assert finished["outcome"] == "success"
    assert finished["http.response.status_code"] == 200
    assert finished["duration_ms"] >= 0
    assert not {
        "logical_call_id",
        "operation",
        "provider",
        "model",
        "destination_host",
        "destination_port",
        "request_body_bytes",
        "response_body_bytes",
        "response_complete",
        "failure_category",
    }.intersection(started | finished)


@pytest.mark.parametrize(
    ("status", "error", "outcome", "error_type"),
    [
        (403, None, "http_error", "http_error"),
        (None, socket.gaierror("secret"), "transport_error", "dns_error"),
        (None, asyncio.CancelledError("secret"), "cancelled", "cancelled"),
    ],
)
def test_public_record_maps_request_kind_and_safe_outcome(
    audit_records, status, error, outcome, error_type
):
    """Map credential requests and private transport results into public values."""
    with support.auth_request("google"):
        attempt = support.start_attempt(
            support.Destination("https", "auth.invalid", 443), method="POST"
        )
    if status is not None:
        attempt.response_headers(status)
    attempt.finish(response_complete=status is not None, error=error)

    started, finished = wire_records(audit_records)
    assert started["request.kind"] == "credential"
    assert finished["request.kind"] == "credential"
    assert finished["outcome"] == outcome
    assert finished["error.type"] == error_type


@pytest.mark.parametrize(
    ("conversation_id", "expected_conversation_id"),
    [
        (
            "12345678-1234-4234-8234-123456789abc",
            "12345678-1234-4234-8234-123456789abc",
        ),
        ("not-a-uuid", None),
    ],
)
def test_public_record_carries_only_validated_conversation_id(
    audit_records, otel_setup, conversation_id, expected_conversation_id
):
    """Keep valid conversation correlation but never copy user identity."""
    _, tracer = otel_setup
    with tracer.start_as_current_span(
        "request",
        attributes={
            "gen_ai.conversation.id": conversation_id,
            "user_id": "private-user-id",
        },
    ) as span:
        with dispatch():
            attempt = support.start_attempt(
                support.Destination("https", "model.invalid", 443), method="POST"
            )
        attempt.finish(response_complete=False, error=socket.gaierror("secret"))

    started, finished = wire_records(audit_records)
    if expected_conversation_id:
        assert started["gen_ai.conversation.id"] == expected_conversation_id
        assert finished["gen_ai.conversation.id"] == expected_conversation_id
    else:
        assert "gen_ai.conversation.id" not in started | finished
    context = span.get_span_context()
    expected_trace_id = format(context.trace_id, "032x")
    expected_span_id = format(context.span_id, "016x")
    assert started["trace_id"] == finished["trace_id"] == expected_trace_id
    assert started["span_id"] == finished["span_id"] == expected_span_id
    assert "user_id" not in started | finished
    assert "private-user-id" not in json.dumps(started | finished)


def test_started_emitted_before_transport_and_completed_once(audit_records, otel_setup):
    """Catch delayed starts, duplicate finishes and reassigned causal context."""
    _, tracer = otel_setup
    with tracer.start_as_current_span("original") as span, dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
        assert attempt is not None
        assert [r["event"] for r in wire_records(audit_records)] == [
            "outbound.request.started"
        ]
        attempt.response_headers(200)
    with tracer.start_as_current_span("unrelated"):
        attempt.finish(response_complete=True)
        attempt.finish(response_complete=False)
    started, completed = wire_records(audit_records)
    assert completed["schema_version"] == 1
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["outcome"] == "success"
    assert completed["http.response.status_code"] == 200
    assert completed["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert completed["span_id"] == format(span.get_span_context().span_id, "016x")
    assert not support._attempts
    assert all(item.name == "ols.connection_audit" for item in audit_records.records)


@pytest.mark.parametrize(
    ("error", "status", "complete", "category", "outcome"),
    [
        (None, 200, True, None, "success"),
        (None, 429, True, "http_error", "failure"),
        (None, 200, False, "incomplete_response", "failure"),
        (None, None, False, "incomplete_response", "failure"),
        (socket.gaierror("secret"), None, False, "dns", "failure"),
        (ConnectionRefusedError("secret"), None, False, "connect", "failure"),
        (httpx.ConnectError("secret"), None, False, "connect", "failure"),
        (ssl.SSLCertVerificationError("secret"), None, False, "tls", "failure"),
        (TimeoutError("secret"), None, False, "timeout", "failure"),
        (httpx.ReadTimeout("secret"), 200, False, "timeout", "failure"),
        (httpcore.ReadTimeout("secret"), 200, False, "timeout", "failure"),
        (httpcore.ConnectError("secret"), None, False, "connect", "failure"),
        (httpx2.ReadTimeout("secret"), 200, False, "timeout", "failure"),
        (httpx2.ConnectError("secret"), None, False, "connect", "failure"),
        (ValueError("secret"), 200, False, "stream_error", "failure"),
        (
            ValueError("secret tls certificate verification"),
            None,
            False,
            "transport",
            "failure",
        ),
        (asyncio.CancelledError("secret"), 200, False, "cancelled", "cancelled"),
        (GeneratorExit(), 200, False, "incomplete_response", "failure"),
    ],
)
def test_safe_outcomes(audit_records, error, status, complete, category, outcome):
    """Catch leaked exceptions or semantic success assumptions."""
    with dispatch():
        attempt = support.start_attempt(support.Destination("http", "model.invalid", 0))
    if status is not None:
        attempt.response_headers(status)
    attempt.finish(response_complete=complete, error=error)
    completed = records(audit_records)[-1]
    assert completed.get("failure_category") == category
    assert completed["outcome"] == outcome
    assert completed["destination_port"] == 0
    assert "secret" not in json.dumps(completed)
    assert "trace_id" not in completed
    assert "span_id" not in completed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_httpx_request_body_passes_through_unchanged(audit_records, asynchronous):
    """Preserve request payloads for the native sync and async transports."""
    payload = b"private request payload"

    class SyncTransport(httpx.BaseTransport):
        def handle_request(self, request):
            assert b"".join(request.stream) == payload
            return httpx.Response(200, content=b"ok")

    class AsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            chunks = [chunk async for chunk in request.stream]
            assert b"".join(chunks) == payload
            return httpx.Response(200, content=b"ok")

    client = (
        httpx.AsyncClient(transport=AsyncTransport())
        if asynchronous
        else httpx.Client(transport=SyncTransport())
    )
    support.instrument_httpx_client(client)
    try:
        with dispatch():
            if asynchronous:
                await client.post(
                    "https://model.invalid/chat/completions", content=payload
                )
            else:
                client.post("https://model.invalid/chat/completions", content=payload)
        started, completed = records(audit_records)
        assert started["event"] == "llm.request.started"
        assert completed["outcome"] == "success"
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_httpx_unconsumed_request_body_preserves_transport_error(
    audit_records, asynchronous
):
    """Preserve native connection errors when the request body is not consumed."""

    class SyncTransport(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ConnectError("private connection failure")

    class AsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("private connection failure")

    client = (
        httpx.AsyncClient(transport=AsyncTransport())
        if asynchronous
        else httpx.Client(transport=SyncTransport())
    )
    support.instrument_httpx_client(client)
    try:
        with dispatch(), pytest.raises(httpx.ConnectError):
            if asynchronous:
                await client.post(
                    "https://model.invalid/chat/completions",
                    content=b"private request payload",
                )
            else:
                client.post(
                    "https://model.invalid/chat/completions",
                    content=b"private request payload",
                )
        started, completed = records(audit_records)
        assert started["attempt_id"] == completed["attempt_id"]
        assert completed["failure_category"] == "connect"
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


def test_wrapped_certificate_denial_retains_only_controlled_metadata(audit_records):
    """Catch lost TLS-denial evidence when a native transport wraps verification failure."""
    error = httpx.ConnectError("secret connection detail")
    error.__cause__ = ssl.SSLCertVerificationError("secret certificate detail")
    with dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
        attempt.finish(response_complete=False, error=error)
    completed = records(audit_records)[-1]
    assert completed["failure_category"] == "tls"
    assert "secret" not in json.dumps(completed)


def test_native_retry_and_nested_fallback_keep_attempts_distinct(audit_records):
    """Catch collapsed retries or duplicated native fallback scopes."""
    with dispatch():
        with dispatch():
            first = support.start_attempt(
                support.Destination("https", "model.invalid", 443)
            )
        first.finish(response_complete=False, error=TimeoutError())
        second = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
        second.response_headers(200)
        second.finish(response_complete=True)
    data = wire_records(audit_records)
    assert len(data) == 4
    assert data[0]["attempt_id"] == data[1]["attempt_id"]
    assert data[2]["attempt_id"] == data[3]["attempt_id"]
    assert data[0]["attempt_id"] != data[2]["attempt_id"]
    assert data[1]["outcome"] == "transport_error"
    assert data[3]["outcome"] == "success"
    assert (
        support.start_attempt(support.Destination("https", "model.invalid", 443))
        is None
    )


def test_disabled_and_unscoped_requests_emit_nothing(audit_records):
    """Catch unconditional emission and configuration leakage."""
    assert (
        support.start_attempt(support.Destination("https", "model.invalid", 443))
        is None
    )
    config.ols_config.audit.logging = AuditLoggingMode.DISABLED
    with dispatch():
        assert (
            support.start_attempt(support.Destination("https", "model.invalid", 443))
            is None
        )
    assert records(audit_records) == []


@pytest.mark.asyncio
async def test_concurrent_dispatches_keep_attempt_context_isolated(audit_records):
    """Catch process-global scope and task-context contamination."""

    async def invoke(name):
        with dispatch():
            await asyncio.sleep(0)
            attempt = support.start_attempt(
                support.Destination("https", "model.invalid", 443)
            )
            await asyncio.sleep(0)
            attempt.response_headers(200)
            attempt.finish(response_complete=True)

    await asyncio.gather(invoke("first"), invoke("second"))
    data = wire_records(audit_records)
    assert {item["request.kind"] for item in data} == {"llm"}
    assert len({item["attempt_id"] for item in data}) == 2
    assert {item["event"] for item in data} == {
        "outbound.request.started",
        "outbound.request.finished",
    }


def test_emitter_failure_does_not_replace_native_result(monkeypatch):
    """Catch audit I/O changing native request behavior or leaking active entries."""

    def broken_info(*args, **kwargs):
        raise RuntimeError("secret logging failure")

    monkeypatch.setattr(logging.getLogger("ols.connection_audit"), "info", broken_info)
    with dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
        attempt.response_headers(200)
        attempt.finish(response_complete=True)
    assert not support._attempts


def test_started_attempt_completes_if_audit_is_later_disabled(audit_records):
    """Catch losing a matching completion on a configuration change."""
    with dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
    config.ols_config.audit.logging = AuditLoggingMode.DISABLED
    attempt.response_headers(204)
    attempt.finish(response_complete=True)
    completed = records(audit_records)[-1]
    assert (
        not {"run_id", "phase", "user_id", "conversation_id", "headers", "body", "url"}
        & completed.keys()
    )


def test_in_memory_matching_removes_entry_once_across_threads(audit_records):
    """Catch duplicate finishes and active-entry leaks without a telemetry backend."""
    with dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
    identity = records(audit_records)[0]["attempt_id"]
    assert support._attempts[identity] is attempt
    attempt.response_headers(200)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: attempt.finish(response_complete=True), range(32)))
    assert identity not in support._attempts
    started, completed = records(audit_records)
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["outcome"] == "success"


def test_service_logging_emits_json_metadata_without_otel_pipeline(capsys, monkeypatch):
    """Exercise the actual service console handler and an independent field allowlist."""
    from ols.app.models.config import LoggingConfig
    from ols.utils.logging_configurator import configure_logging

    logger = logging.getLogger("ols.connection_audit")
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    configure_logging(LoggingConfig())
    with dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
        attempt._record["headers"] = {"authorization": "secret sentinel"}
        attempt._record["body"] = "secret sentinel"
        attempt.response_headers(200)
        attempt.finish(response_complete=True)
    captured = capsys.readouterr()
    console = captured.out
    emitted = [json.loads(line) for line in console.splitlines()]
    assert not captured.err
    started, completed = emitted
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["outcome"] == "success"
    allowed = {
        "schema_version",
        "event",
        "timestamp",
        "service.name",
        "attempt_id",
        "request.kind",
        "server.address",
        "server.port",
        "network.peer.address",
        "network.transport",
        "http.request.method",
        "trace_id",
        "span_id",
        "gen_ai.conversation.id",
        "gen_ai.tool.name",
        "outcome",
        "error.type",
        "http.response.status_code",
        "duration_ms",
        "request.body.size",
    }
    assert all(set(record) <= allowed for record in emitted)
    assert [record["event"] for record in emitted] == [
        "outbound.request.started",
        "outbound.request.finished",
    ]
    assert "secret" not in console
    assert "resourceLogs" not in console
    assert not support._attempts


@pytest.fixture(autouse=True)
def loopback_only_network(monkeypatch):
    """Refuse public DNS before any SDK can escape a controlled boundary."""
    original = socket.getaddrinfo

    def lookup(host, *args, **kwargs):
        if host not in {
            "localhost",
            "127.0.0.1",
            "::1",
            b"localhost",
            b"127.0.0.1",
            b"::1",
        }:
            raise AssertionError("SDK test attempted non-loopback DNS")
        return original(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", lookup)


class SyncBody(httpx.SyncByteStream):
    """Expose native EOF and close independently."""

    def __init__(self, error=None):
        """Configure a controlled native stream failure."""
        self.error = error
        self.closes = 0

    def __iter__(self):
        """Yield bytes without reading ahead."""
        yield b"one"
        if self.error:
            raise self.error
        yield b"two"

    def close(self):
        """Count native close calls."""
        self.closes += 1


class AsyncBody(httpx.AsyncByteStream):
    """Expose asynchronous cancellation without buffering."""

    def __init__(self, error=None):
        """Configure a controlled asynchronous stream failure."""
        self.error = error
        self.closes = 0

    async def __aiter__(self):
        """Yield asynchronous bytes without reading ahead."""
        yield b"one"
        if self.error:
            raise self.error
        yield b"two"

    async def aclose(self):
        """Count asynchronous native close calls."""
        self.closes += 1


def test_consolidated_module_owns_attempt_matching(audit_records):
    """Exercise lifecycle matching through the shared, canonical utility module."""
    shared = importlib.import_module("ols.utils.audit_support")
    activate = shared.llm_call_scope()
    with activate():
        attempt = shared.start_attempt(
            shared.Destination("https", "model.invalid", 443)
        )
    identity = records(audit_records)[0]["attempt_id"]
    assert getattr(shared, "_attempts", {}).get(identity) is attempt
    attempt.response_headers(200)
    attempt.finish(response_complete=True)
    attempt.finish(response_complete=False)
    started, completed = records(audit_records)
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["outcome"] == "success"
    assert identity not in support._attempts


@pytest.mark.parametrize("early", [False, True])
def test_native_httpx_body_completion_and_ownership(audit_records, early):
    """Catch pre-reading, finishing at headers, and duplicate closure."""
    body = SyncBody()

    def handler(request):
        assert len(records(audit_records)) == 1
        return httpx.Response(200, stream=body)

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, mounts={"https://bypass.invalid": None}
    ) as client:
        support.instrument_httpx_client(client)
        support.instrument_httpx_client(client)
        assert client._transport.transport is transport
        with (
            dispatch(),
            client.stream(
                "POST", "https://model.invalid/private?token=secret"
            ) as response,
        ):
            assert len(records(audit_records)) == 1
            iterator = response.iter_bytes()
            assert next(iterator) == b"one"
            if not early:
                assert list(iterator) == [b"two"]
        assert body.closes == 1
        complete = records(audit_records)[-1]
        assert complete["response_complete"] is (not early)
        assert "secret" not in json.dumps(records(audit_records))
        assert len(records(audit_records)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [None, asyncio.CancelledError("private"), httpx.ReadError("private")]
)
async def test_native_async_httpx_preserves_cancellation_and_body(audit_records, error):
    """Catch changed native cancellation or early buffered completion."""
    body = AsyncBody(error)

    async def handler(request):
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        support.instrument_httpx_client(client)
        with dispatch():
            async with client.stream("POST", "https://model.invalid") as response:
                assert len(records(audit_records)) == 1
                if error:
                    with pytest.raises(type(error)) as caught:
                        await response.aread()
                    assert caught.value is error
                else:
                    assert await response.aread() == b"onetwo"
        assert body.closes == 1
    assert len(records(audit_records)) == 2
    assert records(audit_records)[-1]["outcome"] == (
        "success"
        if error is None
        else "cancelled" if isinstance(error, asyncio.CancelledError) else "failure"
    )


@asynccontextmanager
async def openai_identity_models(installation, model_class=ChatOpenAI):
    """Own equivalent real SDK models sharing a native cache and controlled HTTP clients."""
    from langchain_core.caches import InMemoryCache

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    native = httpx.Client(transport=httpx.MockTransport(handler))
    native_async = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parameters = {
        "model": "test",
        "api_key": "unused",
        "base_url": "http://127.0.0.1:1/v1",
        "http_client": native,
        "http_async_client": native_async,
        "cache": InMemoryCache(),
        "use_responses_api": False,
    }
    plain = model_class(**parameters)
    second = model_class(**parameters)
    if installation == "existing":
        observed = model_class(**parameters)
        assert support.audit_model(observed) is observed
    else:
        observed = support.audited_model_class(model_class)(**parameters)
    try:
        assert observed.root_client._client is native
        assert observed.root_async_client._client is native_async
        yield plain, second, observed, calls
    finally:
        await asyncio.to_thread(native.close)
        await native_async.aclose()


@pytest.mark.parametrize("installation", ["existing", "factory"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.asyncio
async def test_native_model_public_identity_is_preserved(
    audit_records, installation, enabled
):
    """Catch adapter names or namespaces changing native serialization and cache identity."""
    config.ols_config.audit.logging = (
        AuditLoggingMode.ENABLED if enabled else AuditLoggingMode.DISABLED
    )
    async with openai_identity_models(installation) as (plain, _, observed, calls):
        assert observed.get_name() == plain.get_name() == "ChatOpenAI"
        assert (
            observed.get_name("Input") == plain.get_name("Input") == "ChatOpenAIInput"
        )
        assert observed.get_name("Input", name="custom") == plain.get_name(
            "Input", name="custom"
        )
        assert observed.to_json() == plain.to_json()
        assert observed._serialized == plain._serialized
        assert observed._get_llm_string() == plain._get_llm_string()
        assert observed.get_input_jsonschema() == plain.get_input_jsonschema()
        assert observed.get_output_jsonschema() == plain.get_output_jsonschema()
        assert not calls
        assert not records(audit_records)


@pytest.mark.parametrize("installation", ["existing", "factory"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_model_callback_identity_is_preserved(
    audit_records, installation, enabled, asynchronous
):
    """Catch native callbacks receiving adapter identity instead of native model identity."""
    from langchain_core.callbacks import BaseCallbackHandler

    class Capture(BaseCallbackHandler):
        """Capture only the native model identity and configured callback name."""

        def __init__(self):
            self.rows = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.rows.append((serialized, kwargs.get("name")))

    config.ols_config.audit.logging = (
        AuditLoggingMode.ENABLED if enabled else AuditLoggingMode.DISABLED
    )
    capture = Capture()
    async with openai_identity_models(installation) as (plain, _, observed, calls):
        for model, prompt in (
            (plain, "private baseline"),
            (observed, "private observed"),
        ):
            run_config = {"callbacks": [capture], "run_name": "native-run"}
            response = (
                await model.ainvoke(prompt, config=run_config)
                if asynchronous
                else model.invoke(prompt, config=run_config)
            )
            assert response.content == "ok"
        assert len(capture.rows) == 2
        assert capture.rows[0] == capture.rows[1]
        assert capture.rows[1][1] == "native-run"
        assert len(calls) == 2
        assert len(records(audit_records)) == (2 if enabled else 0)
        assert not support._attempts


@pytest.mark.parametrize("installation", ["existing", "factory"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_model_warm_cache_is_preserved(
    audit_records, installation, enabled, asynchronous
):
    """Catch observer-induced cache misses or synthetic HTTP attempts for cache hits."""
    config.ols_config.audit.logging = (
        AuditLoggingMode.ENABLED if enabled else AuditLoggingMode.DISABLED
    )
    async with openai_identity_models(installation) as (plain, second, observed, calls):
        for model in (plain, second, observed):
            response = (
                await model.ainvoke("private cached")
                if asynchronous
                else model.invoke("private cached")
            )
            assert response.content == "ok"
            assert len(calls) == 1
        assert not records(audit_records)
        fresh = (
            await observed.ainvoke("private fresh")
            if asynchronous
            else observed.invoke("private fresh")
        )
        assert fresh.content == "ok"
        assert len(calls) == 2
        assert len(records(audit_records)) == (2 if enabled else 0)
        assert not support._attempts


@pytest.mark.parametrize(
    "model_class",
    [
        ChatOpenAI,
        pytest.param("langchain_openai.AzureChatOpenAI", id="AzureChatOpenAI"),
        pytest.param("langchain_anthropic.ChatAnthropic", id="ChatAnthropic"),
        pytest.param(
            "langchain_google_genai.ChatGoogleGenerativeAI", id="ChatGoogleGenerativeAI"
        ),
        pytest.param(
            "langchain_google_vertexai.model_garden.ChatAnthropicVertex",
            id="ChatAnthropicVertex",
        ),
        pytest.param("langchain_aws.ChatBedrockConverse", id="ChatBedrockConverse"),
        pytest.param("langchain_ibm.ChatWatsonx", id="ChatWatsonx"),
    ],
)
def test_native_provider_class_identity_is_preserved(model_class):
    """Catch name/namespace drift across native providers, including default namespaces."""
    if isinstance(model_class, str):
        module, name = model_class.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module), name)
    observed = support.audited_model_class(model_class)
    assert observed.__name__ == model_class.__name__
    assert observed.__module__ == model_class.__module__
    assert observed.__qualname__ == model_class.__qualname__
    assert observed.lc_id() == model_class.lc_id()
    assert observed.get_lc_namespace() == model_class.get_lc_namespace()


@pytest.mark.parametrize("installation", ["existing", "factory"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "run_id",
    [
        UUID("12345678-1234-4234-8234-123456789abc"),
        "12345678-1234-4234-8234-123456789abc",
        "PRIVATE_USER@example.invalid",
    ],
)
@pytest.mark.asyncio
async def test_native_run_id_is_not_audit_identity(
    audit_records, installation, asynchronous, run_id
):
    """Catch copying caller identity or reusing it across independent native invocations."""
    from langchain_core.callbacks import BaseCallbackHandler
    from pydantic import ValidationError

    class Capture(BaseCallbackHandler):
        """Capture native callback IDs without changing callback behavior."""

        def __init__(self):
            self.starts = []
            self.ends = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.starts.append(kwargs["run_id"])

        def on_llm_end(self, response, **kwargs):
            self.ends.append(kwargs["run_id"])

    capture = Capture()
    errors = []
    async with openai_identity_models(installation) as (plain, _, observed, calls):
        for index, model in enumerate((plain, observed, observed)):
            try:
                run_config = {"callbacks": [capture], "run_id": run_id}
                result = (
                    await model.ainvoke(f"private prompt {index}", config=run_config)
                    if asynchronous
                    else model.invoke(f"private prompt {index}", config=run_config)
                )
                assert result.content == "ok"
                errors.append(None)
            except ValidationError as error:
                errors.append(error.errors(include_url=False))
            assert len(calls) == index + 1
        assert errors[0] == errors[1] == errors[2]
        assert (errors[0] is not None) is (run_id == "PRIVATE_USER@example.invalid")
        assert capture.starts == [run_id] * 3
        assert capture.ends == [run_id] * 3
        data = wire_records(audit_records)
        assert len(data) == 4
        starts = [row for row in data if row["event"] == "outbound.request.started"]
        assert len(starts) == 2
        assert len({row["attempt_id"] for row in starts}) == 2
        for started, completed in zip(data[::2], data[1::2], strict=True):
            assert completed["attempt_id"] == started["attempt_id"]
            assert completed["outcome"] == "success"
        assert "PRIVATE_USER" not in json.dumps(data)
        assert not support._attempts
        assert support._call.get() is None


@pytest.mark.parametrize("mode", ["nested_sync", "nested_async", "async_fallback"])
@pytest.mark.asyncio
async def test_native_nested_dispatch_preserves_transport_scope(audit_records, mode):
    """Catch losing HTTP observation in nested wrappers or native async-to-sync fallback."""

    class SyncOnlyOpenAI(ChatOpenAI):
        """Exercise the real LangChain executor fallback with a native sync SDK."""

        _agenerate = BaseChatModel._agenerate

    model_class = SyncOnlyOpenAI if mode == "async_fallback" else ChatOpenAI
    async with openai_identity_models("existing", model_class) as (
        _,
        _,
        observed,
        calls,
    ):
        if mode != "async_fallback":
            support.audit_model(observed)
        result = (
            observed.invoke("private nested")
            if mode == "nested_sync"
            else await observed.ainvoke("private nested")
        )
        assert result.content == "ok"
        started, completed = wire_records(audit_records)
        assert len(calls) == 1
        assert started["event"] == "outbound.request.started"
        assert completed["event"] == "outbound.request.finished"
        assert started["attempt_id"] == completed["attempt_id"]
        assert started["request.kind"] == completed["request.kind"] == "llm"
        assert not support._attempts
        assert support._call.get() is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_reentrant_invocations_do_not_share_audit_identity(
    audit_records, asynchronous
):
    """Catch merging independent nested calls that reuse one caller-supplied native ID."""
    calls = []
    parents = []
    run_id = UUID("12345678-1234-4234-8234-123456789abc")

    def response():
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            parent = support._call.get()
            for prompt in ("private child one", "private child two"):
                assert (
                    observed.invoke(prompt, config={"run_id": run_id}).content == "ok"
                )
            parents.append(parent is support._call.get())
        return response()

    async def async_handler(request):
        calls.append(request)
        if len(calls) == 1:
            parent = support._call.get()
            children = await asyncio.gather(
                *(
                    observed.ainvoke(prompt, config={"run_id": run_id})
                    for prompt in ("private child one", "private child two")
                )
            )
            assert all(child.content == "ok" for child in children)
            parents.append(parent is support._call.get())
        return response()

    native = httpx.Client(transport=httpx.MockTransport(handler))
    native_async = httpx.AsyncClient(transport=httpx.MockTransport(async_handler))
    observed = support.audit_model(
        ChatOpenAI(
            model="test",
            api_key="unused",
            base_url="http://127.0.0.1:1/v1",
            http_client=native,
            http_async_client=native_async,
            use_responses_api=False,
        ),
    )
    try:
        result = (
            await observed.ainvoke("private parent", config={"run_id": run_id})
            if asynchronous
            else observed.invoke("private parent", config={"run_id": run_id})
        )
        assert result.content == "ok"
        assert len(calls) == 3
        assert parents == [True]
        data = wire_records(audit_records)
        assert len(data) == 6
        starts = {
            row["attempt_id"]: row
            for row in data
            if row["event"] == "outbound.request.started"
        }
        assert len(starts) == 3
        assert {row["request.kind"] for row in data} == {"llm"}
        assert all(row["attempt_id"] in starts for row in data)
        assert not support._attempts
        assert support._call.get() is None
    finally:
        await asyncio.to_thread(native.close)
        await native_async.aclose()


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_real_openai_retry_attempts_remain_distinct(
    audit_records, responses, asynchronous
):
    """Catch logical-only request observation and breaking API selection."""
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "retry", "type": "rate_limit_error"}},
                headers={"retry-after-ms": "1"},
            )
        if responses:
            return httpx.Response(
                200,
                json={
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "test",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_test",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "ok", "annotations": []}
                            ],
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    native = httpx.Client(transport=httpx.MockTransport(handler))
    native_async = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = support.audited_model_class(ChatOpenAI)(
        model="test",
        api_key="unused",
        http_client=native,
        http_async_client=native_async,
        use_responses_api=responses,
        max_retries=1,
    )
    run_id = UUID("12345678-1234-4234-8234-123456789abc")
    try:
        result = (
            await model.ainvoke("private input", config={"run_id": run_id})
            if asynchronous
            else model.invoke("private input", config={"run_id": run_id})
        )
        assert "ok" in str(result.content)
        assert isinstance(model, ChatOpenAI)
        data = wire_records(audit_records)
        assert len(data) == 4
        starts = [row for row in data if row["event"] == "outbound.request.started"]
        finishes = [row for row in data if row["event"] == "outbound.request.finished"]
        assert len({row["attempt_id"] for row in starts}) == 2
        assert {row["attempt_id"] for row in starts} == {
            row["attempt_id"] for row in finishes
        }
        assert [row["http.response.status_code"] for row in finishes] == [429, 200]
        assert "private input" not in json.dumps(data)
        assert all(
            request.url.path.endswith("responses" if responses else "chat/completions")
            for request in calls
        )
    finally:
        await asyncio.to_thread(native.close)
        await native_async.aclose()


@pytest.mark.parametrize("fault", [None, "lookup", "accessor", "context"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_openai_tls_metadata_is_best_effort(
    audit_records, asynchronous, fault
):
    """Catch optional TLS metadata errors turning successful SDK requests into retries."""
    calls = []
    context = ssl.create_default_context()

    class TLSObject:
        """Expose native verification context or an unavailable metadata property."""

        @property
        def context(self):
            """Return evidence without participating in transport verification."""
            if fault == "context":
                raise RuntimeError("private TLS context unavailable")
            return context

    class NetworkStream:
        """Expose an optional metadata accessor independently of successful I/O."""

        @property
        def get_extra_info(self):
            """Return the native accessor, including failure during attribute lookup."""
            if fault == "lookup":
                raise RuntimeError("private metadata accessor unavailable")

            def extra_info(name):
                if fault == "accessor":
                    raise RuntimeError("private TLS metadata unavailable")
                return TLSObject() if name == "ssl_object" else None

            return extra_info

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
            extensions={"network_stream": NetworkStream()},
        )

    native = httpx.Client(transport=httpx.MockTransport(handler))
    native_async = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = support.audit_model(
        ChatOpenAI(
            model="test",
            api_key="unused",
            base_url="https://127.0.0.1/v1",
            http_client=native,
            http_async_client=native_async,
            use_responses_api=False,
            max_retries=1,
        ),
    )
    try:
        result = (
            await model.ainvoke("private input")
            if asynchronous
            else model.invoke("private input")
        )
        assert result.content == "ok"
        assert len(calls) == 1
        started, completed = records(audit_records)
        assert completed["attempt_id"] == started["attempt_id"]
        assert completed["outcome"] == "success"
        assert completed["http_status"] == 200
        assert completed["response_complete"] is True
        assert wire_records(audit_records)[-1]["outcome"] == "success"
        if fault is None:
            assert wire_records(audit_records)[-1]["http.response.status_code"] == 200
        assert "private" not in json.dumps([started, completed])
        assert not support._attempts
    finally:
        await asyncio.to_thread(native.close)
        await native_async.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_tls_metadata_guard_preserves_native_verification_failure(
    audit_records, asynchronous
):
    """Catch metadata fault isolation accidentally swallowing genuine transport TLS failures."""
    certificate_error = ssl.SSLCertVerificationError("private certificate rejection")
    failure = httpx.ConnectError("private native TLS failure")
    calls = []

    def handler(request):
        calls.append(request)
        raise failure from certificate_error

    native = httpx.Client(transport=httpx.MockTransport(handler))
    native_async = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = native_async if asynchronous else native
    support.instrument_httpx_client(client)
    try:
        with dispatch(), pytest.raises(httpx.ConnectError) as caught:
            if asynchronous:
                await client.post("https://127.0.0.1/chat/completions")
            else:
                client.post("https://127.0.0.1/chat/completions")
        assert caught.value is failure
        assert len(calls) == 1
        started, completed = records(audit_records)
        assert completed["attempt_id"] == started["attempt_id"]
        assert completed["outcome"] == "failure"
        assert completed["failure_category"] == "tls"
        assert wire_records(audit_records)[-1]["error.type"] == "tls_error"
        assert completed["response_complete"] is False
        assert "private" not in json.dumps([started, completed])
        assert not support._attempts
    finally:
        await asyncio.to_thread(native.close)
        await native_async.aclose()


def test_caller_http_between_stream_yields_is_not_a_model_attempt(audit_records):
    """Catch dispatch scope remaining active in a consuming caller."""
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"ok")
        )
    )
    support.instrument_httpx_client(client)

    class StreamingModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "controlled"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise AssertionError("not used")

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            client.get("https://model.invalid")
            yield AIMessageChunk(content="one")
            client.get("https://model.invalid")
            yield AIMessageChunk(content="two")

    model = support.audited_model_class(StreamingModel)()
    iterator = model._stream([])
    assert next(iterator).content == "one"
    client.get("https://unrelated.invalid")
    assert next(iterator).content == "two"
    iterator.close()
    client.close()
    assert len(records(audit_records)) == 4
    assert {r["destination_host"] for r in records(audit_records)} == {"model.invalid"}


@pytest.mark.parametrize(
    "provider", ["openai", "azure_openai", "rhoai_vllm", "rhelai_vllm"]
)
def test_service_loader_attaches_native_audit(provider, audit_records):
    """Catch a working adapter that is never wired into actual service loads."""
    config.config.llm_providers = LLMProviders(
        [
            {
                "name": provider,
                "type": provider,
                "url": "https://model.invalid",
                "credentials_path": "tests/config/secret/apitoken",
                "deployment_name": "test",
                "api_version": "2024-02-15-preview",
                "models": [{"name": "test", "options": {"use_responses_api": False}}],
            }
        ]
    )
    model = load_llm(provider, "test")

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": []})
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    model.http_client._transport = httpx.MockTransport(handler)
    try:
        assert model.invoke("private").content == "ok"
        assert len(records(audit_records)) == 2
        with dispatch():
            assert model.root_client.models.list().data == []
        assert len(records(audit_records)) == 2
    finally:
        model.http_client.close()


def test_real_anthropic_native_client_and_tools_unchanged(audit_records, monkeypatch):
    """Catch missing lazy Anthropic client observation and changed binding."""
    import httpx2
    from langchain_anthropic import ChatAnthropic

    model = support.audited_model_class(ChatAnthropic)(
        model="claude-test",
        api_key="unused",
        max_retries=0,
    )
    native = httpx2.Client(trust_env=False)
    monkeypatch.setattr(
        "langchain_anthropic.chat_models._get_default_httpx_client",
        lambda **kwargs: native,
    )
    assert model._client._client is native

    def handler(request):
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 1})
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    native._transport = httpx2.MockTransport(handler)
    try:
        assert model.invoke("private").content == "ok"
        assert len(records(audit_records)) == 2
        with dispatch():
            assert (
                model._client.messages.count_tokens(
                    model="claude-test",
                    messages=[{"role": "user", "content": "private"}],
                ).input_tokens
                == 1
            )
        assert len(records(audit_records)) == 2
        assert (
            model.bind_tools(
                [
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ]
            ).bound
            is model
        )
        assert model._client._client is native
    finally:
        native.close()


@pytest.fixture()
def local_http_server():
    """Serve controlled SDK responses on loopback without cloud access."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            calls.append((self.path, data))
            status, payload = self.server.respond(self.path, data)
            encoded = (
                payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            )
            self.send_response(status)
            self.send_header(
                "Content-Type",
                (
                    "application/vnd.amazon.eventstream"
                    if isinstance(payload, bytes)
                    else "application/json"
                ),
            )
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST  # noqa: N815

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.respond = lambda path, data: (200, {})
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server, f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()
    server.server_close()
    worker.join(2)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "configured",
    [
        "projects/private-project/locations/us-central1/publishers/google/models/gemini-2.5-flash",
        "projects/private-project/locations/us-central1/models/private-model",
        "projects/private-project/locations/us-central1/endpoints/private-endpoint",
    ],
)
@pytest.mark.asyncio
async def test_loaded_vertex_resource_preserves_native_request(
    audit_records, monkeypatch, asynchronous, configured
):
    """Catch changing SDK resource selection while preserving native client behavior."""
    from google import genai
    from google.genai import types
    from google.oauth2.credentials import Credentials

    from ols.app.models.config import ProviderConfig

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            },
        )

    native = httpx.Client(transport=httpx.MockTransport(handler))
    native_async = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = genai.Client(
        vertexai=True,
        project="private-project",
        location="us-central1",
        credentials=Credentials(token="unused"),
        http_options=types.HttpOptions(
            base_url="http://127.0.0.1:1",
            httpx_client=native,
            httpx_async_client=native_async,
        ),
    )
    credential_json = json.dumps(
        {
            "type": "authorized_user",
            "client_id": "unused",
            "client_secret": "unused",
            "refresh_token": "unused",
            "token": "unused",
            "expiry": "2099-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(ProviderConfig, "get_credentials", lambda self: credential_json)
    config.config.llm_providers = LLMProviders(
        [
            {
                "name": "vertex",
                "type": "google_vertex",
                "url": "http://127.0.0.1:1",
                "google_vertex_config": {
                    "project": "private-project",
                    "location": "us-central1",
                },
                "models": [{"name": configured}],
            }
        ]
    )
    try:
        with patch("langchain_google_genai.chat_models.Client", return_value=client):
            model = load_llm("vertex", configured)
        response = (
            await model.ainvoke("private input")
            if asynchronous
            else model.invoke("private input")
        )
        assert response.content == "ok"
        assert model.model == configured
        assert len(calls) == 1
        assert calls[0].url.path.endswith("/" + configured + ":generateContent")
        started, completed = records(audit_records)
        assert completed["outcome"] == "success"
        assert completed["response_complete"] is True
        assert completed["attempt_id"] == started["attempt_id"]
        assert "private" not in json.dumps([started, completed])
        assert not support._attempts
    finally:
        client.close()
        await client.aio.aclose()
        await native_async.aclose()


@pytest.mark.parametrize("backend", ["sync", "httpx_async", "aiohttp"])
@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.asyncio
async def test_real_google_native_backend_and_auxiliary_exclusion(
    audit_records, local_http_server, backend, refresh
):
    """Catch backend substitution, lost refresh auditing, and helper misclassification."""
    from google import genai
    from google.genai import types
    from langchain_google_genai import ChatGoogleGenerativeAI

    from ols.src.llms.providers.utils import load_vertex_credentials

    server, url, calls = local_http_server
    server.respond = lambda path, data: (
        200,
        (
            {"totalTokens": 4}
            if path.endswith(":countTokens")
            else {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            }
        ),
    )
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient() if backend == "httpx_async" else None
    previous_response = server.respond
    server.respond = lambda path, data: (
        (
            200,
            {"access_token": "test-token", "expires_in": 3600, "token_type": "Bearer"},
        )
        if path == "/token"
        else previous_response(path, data)
    )
    credentials = load_vertex_credentials(
        json.dumps(
            {
                "type": "authorized_user",
                "token": None if refresh else "test-token",
                "expiry": "1970-01-01T00:00:00Z" if refresh else "2099-01-01T00:00:00Z",
                "refresh_token": "refresh",
                "client_id": "client",
                "client_secret": "secret",
            }
        )
    )
    credentials._token_uri = url + "/token"
    client = genai.Client(
        vertexai=True,
        project="project",
        location="global",
        credentials=credentials,
        http_options=types.HttpOptions(
            base_url=url, httpx_client=sync_client, httpx_async_client=async_client
        ),
    )
    with patch("langchain_google_genai.chat_models.Client", return_value=client):
        model = support.audited_model_class(
            ChatGoogleGenerativeAI,
        )(
            model="gemini-test",
            base_url=url,
            vertexai=True,
            project="project",
            location="global",
            credentials=credentials,
        )
    try:
        response = (
            model.invoke("private")
            if backend == "sync"
            else await model.ainvoke("private")
        )
        assert response.content == "ok"
        data = records(audit_records)
        llm_data = [item for item in data if item["event"].startswith("llm.")]
        auth_data = [item for item in data if item["event"].startswith("auth.")]
        assert len(llm_data) == 2
        assert len(auth_data) == (2 if refresh else 0)
        if refresh:
            assert auth_data[0]["attempt_id"] != llm_data[0]["attempt_id"]
        assert "secret" not in json.dumps(data)
        with dispatch():
            client.models.count_tokens(model="gemini-test", contents="private")
        assert len(records(audit_records)) == len(data)
        assert client._api_client._httpx_client is sync_client
        assert client._api_client._credentials is credentials
        assert len(calls) == (3 if refresh else 2)
    finally:
        client.close()
        await client.aio.aclose()
        await asyncio.to_thread(sync_client.close)
        if async_client:
            await async_client.aclose()


def test_native_bedrock_retries_and_real_converse(audit_records, local_http_server):
    """Catch missing native urllib3 attempts and changing SDK request signing."""
    import boto3
    from botocore.config import Config
    from langchain_aws import ChatBedrockConverse

    server, url, calls = local_http_server
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }
    server.respond = lambda path, data: (
        (503, {"message": "retry"}) if len(calls) == 1 else (200, response)
    )
    native = boto3.client(
        "bedrock-runtime",
        endpoint_url=url,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(retries={"max_attempts": 1}, connect_timeout=2, read_timeout=2),
    )
    model = support.audited_model_class(ChatBedrockConverse)(
        model_id="anthropic.test",
        region_name="us-east-1",
        client=native,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    try:
        assert model.invoke("private").content == "ok"
        assert len(calls) == 2
        data = wire_records(audit_records)
        assert len(data) == 4
        starts = [row for row in data if row["event"] == "outbound.request.started"]
        finishes = [row for row in data if row["event"] == "outbound.request.finished"]
        assert len({row["attempt_id"] for row in starts}) == 2
        assert [row["http.response.status_code"] for row in finishes] == [503, 200]
        with dispatch():
            auxiliary = native.count_tokens(
                modelId="anthropic.claude-test-v1:0",
                input={
                    "converse": {
                        "messages": [{"role": "user", "content": [{"text": "private"}]}]
                    }
                },
            )
        assert auxiliary["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert len(records(audit_records)) == 4
    finally:
        native.close()


def test_native_vertex_anthropic_client(audit_records):
    """Catch missing Vertex Anthropic SDK transport wiring."""
    import httpx2
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    def handler(request):
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as native:
        model = support.audited_model_class(
            ChatAnthropicVertex,
        )(
            model_name="claude-test",
            project="project",
            location="us-east5",
            access_token="placeholder",
            http_client=native,
        )
        assert model.invoke("private").content == "ok"
        assert len(records(audit_records)) == 2
        assert model.client._client is native


def _aws_frame(name, payload):
    headers = b""
    for key, value in (
        (":message-type", "event"),
        (":event-type", name),
        (":content-type", "application/json"),
    ):
        key_bytes, value_bytes = key.encode(), value.encode()
        headers += (
            bytes([len(key_bytes)])
            + key_bytes
            + b"\x07"
            + struct.pack(">H", len(value_bytes))
            + value_bytes
        )
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


def test_native_bedrock_eventstream_consumption(audit_records, local_http_server):
    """Catch premature completion and breaking botocore CRC framing."""
    import boto3
    from langchain_aws import ChatBedrockConverse

    server, url, _ = local_http_server
    wire = b"".join(
        [
            _aws_frame("messageStart", {"role": "assistant"}),
            _aws_frame(
                "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "ok"}}
            ),
            _aws_frame("messageStop", {"stopReason": "end_turn"}),
            _aws_frame(
                "metadata",
                {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "metrics": {"latencyMs": 1},
                },
            ),
        ]
    )
    server.respond = lambda path, data: (200, wire)
    native = boto3.client(
        "bedrock-runtime",
        endpoint_url=url,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    model = support.audited_model_class(ChatBedrockConverse)(
        model_id="anthropic.test",
        region_name="us-east-1",
        client=native,
        disable_streaming=False,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    try:
        iterator = model.stream("private")
        first = next(iterator)
        assert len(records(audit_records)) == 1
        chunks = [first, *iterator]
        assert "ok" in str([chunk.content for chunk in chunks])
        assert records(audit_records)[-1]["response_complete"] is True
    finally:
        native.close()


def test_real_openai_done_closes_without_forcing_http_eof(audit_records):
    """Catch early header completion or inventing EOF after SDK DONE."""

    class SSEBody(httpx.SyncByteStream):
        def __iter__(self):
            yield (
                b'data: {"id":"chat_test","object":"chat.completion.chunk",'
                b'"created":1,"model":"test","choices":[{"index":0,"delta":'
                b'{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n'
            )
            assert len(records(audit_records)) == 1
            yield (
                b'data: {"id":"chat_test","object":"chat.completion.chunk",'
                b'"created":1,"model":"test","choices":[{"index":0,"delta":{},'
                b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            )

    native = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=SSEBody()
            )
        )
    )
    model = support.audited_model_class(ChatOpenAI)(
        model="test",
        api_key="unused",
        http_client=native,
        use_responses_api=False,
    )
    try:
        assert "".join(str(chunk.content) for chunk in model.stream("private")) == "ok"
        assert len(records(audit_records)) == 2
        assert records(audit_records)[-1]["response_complete"] is False
        assert records(audit_records)[-1]["failure_category"] == "incomplete_response"
    finally:
        native.close()


@pytest.fixture
def native_watsonx(monkeypatch):  # noqa: C901
    """Control only HTTPcore wire I/O; retain IBM clients, IAM and retry transports."""
    import base64

    import httpcore
    from ibm_watsonx_ai import APIClient, Credentials
    from langchain_ibm import ChatWatsonx

    wire = []
    control = {}
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": 4102444800}).encode())
        .decode()
        .rstrip("=")
    )
    token = f"eyJhbGciOiJub25lIn0.{payload}.placeholder"
    inference_count = 0

    class BrokenWireBody(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __init__(self, first, error):
            self.first, self.error = first, error

        def __iter__(self):
            yield self.first
            raise self.error

        async def __aiter__(self):
            yield self.first
            raise self.error

    def reply(request):
        nonlocal inference_count
        path = request.url.target.decode().split("?", 1)[0]
        wire.append(path)
        status = 200
        if path.endswith("/text/chat") and control.get("error") is not None:
            raise control.pop("error")
        if path.endswith("/text/chat_stream"):
            events = [
                {
                    "id": "test",
                    "created": 1,
                    "model_id": "ibm/granite-4-h-small",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "ok"}}
                    ],
                },
                {
                    "id": "test",
                    "created": 1,
                    "model_id": "ibm/granite-4-h-small",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            data = "".join(
                f"data: {json.dumps(event)}\n\n" for event in events
            ).encode()
            body = httpx.ByteStream(data)
            if control.get("stream_error") is not None:
                first = f"data: {json.dumps(events[0])}\n\n".encode()
                body = BrokenWireBody(first, control["stream_error"])
            return httpcore.Response(
                200, headers=[(b"content-type", b"text/event-stream")], content=body
            )
        if path.endswith("/identity/token"):
            body = {"access_token": token, "expires_in": 3600, "token_type": "Bearer"}
        elif path.endswith("/text/chat"):
            inference_count += 1
            if inference_count == 1:
                status, body = 503, {"error": "retry"}
            else:
                body = {
                    "id": "chat-test",
                    "model_id": "ibm/granite-4-h-small",
                    "created": 1,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
        else:
            body = {"result": {"tokens": ["ignored"], "token_count": 1}}
        return httpcore.Response(
            status,
            headers=[(b"content-type", b"application/json"), (b"retry-after", b"0")],
            content=httpx.ByteStream(json.dumps(body).encode()),
        )

    def sync_wire(_pool, request):
        return reply(request)

    async def async_wire(_pool, request):
        return reply(request)

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", sync_wire)
    monkeypatch.setattr(
        httpcore.AsyncConnectionPool, "handle_async_request", async_wire
    )
    client = APIClient(
        Credentials(url="https://us-south.ml.cloud.ibm.com", api_key="placeholder"),
        project_id="00000000-0000-0000-0000-000000000001",
        scope_validation=False,
    )
    model = ChatWatsonx(
        watsonx_client=client, model_id="ibm/granite-4-h-small", validate_model=False
    )
    yield model, client, wire, control
    client.httpx_client.close()


def test_loader_wires_real_watsonx_model(native_watsonx, audit_records, monkeypatch):
    """Catch a native IBM adapter omitted from the service loader."""
    from unittest.mock import Mock

    from ols.src.llms.providers.registry import LLMProvidersRegistry

    model, _, _, _ = native_watsonx
    config.config.llm_providers = LLMProviders(
        [
            {
                "name": "ibm",
                "type": "watsonx",
                "url": "https://us-south.ml.cloud.ibm.com",
                "project_id": "00000000-0000-0000-0000-000000000001",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [{"name": model.model_id}],
            }
        ]
    )
    provider = Mock()
    provider.return_value.load.return_value = model
    monkeypatch.setitem(LLMProvidersRegistry.llm_providers, "watsonx", provider)
    loaded = load_llm("ibm", model.model_id)
    assert loaded is model
    assert loaded.invoke("ignored").content == "ok"
    started = [r for r in records(audit_records) if r["event"].endswith("started")]
    assert len(started) == 2
    provider.assert_called_once_with(
        model.model_id, config.config.llm_providers.providers["ibm"], {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_native_watsonx_retry_and_nested_refresh(
    native_watsonx, audit_records, use_async
):
    """Catch an outer-only IBM wrapper, phantom IAM/helper events and backend replacement."""
    from datetime import datetime, timedelta

    model, client, wire, _ = native_watsonx
    assert wire == ["/identity/token"]
    assert not records(audit_records)
    sync_client, async_client = client.httpx_client, client.async_httpx_client
    native_transport = sync_client._transport
    original_pool = native_transport._pool
    native_model = support.audit_model(model)
    assert native_model is model
    assert client.httpx_client is sync_client
    assert client.async_httpx_client is async_client
    assert sync_client._transport is native_transport
    assert native_transport._pool is original_pool
    with dispatch():
        assert model.watsonx_model.tokenize("ignored")["result"]["token_count"] == 1
    assert not records(audit_records)
    client._auth_method._hardcoded_expiration_datetime = datetime.now() - timedelta(
        seconds=1
    )
    result = await model.ainvoke("ignored") if use_async else model.invoke("ignored")
    assert result.content == "ok"
    assert sum(path.endswith("/identity/token") for path in wire) == 2
    assert sum(path.endswith("/text/chat") for path in wire) == 2
    data = wire_records(audit_records)
    started = [row for row in data if row["event"] == "outbound.request.started"]
    completed = [row for row in data if row["event"] == "outbound.request.finished"]
    assert len(started) == len(completed) == 2
    assert len({row["attempt_id"] for row in started}) == 2
    assert [row["http.response.status_code"] for row in completed] == [503, 200]
    assert not sync_client.is_closed
    await async_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_native_watsonx_tls_fallback_reobserves_recreated_pool(
    native_watsonx, audit_records, use_async
):
    """Catch audit loss when native IBM TLS fallback rebuilds its HTTPcore pool."""
    import ssl

    import httpcore
    from ibm_watsonx_ai._wrappers.httpx import GlobalHttpxSettings

    model, client, wire, control = native_watsonx
    native = client.async_httpx_client if use_async else client.httpx_client
    transport, old_pool = native._transport, native._transport._pool
    transport.allow_ssl_fallback = True
    transport.original_verify = None
    primary = httpcore.ConnectError("certificate verify failed")
    primary.__cause__ = ssl.SSLCertVerificationError("secret certificate")
    control["error"] = primary
    model = support.audit_model(model)
    previous_verify = GlobalHttpxSettings.verify
    try:
        result = (
            await model.ainvoke("ignored") if use_async else model.invoke("ignored")
        )
        assert result.content == "ok"
        assert native._transport is transport
        assert transport._pool is not old_pool
        assert transport._pool._ols_audit_installed
        completed = [
            r for r in records(audit_records) if r["event"].endswith("completed")
        ]
        assert len(completed) == 3
        assert completed[0]["failure_category"] == "tls"
        assert [r.get("http_status") for r in completed] == [None, 503, 200]
        assert sum(path.endswith("/text/chat") for path in wire) == 3
        assert len({r["attempt_id"] for r in completed}) == 3
        assert not native.is_closed
    finally:
        GlobalHttpxSettings.verify = previous_verify
        await client.async_httpx_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("early", [False, True])
async def test_native_watsonx_stream_lifecycle(
    native_watsonx, audit_records, use_async, early
):
    """Catch premature completion at headers or loss of native IBM streaming cleanup."""
    model, client, _, _ = native_watsonx
    model = support.audit_model(model)
    if use_async:
        stream = model.astream("ignored")
        first = await anext(stream)
        assert first.content == "ok"
        assert len(records(audit_records)) == 1
        if early:
            await stream.aclose()
        else:
            async for _ in stream:
                pass
    else:
        stream = model.stream("ignored")
        first = next(stream)
        assert first.content == "ok"
        assert len(records(audit_records)) == 1
        if early:
            stream.close()
        else:
            list(stream)
    assert not client.httpx_client.is_closed
    assert not client.async_httpx_client.is_closed
    # IBM/LangChain closes nested asynchronous generators through native finalizers.
    # Observe that real close; do not report HTTP completion merely on logical close.
    if use_async and early:
        for _ in range(10):
            if len(records(audit_records)) == 2:
                break
            await asyncio.sleep(0)
    completed = records(audit_records)[-1]
    assert completed["event"] == "llm.request.completed"
    assert completed["response_complete"] is not early
    await client.async_httpx_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_native_watsonx_proxy_mount_is_retained(
    native_watsonx, audit_records, use_async
):
    """Catch replacement of IBM proxy/no-proxy routing or target/proxy metadata."""
    model, client, _, _ = native_watsonx
    native = client.async_httpx_client if use_async else client.httpx_client
    original_transport = native._transport
    mounted = type(original_transport)(
        retries=1,
        backoff_factor=0,
        status_forcelist=[503],
        verify_initial=True,
        allow_ssl_fallback=False,
        proxy="http://127.0.0.1:3128",
    )
    pattern = httpx._utils.URLPattern("https://")
    native._mounts[pattern] = mounted
    model = support.audit_model(model)
    result = await model.ainvoke("ignored") if use_async else model.invoke("ignored")
    assert result.content == "ok"
    assert native._transport is original_transport
    assert native._mounts[pattern] is mounted
    started = [r for r in records(audit_records) if r["event"].endswith("started")]
    assert len(started) == 2
    assert all(r["destination_host"] == "127.0.0.1" for r in started)
    assert all(r["destination_port"] == 3128 for r in started)
    assert all(r["target_host"] == "us-south.ml.cloud.ibm.com" for r in started)
    await client.async_httpx_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ValueError("secret"), asyncio.CancelledError("secret")]
)
async def test_native_watsonx_stream_failure_preserves_error(
    native_watsonx, audit_records, error
):
    """Catch replacement of native stream failure/cancellation and leaked active entries."""
    model, client, _, control = native_watsonx
    control["stream_error"] = error
    model = support.audit_model(model)
    stream = model.astream("ignored")
    assert (await anext(stream)).content == "ok"
    with pytest.raises(type(error)) as raised:
        await anext(stream)
    assert raised.value is error
    started, completed = records(audit_records)
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["response_complete"] is False
    expected = (
        "cancelled" if isinstance(error, asyncio.CancelledError) else "stream_error"
    )
    assert completed["failure_category"] == expected
    assert "secret" not in json.dumps(completed)
    assert not client.async_httpx_client.is_closed
    await stream.aclose()
    await client.async_httpx_client.aclose()


@pytest.mark.asyncio
async def test_watsonx_iam_acquisition_and_refresh_are_audited(
    native_watsonx, audit_records
):
    """Catch missed IBM IAM token acquisition/refresh or leaked credential material."""
    from ibm_watsonx_ai import APIClient, Credentials

    wire = native_watsonx[2]
    with support.watsonx_auth_audit_scope():
        client = APIClient(
            Credentials(
                url="https://us-south.ml.cloud.ibm.com", api_key="placeholder-secret"
            ),
            project_id="00000000-0000-0000-0000-000000000001",
            scope_validation=False,
        )
    try:
        auth = client._auth_method
        auth._hardcoded_expiration_datetime = datetime.now() - timedelta(seconds=1)
        auth._refreshing_timedelta = timedelta(0)
        auth.get_token()
        auth._hardcoded_expiration_datetime = datetime.now() - timedelta(seconds=1)
        await auth.aget_token()

        data = wire_records(audit_records)
        started = [item for item in data if item["event"] == "outbound.request.started"]
        completed = [
            item for item in data if item["event"] == "outbound.request.finished"
        ]
        assert len(started) == len(completed) == 3
        assert {item["attempt_id"] for item in started} == {
            item["attempt_id"] for item in completed
        }
        assert {item["request.kind"] for item in data} == {"credential"}
        assert "placeholder-secret" not in json.dumps(data)
        assert "eyJ" not in json.dumps(data)
        assert wire.count("/identity/token") == 4
    finally:
        client.httpx_client.close()
        await client.async_httpx_client.aclose()


def test_httpx_installation_is_serialized_for_shared_native_client(monkeypatch):
    """Catch concurrent installation losing a wrapper's active-response cleanup set."""
    from concurrent.futures import ThreadPoolExecutor

    native_type = support._SyncTransport
    entered, release = threading.Event(), threading.Event()
    second_started, second_constructed = threading.Event(), threading.Event()
    constructions = []

    class ControlledTransport(native_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructions.append(self)
            if len(constructions) == 1:
                entered.set()
                assert release.wait(5)
            else:
                second_constructed.set()

    monkeypatch.setattr(support, "_SyncTransport", ControlledTransport)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200)), trust_env=False
    ) as client:

        def second_install():
            second_started.set()
            support.instrument_httpx_client(client)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(support.instrument_httpx_client, client)
            try:
                assert entered.wait(5)
                second = executor.submit(second_install)
                assert second_started.wait(5)
                assert not second_constructed.wait(0.1)
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        assert len(constructions) == 1
        assert client._transport is constructions[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        "read",
        "readany",
        "readuntil",
        "readline",
        "readchunk",
        "readexactly",
        "read_nowait",
        "iter_chunked",
        "iter_any",
        "iter_chunks",
        "__aiter__",
    ],
)
async def test_native_aiohttp_read_interfaces(audit_records, local_http_server, method):
    """Preserve native reader semantics while observing response completion."""
    import aiohttp

    server, url, _ = local_http_server
    payload = b"one\ntwo\n"
    server.respond = lambda path, data: (200, payload)
    async with aiohttp.ClientSession() as session:
        support.instrument_aiohttp_session(session)
        support.instrument_aiohttp_session(session)
        with dispatch():
            async with session.post(
                url + "/model:generateContent", data=b"input"
            ) as response:
                assert len(records(audit_records)) == 1
                content = response.content
                if method.startswith("iter_") or method == "__aiter__":
                    args = (3,) if method == "iter_chunked" else ()
                    async for _ in getattr(content, method)(*args):
                        pass
                elif method == "read_nowait":
                    await content.read(1)
                    content.read_nowait()
                    await content.read()
                elif method == "readexactly":
                    with pytest.raises(asyncio.IncompleteReadError):
                        await content.readexactly(len(payload) + 1)
                else:
                    await getattr(content, method)()
                    await content.read()
        assert not session.closed
    started, completed = records(audit_records)
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["response_complete"] is (method != "readexactly")


@pytest.mark.parametrize(
    "kind", ["timeout", "connect", "dns", "tls", "proxy", "retry", "protocol"]
)
def test_native_urllib3_errors_remain_native(audit_records, monkeypatch, kind):
    """Classify native urllib3 failures without changing the raised error or retries."""
    import ssl

    import urllib3
    from urllib3 import exceptions as errors

    pool = urllib3.HTTPConnectionPool("127.0.0.1", 1)
    error, category = {
        "timeout": (
            errors.ReadTimeoutError(pool, "/model/converse", "secret"),
            "timeout",
        ),
        "connect": (errors.NewConnectionError(None, "secret"), "connect"),
        "dns": (
            errors.NameResolutionError("127.0.0.1", None, socket.gaierror("secret")),
            "dns",
        ),
        "tls": (errors.SSLError(ssl.SSLCertVerificationError("secret")), "tls"),
        "proxy": (errors.ProxyError("secret", socket.gaierror("secret")), "dns"),
        "retry": (
            errors.MaxRetryError(pool, "/model/converse", socket.gaierror("secret")),
            "dns",
        ),
        "protocol": (errors.ProtocolError("secret"), "transport"),
    }[kind]

    def wire(*args, **kwargs):
        raise error

    monkeypatch.setattr(pool, "_make_request", wire)
    support._instrument_pool(pool)
    with dispatch(), pytest.raises(type(error)) as caught:
        pool.urlopen("POST", "/model/converse", retries=False)
    assert caught.value is error
    started, completed = records(audit_records)
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["failure_category"] == category
    assert "secret" not in json.dumps(completed)
    pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_native_httpx_dispatch_failure_preserves_error(audit_records, use_async):
    """Catch start/cleanup loss before response headers exist."""
    error = httpx.ConnectTimeout("secret")

    def wire(_request):
        raise error

    client = (httpx.AsyncClient if use_async else httpx.Client)(
        transport=httpx.MockTransport(wire)
    )
    support.instrument_httpx_client(client)
    with dispatch(), pytest.raises(httpx.ConnectTimeout) as caught:
        if use_async:
            await client.get("https://model.invalid")
        else:
            client.get("https://model.invalid")
    assert caught.value is error
    started, completed = records(audit_records)
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["failure_category"] == "timeout"
    if use_async:
        await client.aclose()
    else:
        await asyncio.to_thread(client.close)


@pytest.mark.parametrize(
    "mode", ["read", "stream", "early", "error", "cleanup_error", "buffered"]
)
def test_native_urllib3_body_read_close_and_failure(audit_records, mode):
    """Preserve actual urllib3 read, stream, close, and error semantics."""
    import io

    import urllib3

    primary = ValueError("secret")

    class NativeBody(io.BytesIO):
        fail_read = False
        fail_close = False

        def read(self, *args, **kwargs):
            if self.fail_read:
                raise primary
            return super().read(*args, **kwargs)

        def close(self):
            super().close()
            if self.fail_close:
                self.fail_close = False
                raise primary

    body = NativeBody(b"onetwo")
    response = urllib3.HTTPResponse(
        body=b"onetwo" if mode == "buffered" else body,
        status=200,
        headers={"content-length": "6"},
        preload_content=False,
    )
    with dispatch():
        attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
    attempt.response_headers(200)
    support._observe_urllib3_body(response, attempt)
    if mode in {"read", "error", "cleanup_error"}:
        assert response.read(0) == b""
        assert response.read(3) == b"one"
        if mode == "read":
            assert response.read() == b"two"
        else:
            body.fail_read = mode == "error"
            body.fail_close = mode == "cleanup_error"
            with pytest.raises(ValueError) as raised:
                response.read() if mode == "error" else response.close()
            assert raised.value is primary
    elif mode in {"stream", "early"}:
        stream = response.stream(3)
        assert next(stream) == b"one"
        if mode == "stream":
            assert list(stream) == [b"two"]
        stream.close()
    response.close()
    body.close()
    started, completed = records(audit_records)
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["response_complete"] is (mode in {"read", "stream", "buffered"})
    assert "secret" not in json.dumps(completed)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("has_primary", [False, True])
async def test_native_httpx_cleanup_preserves_primary_error(
    audit_records, use_async, has_primary
):
    """Keep the primary failure when native close also fails, without duplicate completion."""
    primary, cleanup = ValueError("primary secret"), RuntimeError("cleanup secret")
    closes = []

    class NativeBody(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            yield b"unused"

        async def __aiter__(self):
            yield b"unused"

        def close(self):
            closes.append(True)
            raise cleanup

        async def aclose(self):
            closes.append(True)
            raise cleanup

    body = NativeBody()
    client = (httpx.AsyncClient if use_async else httpx.Client)(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body)),
    )
    support.instrument_httpx_client(client)
    expected = primary if has_primary else cleanup
    with dispatch(), pytest.raises(type(expected)) as raised:
        if use_async:
            async with client.stream("POST", "https://model.invalid"):
                if has_primary:
                    raise primary
        else:
            with client.stream("POST", "https://model.invalid"):
                if has_primary:
                    raise primary
    assert raised.value is expected
    assert len(records(audit_records)) == 2
    assert records(audit_records)[-1]["response_complete"] is False
    assert closes == [True]
    if use_async:
        await client.aclose()
    else:
        await asyncio.to_thread(client.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_anthropic_async_native_backend(
    audit_records, streaming, monkeypatch
):
    """Exercise installed Anthropic's native HTTPX2 async response and stream interfaces."""
    import httpx2
    from langchain_anthropic import ChatAnthropic

    message = {
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = [
        {
            "type": "message_start",
            "message": {**message, "content": [], "stop_reason": None},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    payload = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()
    closes = []

    class NativeAsyncStream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield payload

        async def aclose(self):
            closes.append(True)

    def handler(_request):
        if streaming:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=NativeAsyncStream(),
            )
        return httpx2.Response(200, json=message)

    native = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), trust_env=False
    )
    monkeypatch.setattr(
        "langchain_anthropic.chat_models._get_default_async_httpx_client",
        lambda **kwargs: native,
    )
    model = support.audit_model(
        ChatAnthropic(model="claude-test", api_key="placeholder", max_retries=0),
    )
    assert model._async_client._client is native
    try:
        if streaming:
            chunks = [chunk async for chunk in model.astream("private")]
            assert "".join(chunk.content for chunk in chunks) == "ok"
            assert closes == [True]
        else:
            assert (await model.ainvoke("private")).content == "ok"
        started, completed = records(audit_records)
        assert started["attempt_id"] == completed["attempt_id"]
        assert completed["response_complete"] is True
        assert completed["outcome"] == "success"
        assert not native.is_closed
    finally:
        await native.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize(
    "error", [None, asyncio.CancelledError("secret"), httpx.ReadError("secret")]
)
async def test_mcp_native_stream_finishes_at_eof_or_close(audit_records, early, error):
    """Catch premature MCP completion, leaked payloads and replaced native failures."""
    body = AsyncBody(error)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=body))
    native = httpx.AsyncClient(transport=transport, trust_env=False)
    with support.mcp_operation(tool_name="echo"):
        assert support.instrument_mcp_http_client(native) is native
        support.instrument_mcp_http_client(native)
    async with native:
        with dispatch():
            async with native.stream(
                "POST", "https://mcp.invalid/private?token=secret", content=b"secret"
            ) as response:
                (started,) = records(audit_records)
                assert started["event"] == "mcp.request.started"
                assert started["mcp_tool_name"] == "echo"
                iterator = response.aiter_bytes()
                assert await anext(iterator) == b"one"
                if not early:
                    if error is not None:
                        with pytest.raises(type(error)) as caught:
                            await anext(iterator)
                        assert caught.value is error
                    else:
                        assert [part async for part in iterator] == [b"two"]
        assert body.closes == 1
    started, completed = records(audit_records)
    assert completed["event"] == "mcp.request.completed"
    assert completed["mcp_tool_name"] == "echo"
    assert started["attempt_id"] == completed["attempt_id"]
    assert completed["response_complete"] is (not early and error is None)
    assert completed["outcome"] == (
        "success"
        if not early and error is None
        else (
            "cancelled"
            if not early and isinstance(error, asyncio.CancelledError)
            else "failure"
        )
    )
    assert not {"provider", "hosting", "model"} & started.keys()
    assert "secret" not in json.dumps([started, completed])
    assert not support._attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("preinstrumented", [False, True])
async def test_mcp_auth_helpers_are_excluded_but_physical_retries_are_audited(
    audit_records,
    preinstrumented,
):
    """Catch credential exchanges becoming MCP attempts or collapsed HTTP auth retries."""
    calls = []

    class NativeAuth(httpx.Auth):
        async def async_auth_flow(self, request):
            yield httpx.Request("POST", "https://mcp.invalid/token", content=b"secret")
            response = yield request
            if response.status_code == 401:
                yield httpx.Request(
                    "POST", "https://mcp.invalid/refresh", content=b"secret"
                )
                yield request

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(
            401 if calls == ["/token", "/mcp"] else 200, content=b"ok"
        )

    transport = httpx.MockTransport(handler)
    auth = NativeAuth()
    timeout = httpx.Timeout(2, read=3)
    native = httpx.AsyncClient(
        transport=transport,
        mounts={"https://mcp.invalid": transport, "https://bypass.invalid": None},
        auth=auth,
        timeout=timeout,
        trust_env=False,
    )

    if preinstrumented:
        support.instrument_httpx_client(native)

    def factory(headers=None, timeout=None, auth=None):
        assert headers == {"x-private": "secret"}
        assert timeout is native.timeout
        assert auth is native.auth
        return native

    create = support.mcp_client_factory(factory)
    with support.mcp_operation():
        client = create(
            headers={"x-private": "secret"}, timeout=native.timeout, auth=auth
        )
    assert client is native
    assert not records(audit_records)
    async with client:
        assert (await client.post("https://mcp.invalid/mcp")).content == b"ok"
    assert calls == ["/token", "/mcp", "/refresh", "/mcp"]
    data = wire_records(audit_records)
    assert [item["event"] for item in data] == [
        "outbound.request.started",
        "outbound.request.finished",
        "outbound.request.started",
        "outbound.request.finished",
    ]
    assert [
        item["http.response.status_code"]
        for item in data
        if "http.response.status_code" in item
    ] == [401, 200]
    assert {item["request.kind"] for item in data} == {"mcp"}
    assert len({item["attempt_id"] for item in data}) == 2
    assert all("gen_ai.tool.name" not in item for item in data)
    assert client.auth is auth
    assert "secret" not in json.dumps(data)
    assert not support._attempts


@pytest.mark.asyncio
async def test_mcp_disabled_audit_preserves_native_requests(audit_records):
    """Catch MCP factory setup emitting starts or disabled auditing blocking HTTP."""
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(204)

    native = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    support.instrument_mcp_http_client(native)
    assert records(audit_records) == []
    config.ols_config.audit.logging = AuditLoggingMode.DISABLED
    async with native:
        assert (await native.delete("https://mcp.invalid/private")).status_code == 204
    assert calls == ["DELETE"]
    assert records(audit_records) == []
    assert not support._attempts


class MCPTestServer(ThreadingHTTPServer):
    """Own loopback-only session state for real MCP SDK tests."""

    def __init__(self):
        """Initialize independent call records and legacy SSE channels."""
        self.calls = []
        self.channels = {}
        self.stopping = threading.Event()
        super().__init__(("127.0.0.1", 0), MCPTestHandler)


class MCPTestHandler(BaseHTTPRequestHandler):
    """Serve the native MCP wire protocol without replacing SDK session behavior."""

    server: MCPTestServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        """Suppress HTTP server output."""

    def reply(self, status, payload=None, *, initialize=False):
        """Send a framed native JSON response with an optional session header."""
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        if initialize:
            self.send_header("mcp-session-id", "secret-session")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        """Keep legacy SSE sessions open until the native consumer disconnects."""
        self.server.calls.append(
            ("GET", self.path, None, self.headers.get("authorization"))
        )
        if not self.path.startswith("/sse"):
            self.reply(405)
            return
        channel_id = f"secret-session-{len(self.server.channels)}"
        channel = self.server.channels[channel_id] = queue.Queue()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        endpoint = f"/messages?session_id={channel_id}"
        if "tool_error=true" in self.path:
            endpoint += "&tool_error=true"
        self.wfile.write(f"event: endpoint\ndata: {endpoint}\n\n".encode())
        self.wfile.flush()
        while not self.server.stopping.is_set():
            try:
                message = channel.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.wfile.write(
                    f"event: message\ndata: {json.dumps(message)}\n\n".encode()
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
        self.close_connection = True

    def do_DELETE(self):
        """Acknowledge native Streamable HTTP session termination."""
        self.server.calls.append(
            ("DELETE", self.path, None, self.headers.get("authorization"))
        )
        self.reply(204)

    def do_POST(self):
        """Answer native discovery, tool and resource JSON-RPC requests."""
        message = json.loads(self.rfile.read(int(self.headers["content-length"])))
        method = message["method"]
        self.server.calls.append(
            ("POST", self.path, method, self.headers.get("authorization"))
        )
        if "http_error=true" in self.path:
            self.reply(503)
            return
        if "id" not in message:
            self.reply(202)
            return
        result = {
            "initialize": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "test", "version": "1"},
            },
            "tools/list": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo a test value",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    }
                ]
            },
            "tools/call": {"content": [{"type": "text", "text": "secret-result"}]},
            "resources/read": {
                "contents": [
                    {
                        "uri": "ui://test/private",
                        "mimeType": "text/html",
                        "text": "secret-html",
                    }
                ]
            },
        }[method]
        if method == "tools/call" and "tool_error=true" in self.path:
            result["isError"] = True
        response = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        if self.path.startswith("/messages"):
            channel = parse_qs(urlsplit(self.path).query)["session_id"][0]
            self.server.channels[channel].put(response)
            self.reply(202)
        else:
            self.reply(200, response, initialize=method == "initialize")


@pytest.fixture()
def local_mcp_server():
    """Serve Streamable HTTP and legacy SSE on a temporary loopback socket."""
    server = MCPTestServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.calls
    finally:
        server.stopping.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["streamable_http", "sse"])
@pytest.mark.parametrize("rpc_error", [False, True])
async def test_mcp_discovery_and_native_tools_audit_every_http_attempt(
    audit_records, local_mcp_server, transport, rpc_error
):
    """Catch unaudited discovery or tool sessions, session-ID leaks and RPC-only counting."""
    url, calls = local_mcp_server
    endpoint = "/sse?token=secret" if transport == "sse" else "/mcp?token=secret"
    if rpc_error:
        endpoint += "&tool_error=true"
    connections = {
        "test-server": {
            "transport": transport,
            "url": url + endpoint,
            "headers": {"authorization": "Bearer secret"},
        }
    }
    tools = await gather_mcp_tools(connections)
    assert [tool.name for tool in tools] == ["echo"]
    result = await tools[0].ainvoke(
        {
            "type": "tool_call",
            "id": "secret-tool-call",
            "name": "echo",
            "args": {"value": "secret-argument"},
        }
    )
    assert isinstance(result, ToolMessage)
    assert result.status == ("error" if rpc_error else "success")
    assert len(result.content) == 1
    assert result.content[0]["type"] == "text"
    assert result.content[0]["text"] == "secret-result"
    data = wire_records(audit_records)
    assert len(data) == 2 * len(calls)
    assert len(data) > 4
    started = [item for item in data if item["event"] == "outbound.request.started"]
    completed = [item for item in data if item["event"] == "outbound.request.finished"]
    assert len(started) == len(completed) == len(calls)
    assert {item["attempt_id"] for item in started} == {
        item["attempt_id"] for item in completed
    }
    assert len({item["attempt_id"] for item in started}) == len(calls)
    assert {item["request.kind"] for item in data} == {"mcp"}
    tool_attempts = [item for item in started if "gen_ai.tool.name" in item]
    assert {item["gen_ai.tool.name"] for item in tool_attempts} == {"echo"}
    assert len(tool_attempts) < len(started)
    assert {item["server.address"] for item in data} == {"127.0.0.1"}
    assert all(auth == "Bearer secret" for _, _, _, auth in calls)
    assert "secret" not in json.dumps(data)
    assert "httpx_client_factory" not in connections["test-server"]
    assert all("outcome" in item for item in completed)
    assert {item["http.response.status_code"] for item in completed} <= {
        200,
        202,
        204,
        405,
    }
    allowed = {
        "schema_version",
        "event",
        "timestamp",
        "service.name",
        "attempt_id",
        "request.kind",
        "server.address",
        "server.port",
        "network.peer.address",
        "network.transport",
        "http.request.method",
        "trace_id",
        "span_id",
        "gen_ai.conversation.id",
        "gen_ai.tool.name",
        "outcome",
        "error.type",
        "http.response.status_code",
        "duration_ms",
        "request.body.size",
    }
    assert all(set(item) <= allowed for item in data)
    assert not support._attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["tool", "resource"])
@pytest.mark.parametrize("preflight_denied", [False, True])
async def test_mcp_apps_native_requests_share_the_audit_lifecycle(
    audit_records, local_mcp_server, monkeypatch, operation, preflight_denied
):
    """Catch bypassing Apps audit or emitting HTTP attempts for credential preflight failures."""
    config.ols_config.authentication_config.module = "k8s"
    apps = importlib.import_module("ols.app.endpoints.mcp_apps")
    url, calls = local_mcp_server
    server = MCPServerConfig(name="test-server", url=url + "/mcp?token=secret")
    server._resolved_headers = {
        "authorization": (
            constants.MCP_CLIENT_PLACEHOLDER if preflight_denied else "Bearer secret"
        )
    }
    monkeypatch.setattr(config.config, "mcp_servers", MCPServers(servers=[server]))
    monkeypatch.setitem(config.__dict__, "mcp_servers_dict", {"test-server": server})
    auth = ("secret-user", "secret-name", False, "secret-token")
    if preflight_denied:
        with pytest.raises(HTTPException) as caught:
            if operation == "tool":
                await apps.call_mcp_app_tool(
                    MCPAppToolCallRequest(server_name="test-server", tool_name="echo"),
                    auth=auth,
                )
            else:
                await apps.get_mcp_app_resource(
                    MCPAppResourceRequest(
                        server_name="test-server", resource_uri="ui://test/private"
                    ),
                    auth=auth,
                )
        assert caught.value.status_code == 401
        assert not calls
        assert records(audit_records) == []
        return
    if operation == "tool":
        result = await apps.call_mcp_app_tool(
            MCPAppToolCallRequest(
                server_name="test-server",
                tool_name="echo",
                arguments={"value": "secret"},
            ),
            auth=auth,
        )
        assert result.content == [{"type": "text", "text": "secret-result"}]
    else:
        result = await apps.get_mcp_app_resource(
            MCPAppResourceRequest(
                server_name="test-server", resource_uri="ui://test/private"
            ),
            auth=auth,
        )
        assert result.content == "secret-html"
    data = wire_records(audit_records)
    assert len(data) == 2 * len(calls)
    assert len(calls) >= 4
    assert {item["request.kind"] for item in data} == {"mcp"}
    tool_attempts = [item for item in data if "gen_ai.tool.name" in item]
    if operation == "tool":
        assert {item["gen_ai.tool.name"] for item in tool_attempts} == {"echo"}
    else:
        assert not tool_attempts
    assert {item["event"] for item in data} == {
        "outbound.request.started",
        "outbound.request.finished",
    }
    assert len({item["attempt_id"] for item in data}) == len(calls)
    assert all(auth == "Bearer secret" for _, _, _, auth in calls)
    assert "secret" not in json.dumps(data)
    assert not support._attempts


@pytest.mark.asyncio
async def test_mcp_parallel_tools_preserve_server_trace_and_outer_llm_identity(
    audit_records, local_mcp_server, otel_setup
):
    """Catch shared mutable MCP identity or task scopes leaking into the owning LLM call."""
    url, calls = local_mcp_server
    tools = await gather_mcp_tools(
        {
            name: {"transport": "streamable_http", "url": url + "/mcp"}
            for name in ("first", "second")
        }
    )
    assert len(tools) == 2
    audit_records.clear()
    calls.clear()
    expected = {}
    _, tracer = otel_setup

    async def invoke(tool):
        name = tool.metadata["mcp_server"]
        with tracer.start_as_current_span(name) as span:
            context = span.get_span_context()
            expected[name] = (
                format(context.trace_id, "032x"),
                format(context.span_id, "016x"),
            )
            result = await tool.ainvoke({"value": "secret"})
            assert result[0]["text"] == "secret-result"

    with dispatch():
        await asyncio.gather(*(invoke(tool) for tool in tools))
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(204))
        ) as client:
            support.instrument_httpx_client(client)
            assert (
                await client.post("https://model.invalid/chat/completions")
            ).status_code == 204
    data = wire_records(audit_records)
    assert len(data) == 2 * len(calls) + 2
    assert [item["event"] for item in data[-2:]] == [
        "outbound.request.started",
        "outbound.request.finished",
    ]
    assert {item["request.kind"] for item in data[-2:]} == {"llm"}
    mcp = data[:-2]
    assert {item["request.kind"] for item in mcp} == {"mcp"}
    for trace_context in expected.values():
        own = [
            item for item in mcp if (item["trace_id"], item["span_id"]) == trace_context
        ]
        assert own
        assert {item["gen_ai.tool.name"] for item in own} == {"echo"}
    assert "secret" not in json.dumps(mcp)
    assert not support._attempts


@pytest.mark.asyncio
async def test_mcp_scope_and_trace_snapshot_do_not_leak_to_later_connections(
    audit_records, otel_setup
):
    """Catch stale operation context and trace IDs read from the consumer instead of the owner."""
    _, tracer = otel_setup

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(204)),
            trust_env=False,
        )

    create = support.mcp_client_factory(factory)
    with tracer.start_as_current_span("owner") as owner, support.mcp_operation():
        first = create()
    second = create()
    assert records(audit_records) == []
    with tracer.start_as_current_span("unrelated"):
        async with first, second:
            assert (await first.get("https://mcp.invalid/mcp")).status_code == 204
            assert (await second.get("https://mcp.invalid/mcp")).status_code == 204
    first_start, first_end, second_start, second_end = wire_records(audit_records)
    assert first_start["event"] == second_start["event"] == "outbound.request.started"
    assert first_end["event"] == second_end["event"] == "outbound.request.finished"
    assert first_start["attempt_id"] == first_end["attempt_id"]
    assert second_start["attempt_id"] == second_end["attempt_id"]
    assert (
        first_start["trace_id"]
        == first_end["trace_id"]
        == format(owner.get_span_context().trace_id, "032x")
    )
    assert (
        first_start["span_id"]
        == first_end["span_id"]
        == format(owner.get_span_context().span_id, "016x")
    )
    assert "trace_id" not in second_start
    assert "span_id" not in second_end
    assert all(
        item["request.kind"] == "mcp"
        for item in (first_start, first_end, second_start, second_end)
    )
    assert all(
        "gen_ai.tool.name" not in item
        for item in (first_start, first_end, second_start, second_end)
    )
    assert not support._attempts


@pytest.mark.asyncio
async def test_mcp_native_http_failure_keeps_other_servers_available(
    audit_records, local_mcp_server
):
    """Catch lost failure completions or changing native per-server failure isolation."""
    url, calls = local_mcp_server
    tools = await gather_mcp_tools(
        {
            "unavailable": {
                "transport": "streamable_http",
                "url": url + "/mcp?http_error=true",
            },
            "healthy": {"transport": "streamable_http", "url": url + "/mcp"},
        }
    )
    assert [tool.name for tool in tools] == ["echo"]
    assert tools[0].metadata["mcp_server"] == "healthy"
    data = wire_records(audit_records)
    assert len(data) == len(calls) * 2
    started, completed = next(
        (start, finish)
        for start, finish in zip(data[::2], data[1::2], strict=True)
        if finish.get("http.response.status_code") == 503
    )
    assert started["event"] == "outbound.request.started"
    assert completed["event"] == "outbound.request.finished"
    assert completed["attempt_id"] == started["attempt_id"]
    assert completed["http.response.status_code"] == 503
    assert completed["outcome"] == "http_error"
    assert not support._attempts


def test_auth_sandbox_scope_retains_retry_identity_and_excludes_model_fields(
    audit_records,
):
    """Catch token attempts mislabelled as inference or nested refresh scopes splitting identity."""
    metadata = "google"
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        support.instrument_httpx_client(client)
        with dispatch(), support.auth_request(metadata):
            client.post("https://token.invalid/private?secret=query", content=b"secret")
            with support.auth_request(metadata):
                client.post(
                    "https://token.invalid/private?secret=query", content=b"secret"
                )
        assert client.post("https://token.invalid/outside").status_code == 200
    data = wire_records(audit_records)
    assert [item["event"] for item in data] == [
        "outbound.request.started",
        "outbound.request.finished",
        "outbound.request.started",
        "outbound.request.finished",
    ]
    assert {item["request.kind"] for item in data} == {"credential"}
    assert len({item["attempt_id"] for item in data}) == 2
    assert "secret" not in json.dumps(data)
    assert not support._attempts


def test_auth_google_native_refresh_retries_and_cached_tokens(
    audit_records, local_http_server
):
    """Catch missed refresh, collapsed retries, forced refresh and credential leaks."""
    from google.auth.transport.requests import Request

    from ols.src.llms.providers.utils import load_vertex_credentials

    server, url, calls = local_http_server
    server.respond = lambda path, data: (
        (503, {"error": "temporarily_unavailable"})
        if len(calls) == 1
        else (
            200,
            {
                "access_token": "secret-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )
    )
    credentials = load_vertex_credentials(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "secret-client",
                "client_secret": "secret-client-key",
                "refresh_token": "secret-refresh",
            }
        )
    )
    credentials._token_uri = url + "/token?secret=query"
    request = Request()
    try:
        headers = {}
        credentials.before_request(request, "POST", "https://model.invalid", headers)
        assert headers["authorization"] == "Bearer secret-token"
        credentials.before_request(request, "POST", "https://model.invalid", {})
        assert len(calls) == 2
        data = wire_records(audit_records)
        assert [item["event"] for item in data] == [
            "outbound.request.started",
            "outbound.request.finished",
            "outbound.request.started",
            "outbound.request.finished",
        ]
        assert {item["request.kind"] for item in data} == {"credential"}
        assert len({item["attempt_id"] for item in data}) == 2
        assert data[1]["http.response.status_code"] == 503
        assert data[3]["http.response.status_code"] == 200
        assert "secret" not in json.dumps(data)
        assert not support._attempts
    finally:
        request.session.close()


@pytest.mark.parametrize("retry", [False, True])
def test_auth_azure_native_provider_keeps_transport_and_token_cache(
    audit_records, local_http_server, monkeypatch, retry
):
    """Catch missed acquisition, replaced Azure transport, cache misses and secret leaks."""
    from azure.core.pipeline.transport import RequestsTransport
    from azure.identity import ClientSecretCredential

    from ols.app.models.config import AzureOpenAIConfig, ProviderConfig
    from ols.src.llms.providers import azure_openai

    server, url, calls = local_http_server
    authority = "https://login.invalid/secret-tenant"

    class LocalTransport(RequestsTransport):
        def send(self, request, **kwargs):
            parsed = urlsplit(request.url)
            request.url = url + parsed.path
            return super().send(request, **kwargs)

    def respond(path, data):
        if path.endswith("openid-configuration"):
            return 200, {
                "authorization_endpoint": authority + "/oauth2/v2.0/authorize",
                "token_endpoint": authority + "/oauth2/v2.0/token",
                "issuer": authority + "/v2.0",
            }
        token_calls = [item for item in calls if item[0].endswith("/token")]
        if retry and len(token_calls) == 1:
            return 503, {"error": "temporarily_unavailable"}
        return 200, {
            "access_token": "secret-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

    server.respond = respond
    transport = LocalTransport()
    credential = ClientSecretCredential(
        "secret-tenant",
        "secret-client",
        "secret-key",
        transport=transport,
        authority="https://login.invalid",
        disable_instance_discovery=True,
        retry_backoff_factor=0,
    )
    monkeypatch.setattr(
        azure_openai, "ClientSecretCredential", lambda *args, **kwargs: credential
    )
    monkeypatch.setattr(azure_openai, "TOKEN_CACHES", {})
    provider = azure_openai.AzureOpenAI(
        model="gpt-test",
        params={},
        provider_config=ProviderConfig(
            {
                "name": "azure-test",
                "type": "azure_openai",
                "url": "https://model.invalid",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [{"name": "gpt-test"}],
            }
        ),
    )
    azure = AzureOpenAIConfig(
        url="https://model.invalid",
        deployment_name="test-deployment",
        tenant_id="secret-tenant",
        client_id="secret-client",
        client_secret="secret-key",
    )
    try:
        assert provider.resolve_access_token(azure) == "secret-token"
        count = len(calls)
        assert provider.resolve_access_token(azure) == "secret-token"
        assert len(calls) == count == (3 if retry else 2)
        assert credential._client._pipeline._transport is transport
        data = wire_records(audit_records)
        assert len(data) == count * 2
        assert {item["event"] for item in data} == {
            "outbound.request.started",
            "outbound.request.finished",
        }
        assert {item["request.kind"] for item in data} == {"credential"}
        assert len({item["attempt_id"] for item in data}) == count
        assert "secret" not in json.dumps(data)
        assert not support._attempts
    finally:
        credential.close()


@pytest.mark.parametrize("retry", [False, True])
def test_auth_aws_native_assume_role_preserves_retry_signing_and_session(
    audit_records, local_http_server, monkeypatch, retry
):
    """Catch STS exchanges missing audits, collapsed retries or changed role credential behavior."""
    import boto3
    from botocore.config import Config

    from ols.app.models.config import ProviderConfig
    from ols.src.llms.providers import bedrock

    server, url, calls = local_http_server
    payload = b"""<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
      <AssumeRoleResult><Credentials><AccessKeyId>secret-assumed-key</AccessKeyId>
      <SecretAccessKey>secret-assumed-secret</SecretAccessKey><SessionToken>secret-session-token</SessionToken>
      <Expiration>2035-01-01T00:00:00Z</Expiration></Credentials>
      <AssumedRoleUser><AssumedRoleId>secret-role-id</AssumedRoleId>
      <Arn>arn:aws:sts::123456789012:assumed-role/secret-role/ols-bedrock</Arn></AssumedRoleUser>
      </AssumeRoleResult><ResponseMetadata><RequestId>secret-request</RequestId></ResponseMetadata>
      </AssumeRoleResponse>"""
    server.respond = lambda path, data: (
        (
            503,
            b"<ErrorResponse><Error><Code>ServiceUnavailable</Code></Error></ErrorResponse>",
        )
        if retry and len(calls) == 1
        else (200, payload)
    )
    original_session = boto3.Session
    source = original_session(
        aws_access_key_id="source-key",
        aws_secret_access_key="secret-source",
        region_name="us-east-1",
    )
    sts = source.client(
        "sts", endpoint_url=url, config=Config(retries={"max_attempts": 1})
    )

    def client(service, **kwargs):
        assert service == "sts"
        return sts

    def session(**kwargs):
        if kwargs["aws_access_key_id"] == "source-key":
            return source
        return original_session(**kwargs)

    monkeypatch.setattr(source, "client", client)
    monkeypatch.setattr(bedrock.boto3, "Session", session)
    provider = bedrock.Bedrock(
        model="anthropic.claude-test",
        params={},
        provider_config=ProviderConfig(
            {
                "name": "aws-test",
                "type": "bedrock",
                "url": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "credentials_path": "tests/config/secret/apitoken",
                "models": [{"name": "anthropic.claude-test"}],
            }
        ),
    )
    try:
        assumed = provider._build_boto3_session(
            "us-east-1",
            (
                "source-key",
                "secret-source",
                "arn:aws:iam::123456789012:role/secret-role",
            ),
        )
        frozen = assumed.get_credentials().get_frozen_credentials()
        assert frozen.access_key == "secret-assumed-key"
        assert frozen.secret_key == "secret-assumed-secret"  # noqa: S105
        assert frozen.token == "secret-session-token"  # noqa: S105
        assert len(calls) == (2 if retry else 1)
        assert b"RoleSessionName=ols-bedrock" in calls[-1][1]
        data = wire_records(audit_records)
        assert len(data) == 2 * len(calls)
        assert {item["request.kind"] for item in data} == {"credential"}
        assert {item["event"] for item in data} == {
            "outbound.request.started",
            "outbound.request.finished",
        }
        assert len({item["attempt_id"] for item in data}) == len(calls)
        assert "secret" not in json.dumps(data)
        assert not support._attempts
    finally:
        sts.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_auth_google_failure_preserves_native_exception_and_audit_gate(
    audit_records, local_http_server, enabled
):
    """Catch swallowed auth errors, raw exception/token leaks and ignored audit gates."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request

    from ols.src.llms.providers.utils import load_vertex_credentials

    server, url, calls = local_http_server
    server.respond = lambda path, data: (
        400,
        {"error": "invalid_grant", "error_description": "secret"},
    )
    credentials = load_vertex_credentials(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "secret-client",
                "client_secret": "secret-key",
                "refresh_token": "secret-refresh",
            }
        )
    )
    credentials._token_uri = url + "/token?secret=query"
    if not enabled:
        config.ols_config.audit.logging = AuditLoggingMode.DISABLED
    request = Request()
    try:
        with pytest.raises(RefreshError):
            credentials.refresh(request)
        assert len(calls) == 1
        data = records(audit_records)
        if enabled:
            started, completed = data
            assert completed["event"] == "auth.request.completed"
            assert started["attempt_id"] == completed["attempt_id"]
            assert completed["failure_category"] == "http_error"
            assert completed["http_status"] == 400
            assert "secret" not in json.dumps(data)
        else:
            assert not data
        assert not support._attempts
    finally:
        request.session.close()


@pytest.mark.asyncio
async def test_auth_google_native_async_sdk_uses_refresh_scope_in_worker(
    audit_records, local_http_server
):
    """Catch lost auth context in Google's native async refresh worker or cache regressions."""
    from google.genai._api_client import async_get_token_from_credentials

    from ols.src.llms.providers.utils import load_vertex_credentials

    server, url, calls = local_http_server
    server.respond = lambda path, data: (
        200,
        {"access_token": "secret-token", "expires_in": 3600, "token_type": "Bearer"},
    )
    credentials = load_vertex_credentials(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "secret-client",
                "client_secret": "secret-key",
                "refresh_token": "secret-refresh",
            }
        )
    )
    credentials._token_uri = url + "/token"
    from google import genai
    from google.genai import types

    native = genai.Client(
        vertexai=True,
        project="secret-project",
        location="global",
        credentials=credentials,
        http_options=types.HttpOptions(base_url=url),
    )
    try:
        assert (
            await async_get_token_from_credentials(native._api_client, credentials)
            == "secret-token"
        )
        assert (
            await async_get_token_from_credentials(native._api_client, credentials)
            == "secret-token"
        )
        assert len(calls) == 1
        started, completed = records(audit_records)
        assert started["event"] == "auth.request.started"
        assert completed["event"] == "auth.request.completed"
        assert completed["outcome"] == "success"
        assert "secret" not in json.dumps([started, completed])
        assert not support._attempts
    finally:
        native.close()
        await native.aio.aclose()


def test_auth_attempt_keeps_owner_trace_and_completes_after_disable(
    audit_records, otel_setup
):
    """Catch scope leakage, trace-based matching or lost completion on disable."""
    _, tracer = otel_setup
    with tracer.start_as_current_span("owner") as span, dispatch():
        with support.auth_request("aws"):
            attempt = support.start_attempt(
                support.Destination("https", "sts.invalid", 443)
            )
        model_attempt = support.start_attempt(
            support.Destination("https", "model.invalid", 443)
        )
        model_attempt.response_headers(204)
        model_attempt.finish(response_complete=True)
    config.ols_config.audit.logging = AuditLoggingMode.DISABLED
    with tracer.start_as_current_span("unrelated"):
        attempt.response_headers(200)
        attempt.finish(response_complete=True)
        attempt.finish(response_complete=False)
    started, model_start, model_end, completed = wire_records(audit_records)
    assert completed["event"] == "outbound.request.finished"
    assert completed["attempt_id"] == started["attempt_id"]
    assert completed["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert completed["span_id"] == format(span.get_span_context().span_id, "016x")
    assert model_start["event"] == "outbound.request.started"
    assert model_start["request.kind"] == model_end["request.kind"] == "llm"
    assert completed["request.kind"] == started["request.kind"] == "credential"
    assert not support._attempts


@pytest.mark.parametrize("local_signing", [False, True])
def test_auth_google_service_account_native_grant_or_local_signing(
    audit_records, local_http_server, local_signing
):
    """Catch missed JWT grants, synthetic signing attempts or signing-policy changes."""
    from google.auth.transport.requests import Request

    from ols.src.llms.providers.utils import load_vertex_credentials
    from tests.unit.llms.providers.utils import generate_service_account_json_string

    server, url, calls = local_http_server
    server.respond = lambda path, data: (
        200,
        {"access_token": "secret-token", "expires_in": 3600, "token_type": "Bearer"},
    )
    payload = json.loads(
        generate_service_account_json_string(
            project_id="secret-project",
            client_email="secret@invalid",
            client_id="secret-client",
        )
    )
    payload["token_uri"] = url + "/token?secret=query"
    credentials = load_vertex_credentials(json.dumps(payload))
    credentials._always_use_jwt_access = local_signing
    request = Request()
    try:
        headers = {}
        credentials.before_request(request, "POST", "https://model.invalid", headers)
        assert headers["authorization"].startswith("Bearer ")
        if local_signing:
            assert not calls
            assert not records(audit_records)
        else:
            assert len(calls) == 1
            assert b"assertion=" in calls[0][1]
            started, completed = records(audit_records)
            assert started["event"] == "auth.request.started"
            assert completed["event"] == "auth.request.completed"
            assert completed["outcome"] == "success"
            assert "secret" not in json.dumps([started, completed])
        assert not support._attempts
    finally:
        request.session.close()


def test_auth_google_native_proxy_retains_routing_and_redacts_authorities(
    audit_records, local_http_server
):
    """Catch replaced proxy policy, missing proxy-manager audit or proxy/credential leakage."""
    import requests
    from google.auth.transport.requests import Request

    from ols.src.llms.providers.utils import load_vertex_credentials

    server, url, calls = local_http_server
    server.respond = lambda path, data: (
        200,
        {"access_token": "secret-token", "expires_in": 3600, "token_type": "Bearer"},
    )
    session = requests.Session()
    session.trust_env = False
    session.proxies["http"] = url.replace("://", "://secret-user:secret-pass@")
    credentials = load_vertex_credentials(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "secret-client",
                "client_secret": "secret-key",
                "refresh_token": "secret-refresh",
            }
        )
    )
    credentials._token_uri = "http://token.invalid/token?secret=query"  # noqa: S105
    request = Request(session=session)
    try:
        credentials.refresh(request)
        assert request.session is session
        assert len(calls) == 1
        assert calls[0][0] == "http://token.invalid/token?secret=query"
        started, completed = records(audit_records)
        assert started["destination_host"] == "127.0.0.1"
        assert started["destination_port"] == server.server_port
        assert started["target_host"] == "token.invalid"
        assert completed["outcome"] == "success"
        assert "secret" not in json.dumps([started, completed])
        assert not support._attempts
    finally:
        session.close()
