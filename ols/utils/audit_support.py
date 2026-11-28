"""Audit owned native LLM and MCP HTTP attempts without replacing transport behavior."""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import json
import logging
import re
import socket
import ssl
import sys
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from threading import Lock, RLock
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpcore
import httpx
from aiohttp.streams import AsyncStreamIterator, ChunkTupleAsyncStreamIterator
from langchain_core.language_models import BaseChatModel
from mcp.shared._httpx_utils import McpHttpClientFactory, create_mcp_http_client
from opentelemetry import trace
from urllib3 import exceptions as urllib3_errors
from urllib3.util import parse_url

from ols import config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from types import TracebackType

try:
    httpx2: Any = importlib.import_module("httpx2")
except ImportError:
    httpx2 = None  # pylint: disable=invalid-name

_timeout_errors: tuple[type[BaseException], ...] = (
    TimeoutError,
    httpx.TimeoutException,
    httpcore.TimeoutException,
)
_connect_errors: tuple[type[BaseException], ...] = (
    ConnectionError,
    httpx.ConnectError,
    httpcore.ConnectError,
)
if httpx2 is not None:
    _timeout_errors += (httpx2.TimeoutException,)
    _connect_errors += (httpx2.ConnectError,)

AuthProvider = Literal["azure", "google", "aws", "ibm"]


@dataclass(frozen=True)
class Destination:
    """Describe only an HTTP authority, never a complete URL."""

    scheme: str
    host: str
    port: int


@dataclass(frozen=True)
class _Call:
    kind: Literal["llm", "mcp", "auth"]
    span_context: trace.SpanContext
    mcp_tool_name: str | None = None
    conversation_id: str | None = None
    auth_provider: AuthProvider | None = None


_LOGGER = logging.getLogger("ols.connection_audit")
_audit_logger_lock = Lock()
_attempts: dict[str, RequestAttempt] = {}
_attempts_lock = Lock()
_call: ContextVar[_Call | None] = ContextVar("ols_llm_audit_call", default=None)
_mcp_operation: ContextVar[str | None] = ContextVar("ols_mcp_operation", default=None)
_watsonx_auth_initialization: ContextVar[bool] = ContextVar(
    "ols_watsonx_auth_initialization", default=False
)


class _JSONHandler(logging.Handler):
    """Write audit JSON as unprefixed lines to the active stdout stream."""

    def emit(self, record: logging.LogRecord) -> None:
        with suppress(Exception):
            sys.stdout.write(record.getMessage() + "\n")
            sys.stdout.flush()


def configure_connection_audit() -> None:
    """Install the audit stdout handler without changing application logging."""
    with _audit_logger_lock:
        _LOGGER.propagate = False
        _LOGGER.disabled = False
        _LOGGER.setLevel(logging.INFO)
        if not any(isinstance(handler, _JSONHandler) for handler in _LOGGER.handlers):
            _LOGGER.addHandler(_JSONHandler())


def llm_call_scope() -> Callable[[], AbstractContextManager[None]]:
    """Snapshot trace context and activate it only while native execution advances."""
    current = _Call(
        "llm",
        trace.get_current_span().get_span_context(),
        conversation_id=_current_conversation_id(),
    )

    @contextmanager
    def activate() -> Iterator[None]:
        token = _call.set(current)
        try:
            yield
        finally:
            _call.reset(token)

    return activate


def _public_authority(values: dict[str, Any]) -> dict[str, Any]:
    """Return only a validated destination authority."""
    host = values.get("target_host")
    if host is None:
        host = values.get("destination_host")
    port = values.get("target_port")
    if port is None:
        port = values.get("destination_port")
    fields: dict[str, Any] = {}
    if isinstance(host, str):
        candidate = host.rstrip(".").lower()
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            labels = candidate.split(".")
            if len(candidate) <= 253 and all(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in labels
            ):
                fields["server.address"] = candidate
        else:
            fields["server.address"] = address.compressed
    if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
        fields["server.port"] = port
    return fields


def _validated_conversation_id(value: Any) -> str | None:
    """Normalize a server-owned conversation identifier only when it is a UUID."""
    if not isinstance(value, str):
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def _current_conversation_id() -> str | None:
    """Read only the validated conversation UUID from the active application span."""
    with suppress(Exception):
        attributes = getattr(trace.get_current_span(), "attributes", {})
        return _validated_conversation_id(attributes.get("gen_ai.conversation.id"))
    return None


def _valid_nonzero_hex(value: str, width: int) -> bool:
    """Check a bounded lowercase hexadecimal identifier with at least one nonzero digit."""
    return (
        len(value) == width
        and re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is not None
        and value != "0" * width
    )


def _public_trace_context(values: dict[str, Any]) -> dict[str, str]:
    """Return only a complete, well-formed OTel trace/span pair."""
    trace_id, span_id = values.get("trace_id"), values.get("span_id")
    if (
        isinstance(trace_id, str)
        and isinstance(span_id, str)
        and _valid_nonzero_hex(trace_id, 32)
        and _valid_nonzero_hex(span_id, 16)
    ):
        return {"trace_id": trace_id, "span_id": span_id}
    return {}


def _public_conversation(values: dict[str, Any]) -> dict[str, str]:
    """Include only a validated conversation UUID, never other span attributes."""
    conversation_id = _validated_conversation_id(values.get("conversation_id"))
    return {"gen_ai.conversation.id": conversation_id} if conversation_id else {}


def _public_outcome(values: dict[str, Any]) -> dict[str, Any]:
    """Map private lifecycle details to the shared outcome vocabulary."""
    status = values.get("http_status")
    status_code = (
        status
        if isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
        else None
    )
    error_types = {
        "dns": "dns_error",
        "connect": "connect_error",
        "tls": "tls_error",
        "tls_verification": "tls_error",
        "timeout": "timeout",
        "stream_error": "protocol_error",
        "http_error": "http_error",
        "cancelled": "cancelled",
        "transport": "unknown_error",
        "incomplete_response": "unknown_error",
    }
    error_type = error_types.get(values.get("failure_category"))
    if status_code is not None and status_code >= 400:
        error_type = "http_error"
    if values.get("outcome") == "cancelled":
        outcome, error_type = "cancelled", "cancelled"
    elif error_type == "http_error":
        outcome = "http_error"
    elif values.get("outcome") == "success":
        outcome = "success"
    else:
        outcome, error_type = "transport_error", error_type or "unknown_error"
    fields: dict[str, Any] = {
        "duration_ms": max(0.0, float(values.get("duration_ms", 0.0))),
        "outcome": outcome,
    }
    if status_code is not None:
        fields["http.response.status_code"] = status_code
    if error_type is not None:
        fields["error.type"] = error_type
    return fields


def _public_record(
    values: dict[str, Any], *, family: Literal["llm", "mcp", "auth"]
) -> dict[str, Any]:
    """Project private transport bookkeeping into the shared public schema."""
    kind = {"llm": "llm", "mcp": "mcp", "auth": "credential"}[family]
    record: dict[str, Any] = {
        "schema_version": 1,
        "event": (
            "outbound.request.started"
            if values["event"].endswith(".started")
            else "outbound.request.finished"
        ),
        "timestamp": values["timestamp"],
        "service.name": "lightspeed-service",
        "attempt_id": values["attempt_id"],
        "request.kind": kind,
        "network.transport": "tcp",
        **_public_authority(values),
        **_public_trace_context(values),
        **_public_conversation(values),
    }
    method = values.get("request_method")
    if isinstance(method, str) and re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", method):
        record["http.request.method"] = method
    tool_name = values.get("mcp_tool_name")
    if kind == "mcp" and isinstance(tool_name, str) and len(tool_name) <= 256:
        if tool_name.isprintable():
            record["gen_ai.tool.name"] = tool_name
    if values["event"].endswith(".completed"):
        record.update(_public_outcome(values))
    return record


def _failure(error: BaseException) -> str:
    chain: list[BaseException] = []
    pending = [error]
    while pending:
        item = pending.pop()
        if any(item is seen for seen in chain):
            continue
        chain.append(item)
        pending.extend(
            nested
            for nested in (item.__cause__, item.__context__)
            if nested is not None
        )
    if any(isinstance(item, asyncio.CancelledError) for item in chain):
        return "cancelled"
    if any(isinstance(item, ssl.SSLError) for item in chain):
        return (
            "tls_verification"
            if any(isinstance(item, ssl.SSLCertVerificationError) for item in chain)
            else "tls"
        )
    if any(isinstance(item, socket.gaierror) for item in chain):
        return "dns"
    if any(isinstance(item, _timeout_errors) for item in chain):
        return "timeout"
    if any(isinstance(item, _connect_errors) for item in chain):
        return "connect"
    return "transport"


class RequestAttempt:
    """Track one HTTP attempt independently of the consuming caller's context."""

    def __init__(
        self, record: dict[str, Any], *, family: Literal["llm", "mcp", "auth"] = "llm"
    ) -> None:
        """Emit the start immediately, retaining only safe metadata."""
        self._family = family
        self._record = record
        self._started = monotonic()
        self._attempt_id = record["attempt_id"]
        self._status: int | None = None
        with _attempts_lock:
            _attempts[self._attempt_id] = self
        self._send(f"{self._family}.request.started")

    def _send(self, event: str, **fields: Any) -> None:
        with suppress(Exception):
            values = {
                **self._record,
                **fields,
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            body = _public_record(values, family=self._family)
            configure_connection_audit()
            _LOGGER.info(json.dumps(body, allow_nan=False, separators=(",", ":")))

    def response_headers(self, status: int) -> None:
        """Record status without completing a streamed response."""
        self._status = status

    def finish(
        self,
        *,
        response_complete: bool,
        error: BaseException | None = None,
        cancelled: bool = False,
    ) -> None:
        """Finish once without retaining or exposing a raw exception."""
        with _attempts_lock:
            if _attempts.get(self._attempt_id) is not self:
                return
            del _attempts[self._attempt_id]
        if isinstance(error, GeneratorExit):
            error = None
        category = _failure(error) if error is not None else None
        if category == "tls_verification":
            category = "tls"
        if category == "transport" and self._status is not None:
            category = "stream_error"
        if cancelled or category == "cancelled":
            outcome, category = "cancelled", "cancelled"
        elif category is not None:
            outcome = "failure"
        elif self._status is not None and self._status >= 400:
            outcome, category = "failure", "http_error"
        elif not response_complete or self._status is None:
            outcome, category = "failure", "incomplete_response"
        else:
            outcome = "success"
        self._send(
            f"{self._family}.request.completed",
            outcome=outcome,
            duration_ms=(monotonic() - self._started) * 1000,
            http_status=self._status,
            response_complete=response_complete,
            failure_category=category,
        )


def start_attempt(
    destination: Destination,
    *,
    target: Destination | None = None,
    method: str | None = None,
) -> RequestAttempt | None:
    """Start only an eligible scoped native HTTP attempt when audit is enabled."""
    call = _call.get()
    audit_config = config.ols_config.audit
    if call is None or audit_config is None or not audit_config.enabled:
        return None
    span = call.span_context
    family = call.kind
    record: dict[str, Any] = {
        "attempt_id": str(uuid4()),
        "trace_id": format(span.trace_id, "032x") if span.is_valid else None,
        "span_id": format(span.span_id, "016x") if span.is_valid else None,
        "destination_host": destination.host,
        "destination_port": destination.port,
        "request_method": method,
        "conversation_id": call.conversation_id,
    }
    if call.kind == "mcp" and call.mcp_tool_name is not None:
        record["mcp_tool_name"] = call.mcp_tool_name
    if target is not None:
        record.update(
            target_host=target.host,
            target_port=target.port,
        )
    return RequestAttempt(record, family=family)


_SYNC_CLIENTS = (httpx.Client,) + ((httpx2.Client,) if httpx2 is not None else ())
_ASYNC_CLIENTS = (httpx.AsyncClient,) + (
    (httpx2.AsyncClient,) if httpx2 is not None else ()
)


def destination_from_url(url: Any) -> Destination:
    """Project only native HTTP authority components."""

    def text(value: str | bytes) -> str:
        return value.decode("ascii") if isinstance(value, bytes) else value

    scheme = text(url.scheme)
    return Destination(
        scheme,
        text(url.host),
        url.port if url.port is not None else 443 if scheme == "https" else 80,
    )


def _begin(request: httpx.Request, transport: Any) -> RequestAttempt | None:
    proxy = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
    target = destination_from_url(request.url)
    return start_attempt(
        destination_from_url(proxy) if proxy is not None else target,
        target=target if proxy is not None else None,
        method=request.method,
    )


class _BodyState:
    def __init__(
        self, stream: Any, attempt: RequestAttempt, active: set[RequestAttempt]
    ) -> None:
        self.stream = stream
        self.attempt = attempt
        self.active = active
        self.failed = False
        self.closed = False

    def finish(self, complete: bool, error: BaseException | None = None) -> None:
        """Finish this body without retaining it in the client cleanup set."""
        self.failed = self.failed or error is not None
        self.active.discard(self.attempt)
        self.attempt.finish(response_complete=complete, error=error)


class _SyncBody(httpx.SyncByteStream, _BodyState):
    def __init__(
        self, stream: Any, attempt: RequestAttempt, active: set[RequestAttempt]
    ) -> None:
        _BodyState.__init__(self, stream, attempt, active)

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self.stream
        except BaseException as error:
            self.finish(False, error)
            raise
        self.finish(True)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        primary = sys.exc_info()[1]
        failed = self.failed
        try:
            self.stream.close()
        except BaseException as error:
            self.finish(False, error)
            if primary is None and not failed:
                raise
        finally:
            self.finish(False)


class _AsyncBody(httpx.AsyncByteStream, _BodyState):
    def __init__(
        self, stream: Any, attempt: RequestAttempt, active: set[RequestAttempt]
    ) -> None:
        _BodyState.__init__(self, stream, attempt, active)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.stream:
                yield chunk
        except BaseException as error:
            self.finish(False, error)
            raise
        self.finish(True)

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        primary = sys.exc_info()[1]
        failed = self.failed
        try:
            await self.stream.aclose()
        except BaseException as error:
            self.finish(False, error)
            if primary is None and not failed:
                raise
        finally:
            self.finish(False)


if httpx2 is not None:

    class _Httpx2SyncBody(_SyncBody, httpx2.SyncByteStream):
        """Preserve the selected Anthropic SDK byte-stream interface."""

    class _Httpx2AsyncBody(_AsyncBody, httpx2.AsyncByteStream):
        """Preserve the selected asynchronous Anthropic stream interface."""


def _headers(
    response: httpx.Response, attempt: RequestAttempt, active: set[RequestAttempt]
) -> bool:
    attempt.response_headers(response.status_code)
    if response.is_stream_consumed:
        attempt.finish(response_complete=True)
        active.discard(attempt)
        return False
    return True


@contextmanager
def _closing(
    active: set[RequestAttempt], primary: BaseException | None
) -> Iterator[None]:
    error = primary
    try:
        yield
    except BaseException as cleanup:
        error = primary or cleanup
        if primary is None:
            raise
    finally:
        for attempt in tuple(active):
            attempt.finish(response_complete=False, error=error)
        active.clear()


class _SyncTransport(httpx.BaseTransport):
    def __init__(
        self,
        transport: httpx.BaseTransport,
        eligible: Callable[[Any], bool] | None = None,
    ) -> None:
        self.eligible = eligible
        self.transport = transport
        self.active: set[RequestAttempt] = set()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.eligible is not None and not self.eligible(request):
            return self.transport.handle_request(request)
        attempt = _begin(request, self.transport)
        if attempt is None:
            return self.transport.handle_request(request)
        self.active.add(attempt)
        try:
            response = self.transport.handle_request(request)
        except BaseException as error:
            self.active.discard(attempt)
            attempt.finish(response_complete=False, error=error)
            raise
        if _headers(response, attempt, self.active):
            body_type = (
                _Httpx2SyncBody  # pylint: disable=possibly-used-before-assignment
                if httpx2 is not None
                and isinstance(response.stream, httpx2.SyncByteStream)
                else _SyncBody
            )
            response.stream = body_type(response.stream, attempt, self.active)
        return response

    def __enter__(self) -> _SyncTransport:
        self.transport.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with _closing(self.active, exc_value):
            self.transport.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        with _closing(self.active, sys.exc_info()[1]):
            self.transport.close()


class _AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        eligible: Callable[[Any], bool] | None = None,
    ) -> None:
        self.eligible = eligible
        self.transport = transport
        self.active: set[RequestAttempt] = set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.eligible is not None and not self.eligible(request):
            return await self.transport.handle_async_request(request)
        attempt = _begin(request, self.transport)
        if attempt is None:
            return await self.transport.handle_async_request(request)
        self.active.add(attempt)
        try:
            response = await self.transport.handle_async_request(request)
        except BaseException as error:
            self.active.discard(attempt)
            attempt.finish(response_complete=False, error=error)
            raise
        if _headers(response, attempt, self.active):
            body_type = (
                _Httpx2AsyncBody  # pylint: disable=possibly-used-before-assignment
                if httpx2 is not None
                and isinstance(response.stream, httpx2.AsyncByteStream)
                else _AsyncBody
            )
            response.stream = body_type(response.stream, attempt, self.active)
        return response

    async def __aenter__(self) -> _AsyncTransport:
        await self.transport.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with _closing(self.active, exc_value):
            await self.transport.__aexit__(exc_type, exc_value, traceback)

    async def aclose(self) -> None:
        with _closing(self.active, sys.exc_info()[1]):
            await self.transport.aclose()


def instrument_httpx_client(
    client: Any, *, eligible: Callable[[Any], bool] | None = None
) -> None:
    """Observe selected native transports while preserving mounts and ownership."""
    wrappers: dict[int, Any] = {}

    def wrap(transport: Any) -> Any:
        if transport is None:
            return transport
        if isinstance(transport, (_SyncTransport, _AsyncTransport)):
            if eligible is not None:
                transport.eligible = eligible
            return transport
        if id(transport) not in wrappers:
            wrappers[id(transport)] = (
                _AsyncTransport(transport, eligible)
                if isinstance(client, _ASYNC_CLIENTS)
                else _SyncTransport(transport, eligible)
            )
        return wrappers[id(transport)]

    with _install_lock:
        client._transport = wrap(client._transport)
        client._mounts = {
            pattern: wrap(transport) for pattern, transport in client._mounts.items()
        }


@contextmanager
def auth_request(provider: AuthProvider) -> Iterator[None]:
    """Scope native acquisition/refresh, sharing nested same-provider exchanges."""
    previous = _call.get()
    if (
        previous is not None
        and previous.kind == "auth"
        and previous.auth_provider == provider
    ):
        yield
        return
    token = _call.set(
        _Call(
            "auth",
            trace.get_current_span().get_span_context(),
            conversation_id=_current_conversation_id(),
            auth_provider=provider,
        )
    )
    try:
        yield
    finally:
        _call.reset(token)


@contextmanager
def mcp_operation(*, tool_name: str | None = None) -> Iterator[None]:
    """Scope an MCP request and, when known, its safe tool name."""
    token = _mcp_operation.set(tool_name)
    try:
        yield
    finally:
        _mcp_operation.reset(token)


def instrument_mcp_http_client(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Observe this native MCP client while excluding its authentication exchanges."""
    with _install_lock:
        if getattr(client, "_ols_mcp_audit_installed", False):
            return client
        purpose = _mcp_operation.get()
        call = _Call(
            "mcp",
            trace.get_current_span().get_span_context(),
            mcp_tool_name=purpose,
            conversation_id=_current_conversation_id(),
        )
        marker = object()
        original_send = client.send

        async def send(
            request: httpx.Request, *args: Any, **kwargs: Any
        ) -> httpx.Response:
            request.extensions["ols_mcp_audit"] = marker
            token = _call.set(call)
            try:
                return await original_send(request, *args, **kwargs)
            finally:
                _call.reset(token)

        instrument_httpx_client(
            client,
            eligible=lambda request: request.extensions.get("ols_mcp_audit") is marker,
        )
        setattr(client, "send", send)
        setattr(client, "_ols_mcp_audit_installed", True)
        return client


def mcp_client_factory(
    factory: McpHttpClientFactory | None = None,
) -> McpHttpClientFactory:
    """Decorate the native MCP client factory without replacing configuration or ownership."""
    original = factory if factory is not None else create_mcp_http_client

    def create(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        client = original(headers=headers, timeout=timeout, auth=auth)
        return instrument_mcp_http_client(client)

    return create


def _model_inference(request: Any) -> bool:
    return request.method == "POST" and request.url.path.endswith(
        (
            "/responses",
            "/chat/completions",
            "/messages",
            ":rawPredict",
            ":streamRawPredict",
        )
    )


def instrument_model_http_clients(  # noqa: C901
    model: BaseChatModel, *, use_async: bool = False
) -> None:
    """Observe native model clients, creating only the selected lazy SDK client."""
    watsonx_client = getattr(model, "watsonx_client", None)
    if watsonx_client is not None:
        instrument_watsonx_client(watsonx_client)
        return
    for value in vars(model).values():
        if isinstance(value, _SYNC_CLIENTS + _ASYNC_CLIENTS):
            instrument_httpx_client(value, eligible=_model_inference)
    native = getattr(model, "client", None)
    if native is not None and hasattr(native, "_api_client"):
        instrument_google_client(native)
        return
    endpoint = getattr(native, "_endpoint", None)
    if endpoint is not None:
        session = endpoint.http_session
        instrument_urllib3_manager(session._manager)
        for manager in session._proxy_managers.values():
            instrument_urllib3_manager(manager)
        if not getattr(session, "_ols_audit_installed", False):
            original = session._get_proxy_manager

            def get_proxy_manager(*args: Any, **kwargs: Any) -> Any:
                manager = original(*args, **kwargs)
                instrument_urllib3_manager(manager)
                return manager

            session._get_proxy_manager = get_proxy_manager
            session._ols_audit_installed = True
        return
    root = getattr(model, "root_async_client" if use_async else "root_client", None)
    if root is None:
        root = getattr(model, "async_client" if use_async else "client", None)
    if root is None and "anthropic_api_key" in type(model).model_fields:
        root = getattr(model, "_async_client" if use_async else "_client", None)
    client = getattr(root, "_client", None)
    if isinstance(client, _SYNC_CLIENTS + _ASYNC_CLIENTS):
        instrument_httpx_client(client, eligible=_model_inference)


def audited_model_class(model_class: Any) -> type[BaseChatModel]:
    """Scope native model execution without holding context across caller yields."""

    def scope(model: BaseChatModel, use_async: bool = False) -> Any:
        instrument_model_http_clients(model, use_async=use_async)
        return llm_call_scope()

    class AuditedModel(model_class):
        """Delegate model execution under metadata-only dispatch scopes."""

        def _generate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            with scope(self)():
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

        async def _agenerate(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            with scope(self, True)():
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )

        def _stream(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Iterator[Any]:
            activate = scope(self)
            iterator = super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            try:
                while True:
                    with activate():
                        try:
                            chunk = next(iterator)
                        except StopIteration:
                            return
                    yield chunk
            finally:
                with activate():
                    iterator.close()

        async def _astream(
            self,
            messages: list[Any],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> AsyncIterator[Any]:
            activate = scope(self, True)
            iterator = super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            try:
                while True:
                    with activate():
                        try:
                            chunk = await anext(iterator)
                        except StopAsyncIteration:
                            return
                    yield chunk
            finally:
                with activate():
                    await iterator.aclose()

    # LangChain derives public names, serialization IDs and cache keys from these attributes.
    AuditedModel.__name__ = model_class.__name__
    AuditedModel.__qualname__ = model_class.__qualname__
    AuditedModel.__module__ = model_class.__module__
    return cast("type[BaseChatModel]", AuditedModel)


def audit_model(model: Any) -> Any:
    """Retain the constructed model and observe its native dispatch methods."""
    if isinstance(model, BaseChatModel):
        object.__setattr__(model, "__class__", audited_model_class(type(model)))
    return model


def _google_inference(request: Any) -> bool:
    return request.method == "POST" and request.url.path.endswith(
        (":generateContent", ":streamGenerateContent")
    )


class _ObservedContent:
    def __init__(
        self, content: Any, attempt: RequestAttempt, active: set[RequestAttempt]
    ) -> None:
        self.content = content
        self.attempt = attempt
        self.active = active
        self.error: BaseException | None = None

    def finish(self, error: BaseException | None = None) -> None:
        """Finish using only the reader's native EOF and error state."""
        self.error = self.error or error
        self.active.discard(self.attempt)
        self.attempt.finish(
            response_complete=self.content.at_eof() and self.error is None,
            error=self.error,
        )

    def observed(self, data: Any) -> Any:
        """Observe EOF without retaining or changing returned data."""
        if self.content.at_eof():
            self.finish()
        return data

    def __getattr__(self, name: str) -> Any:
        delegate = getattr(self.content, name)
        if name not in {
            "read",
            "readany",
            "readline",
            "readuntil",
            "readchunk",
            "readexactly",
        }:
            return delegate

        async def read(*args: Any, **kwargs: Any) -> Any:
            try:
                return self.observed(await delegate(*args, **kwargs))
            except BaseException as error:
                self.finish(error)
                raise

        return read

    def read_nowait(self, *args: Any, **kwargs: Any) -> bytes:
        """Observe the native synchronous reader without awaiting or buffering."""
        try:
            return cast(
                "bytes", self.observed(self.content.read_nowait(*args, **kwargs))
            )
        except BaseException as error:
            self.finish(error)
            raise

    def __aiter__(self) -> Any:
        return AsyncStreamIterator(self.readline)

    def iter_chunked(self, size: int) -> Any:
        """Preserve native chunk-size iteration."""
        return AsyncStreamIterator(lambda: self.read(size))

    def iter_any(self) -> Any:
        """Preserve native available-byte iteration."""
        return AsyncStreamIterator(self.readany)

    def iter_chunks(self) -> Any:
        """Preserve native HTTP chunk boundaries."""
        return ChunkTupleAsyncStreamIterator(self)


def instrument_aiohttp_session(session: Any) -> None:  # noqa: C901
    """Observe each native inference attempt before connector work."""
    if "_ols_audit_middleware" in vars(session):
        return
    active: set[RequestAttempt] = set()

    async def middleware(request: Any, handler: Any) -> Any:
        if not _google_inference(request):
            return await handler(request)
        target = destination_from_url(request.url)
        proxy = request.proxy
        attempt = start_attempt(
            destination_from_url(proxy) if proxy else target,
            target=target if proxy else None,
            method=request.method,
        )
        if attempt is None:
            return await handler(request)
        active.add(attempt)
        try:
            response = await handler(request)
        except BaseException as error:
            active.discard(attempt)
            attempt.finish(response_complete=False, error=error)
            raise
        attempt.response_headers(response.status)
        content = _ObservedContent(response.content, attempt, active)
        response.content = content

        def wrap(delegate: Any) -> Any:
            def close(*args: Any, **kwargs: Any) -> Any:
                primary = sys.exc_info()[1] or content.error
                try:
                    return delegate(*args, **kwargs)
                except BaseException as error:
                    content.finish(primary or error)
                    if primary is None:
                        raise
                    return None
                finally:
                    content.finish(primary)

            return close

        response.close = wrap(response.close)
        response.release = wrap(response.release)
        if content.at_eof():
            content.finish()
        return response

    session._ols_audit_middleware = middleware
    if hasattr(session, "_middlewares"):
        session._middlewares = (*session._middlewares, middleware)
    original_close = session.close

    async def close() -> None:
        error = sys.exc_info()[1]
        try:
            await original_close()
        except BaseException as cleanup:
            if error is None:
                error = cleanup
                raise
        finally:
            for attempt in tuple(active):
                attempt.finish(response_complete=False, error=error)
            active.clear()

    session.close = close


def instrument_google_client(client: Any) -> None:
    """Observe the selected Google backend without wrapping credential sessions."""
    api = client._api_client
    for name in ("_httpx_client", "_async_httpx_client"):
        native = getattr(api, name, None)
        if isinstance(native, _SYNC_CLIENTS + _ASYNC_CLIENTS):
            instrument_httpx_client(native, eligible=_google_inference)
    if getattr(api, "_ols_audit_installed", False):
        return
    api._ols_audit_installed = True
    original = api._get_aiohttp_session
    previous_middleware: Any = None

    async def get_session() -> Any:
        nonlocal previous_middleware
        session = await original()
        instrument_aiohttp_session(session)
        middleware = session._ols_audit_middleware
        args = api._async_client_session_request_args
        previous = args.get("middlewares", getattr(session, "_middlewares", ()))
        args["middlewares"] = (
            *(
                item
                for item in previous
                if item is not middleware and item is not previous_middleware
            ),
            middleware,
        )
        previous_middleware = middleware
        return session

    api._get_aiohttp_session = get_session


_pool_attempt: ContextVar[RequestAttempt | None] = ContextVar(
    "ols_pool_attempt", default=None
)
_install_lock = RLock()


def _audit_error(error: BaseException) -> BaseException:
    pending = [error]
    seen: set[int] = set()
    causes: list[BaseException] = []
    while pending and len(causes) < 16:
        cause = pending.pop()
        if id(cause) in seen:
            continue
        seen.add(id(cause))
        causes.append(cause)
        if cause.__cause__ is not None:
            pending.append(cause.__cause__)
        if isinstance(cause, urllib3_errors.SSLError):
            pending.extend(
                item for item in cause.args if isinstance(item, BaseException)
            )
        if isinstance(cause, urllib3_errors.MaxRetryError) and isinstance(
            cause.reason, BaseException
        ):
            pending.append(cause.reason)
        if isinstance(cause, urllib3_errors.ProxyError):
            pending.append(cause.original_error)
    for kinds, normalized in (
        (ssl.SSLCertVerificationError, None),
        ((socket.gaierror, urllib3_errors.NameResolutionError), socket.gaierror),
        ((ssl.SSLError, urllib3_errors.SSLError), ssl.SSLError),
        ((ConnectionError, urllib3_errors.NewConnectionError), ConnectionError),
        ((TimeoutError, urllib3_errors.TimeoutError), TimeoutError),
    ):
        for cause in causes:
            if isinstance(cause, kinds):
                return cause if normalized is None else normalized()
    return error


def _observe_urllib3_body(response: Any, attempt: RequestAttempt) -> None:  # noqa: C901
    if isinstance(response._body, bytes):
        attempt.finish(response_complete=True)
        return
    original_read, original_close, original_stream = (
        response.read,
        response.close,
        response.stream,
    )
    streaming = False

    def read(*args: Any, **kwargs: Any) -> Any:
        try:
            data = original_read(*args, **kwargs)
        except BaseException as error:
            attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        if not streaming:
            amount = kwargs.get("amt", args[0] if args else None)
            if (
                amount is None
                or (response.length_remaining == 0 and response._decoder is None)
                or (amount != 0 and not data)
            ):
                attempt.finish(response_complete=True)
        return data

    def stream(*args: Any, **kwargs: Any) -> Any:
        nonlocal streaming
        iterator = iter(original_stream(*args, **kwargs))
        try:
            while True:
                streaming = True
                try:
                    data = next(iterator)
                except StopIteration:
                    attempt.finish(response_complete=True)
                    return
                finally:
                    streaming = False
                yield data
        except BaseException as error:
            attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        finally:
            iterator.close()

    def close() -> Any:
        try:
            return original_close()
        except BaseException as error:
            attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        finally:
            attempt.finish(response_complete=False)

    response.read, response.stream, response.close = read, stream, close


def _instrument_pool(pool: Any) -> None:  # noqa: C901
    if getattr(pool, "_ols_audit_installed", False):
        return
    original_urlopen, original_make_request = pool.urlopen, pool._make_request

    def urlopen(method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        parsed = parse_url(url)
        target = (
            destination_from_url(parsed)
            if parsed.scheme and parsed.host
            else Destination(
                pool.scheme,
                pool.host,
                (
                    pool.port
                    if pool.port is not None
                    else 443 if pool.scheme == "https" else 80
                ),
            )
        )
        proxy = pool.proxy
        call = _call.get()
        eligible = (call is not None and call.kind == "auth") or (
            method == "POST"
            and parsed.path is not None
            and parsed.path.endswith(("/converse", "/converse-stream"))
        )
        attempt = (
            start_attempt(
                destination_from_url(proxy) if proxy else target,
                target=target if proxy else None,
                method=method,
            )
            if eligible
            else None
        )
        token = _pool_attempt.set(attempt)
        try:
            return original_urlopen(method, url, *args, **kwargs)
        except BaseException as error:
            if attempt is not None:
                attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        finally:
            _pool_attempt.reset(token)

    def make_request(conn: Any, *args: Any, **kwargs: Any) -> Any:
        attempt = _pool_attempt.get()
        try:
            response = original_make_request(conn, *args, **kwargs)
        except BaseException as error:
            if attempt is not None:
                attempt.finish(response_complete=False, error=_audit_error(error))
            raise
        if attempt is not None:
            attempt.response_headers(response.status)
            _observe_urllib3_body(response, attempt)
        return response

    if hasattr(pool, "_prepare_proxy"):
        original_prepare = pool._prepare_proxy

        def prepare_proxy(*args: Any, **kwargs: Any) -> Any:
            try:
                return original_prepare(*args, **kwargs)
            except BaseException as error:
                attempt = _pool_attempt.get()
                if attempt is not None:
                    attempt.finish(response_complete=False, error=_audit_error(error))
                raise

        pool._prepare_proxy = prepare_proxy
    pool.urlopen, pool._make_request = urlopen, make_request
    pool._ols_audit_installed = True


def instrument_urllib3_manager(manager: Any) -> None:
    """Observe owned native pools before CONNECT and each retry attempt."""
    with _install_lock:
        if getattr(manager, "_ols_audit_installed", False):
            return
        original = manager.connection_from_host

        def connection_from_host(*args: Any, **kwargs: Any) -> Any:
            pool = original(*args, **kwargs)
            with _install_lock:
                _instrument_pool(pool)
            return pool

        manager.connection_from_host = connection_from_host
        manager._ols_audit_installed = True


def instrument_google_request(request: Any) -> None:
    """Decorate only the native requests session supplied to credential refresh."""
    session = getattr(request, "session", None)
    if session is None or getattr(session, "_ols_auth_audit_installed", False):
        return
    original = getattr(session, "get_adapter", None)
    if not callable(original):
        return

    def get_adapter(*args: Any, **kwargs: Any) -> Any:
        adapter = original(*args, **kwargs)
        manager = getattr(adapter, "poolmanager", None)
        original_proxy = getattr(adapter, "proxy_manager_for", None)
        if manager is None or not callable(original_proxy):
            return adapter
        with _install_lock:
            if not getattr(adapter, "_ols_auth_audit_installed", False):
                instrument_urllib3_manager(manager)

                def proxy_manager_for(*args: Any, **kwargs: Any) -> Any:
                    manager = original_proxy(*args, **kwargs)
                    instrument_urllib3_manager(manager)
                    return manager

                adapter.proxy_manager_for = proxy_manager_for
                adapter._ols_auth_audit_installed = True
        return adapter

    session.get_adapter = get_adapter
    session._ols_auth_audit_installed = True


def instrument_azure_credential(credential: Any) -> Any:
    """Scope native Azure token methods and observe its existing lazy requests transport."""
    if getattr(credential, "_ols_auth_audit_installed", False):
        return credential
    client = getattr(credential, "_client", None)
    pipeline = getattr(client, "_pipeline", None)
    transport = getattr(pipeline, "_transport", None)
    original_open = getattr(transport, "open", None)
    if not callable(original_open):
        return credential

    def open_transport() -> None:
        original_open()
        instrument_google_request(transport)

    transport.open = open_transport

    def scoped(method: Any) -> Any:
        def get_token(*args: Any, **kwargs: Any) -> Any:
            with auth_request("azure"):
                return method(*args, **kwargs)

        return get_token

    for name in ("get_token", "get_token_info"):
        method = getattr(credential, name, None)
        if callable(method):
            setattr(credential, name, scoped(method))
    credential._ols_auth_audit_installed = True
    return credential


def instrument_botocore_auth_client(client: Any) -> None:
    """Observe the native STS connection managers below botocore retries and signing."""
    if getattr(client, "_ols_auth_audit_installed", False):
        return
    endpoint = getattr(client, "_endpoint", None)
    session = getattr(endpoint, "http_session", None)
    original = getattr(session, "_get_connection_manager", None)
    if not callable(original):
        return

    def connection_manager(*args: Any, **kwargs: Any) -> Any:
        manager = original(*args, **kwargs)
        instrument_urllib3_manager(manager)
        return manager

    session._get_connection_manager = connection_manager
    client._ols_auth_audit_installed = True


def instrument_google_credentials(credentials: Any) -> Any:
    """Preserve Google's native refresh and cache while observing its supplied request session."""
    if getattr(credentials, "_ols_auth_audit_installed", False):
        return credentials
    original = credentials.refresh

    def refresh(request: Any) -> Any:
        with auth_request("google"):
            instrument_google_request(request)
            return original(request)

    credentials.refresh = refresh
    credentials._ols_auth_audit_installed = True
    return credentials


def instrument_watsonx_client(  # noqa: C901  # pylint: disable=too-many-statements
    client: Any,
) -> None:
    """Observe IBM-owned pools below retries, including native TLS reinitialization."""
    paths = {
        urlsplit(client._href_definitions.get_fm_chat_href(operation)).path.encode()
        for operation in ("chat", "chat_stream")
    }

    def observe_pool(pool: Any, use_async: bool) -> None:  # noqa: C901
        if getattr(pool, "_ols_audit_installed", False):
            return
        active: set[RequestAttempt] = set()
        original = pool.handle_async_request if use_async else pool.handle_request
        original_close = pool.aclose if use_async else pool.close

        def begin(request: Any) -> RequestAttempt | None:
            call = _call.get()
            is_iam_request = (
                call is not None and call.kind == "auth" and call.auth_provider == "ibm"
            )
            if request.method != b"POST" or (
                request.url.target.split(b"?", 1)[0] not in paths and not is_iam_request
            ):
                return None
            target = destination_from_url(request.url)
            proxy = getattr(pool, "_proxy_url", None)
            attempt = start_attempt(
                destination_from_url(proxy) if proxy is not None else target,
                target=target if proxy is not None else None,
                method=(
                    request.method.decode("ascii") if request.method.isascii() else None
                ),
            )
            if attempt is not None:
                active.add(attempt)
            return attempt

        def observe_response(response: Any, attempt: RequestAttempt | None) -> Any:
            if attempt is not None:
                attempt.response_headers(response.status)
                body_class = _AsyncBody if use_async else _SyncBody
                response.stream = body_class(response.stream, attempt, active)
            return response

        def send(request: Any) -> Any:
            attempt = begin(request)
            try:
                return observe_response(original(request), attempt)
            except BaseException as error:
                if attempt is not None:
                    active.discard(attempt)
                    attempt.finish(response_complete=False, error=error)
                raise

        async def asend(request: Any) -> Any:
            attempt = begin(request)
            try:
                return observe_response(await original(request), attempt)
            except BaseException as error:
                if attempt is not None:
                    active.discard(attempt)
                    attempt.finish(response_complete=False, error=error)
                raise

        def close() -> None:
            with _closing(active, sys.exc_info()[1]):
                original_close()

        async def aclose() -> None:
            with _closing(active, sys.exc_info()[1]):
                await original_close()

        if use_async:
            pool.handle_async_request = asend
            pool.aclose = aclose
        else:
            pool.handle_request = send
            pool.close = close
        pool._ols_audit_installed = True

    def observe_transport(transport: Any, use_async: bool) -> None:
        if getattr(transport, "_ols_ibm_audit_installed", False) or not hasattr(
            transport, "_pool"
        ):
            return
        native_class: Any = type(transport)

        class ObservedTransport(native_class):
            """Observe new pools created by the native retry transport constructor."""

            def __setattr__(self, name: str, value: Any) -> None:
                super().__setattr__(name, value)
                if name == "_pool":
                    observe_pool(value, use_async)

        object.__setattr__(transport, "__class__", ObservedTransport)
        observe_pool(transport._pool, use_async)
        transport._ols_ibm_audit_installed = True

    with _install_lock:
        for native, use_async in (
            (client.httpx_client, False),
            (client.async_httpx_client, True),
        ):
            for transport in (native._transport, *native._mounts.values()):
                observe_transport(transport, use_async)


def instrument_watsonx_auth() -> None:
    """Observe native IAM token generation while leaving IBM auth logic intact."""
    from ibm_watsonx_ai.utils.auth.iam_auth import (  # pylint: disable=import-outside-toplevel
        IAMTokenAuth,
    )

    with _install_lock:
        if getattr(IAMTokenAuth, "_ols_auth_audit_installed", False):
            return

        def scoped(method: Any, *, use_async: bool) -> Any:
            if use_async:

                @wraps(method)
                async def agenerate(auth_method: Any, *args: Any, **kwargs: Any) -> Any:
                    if not (
                        _watsonx_auth_initialization.get()
                        or getattr(auth_method, "_ols_auth_audit_enabled", False)
                    ):
                        return await method(auth_method, *args, **kwargs)
                    auth_method._ols_auth_audit_enabled = True
                    with auth_request("ibm"):
                        instrument_watsonx_client(auth_method._api_client)
                        return await method(auth_method, *args, **kwargs)

                return agenerate

            @wraps(method)
            def generate(auth_method: Any, *args: Any, **kwargs: Any) -> Any:
                if not (
                    _watsonx_auth_initialization.get()
                    or getattr(auth_method, "_ols_auth_audit_enabled", False)
                ):
                    return method(auth_method, *args, **kwargs)
                auth_method._ols_auth_audit_enabled = True
                with auth_request("ibm"):
                    instrument_watsonx_client(auth_method._api_client)
                    return method(auth_method, *args, **kwargs)

            return generate

        setattr(
            IAMTokenAuth,
            "_generate_token",
            scoped(IAMTokenAuth._generate_token, use_async=False),
        )
        setattr(
            IAMTokenAuth,
            "_agenerate_token",
            scoped(IAMTokenAuth._agenerate_token, use_async=True),
        )
        setattr(IAMTokenAuth, "_ols_auth_audit_installed", True)


@contextmanager
def watsonx_auth_audit_scope() -> Iterator[None]:
    """Enable initial IBM IAM auditing while constructing the native Watsonx client."""
    instrument_watsonx_auth()
    token = _watsonx_auth_initialization.set(True)
    try:
        yield
    finally:
        _watsonx_auth_initialization.reset(token)
