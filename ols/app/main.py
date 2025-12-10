"""Entry point to FastAPI-based web service."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute

from ols import config, constants, version
from ols.app import metrics, routers
from ols.customize import metadata
from ols.src.config_status import extract_config_status, store_config_status

app = FastAPI(
    title=f"Swagger {metadata.SERVICE_NAME} service - OpenAPI",
    description=f"{metadata.SERVICE_NAME} service API specification.",
    version=version.__version__,
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


logger = logging.getLogger(__name__)

# Endpoints that don't require security headers (health checks, metrics, etc.)
ENDPOINTS_WITHOUT_SECURITY_HEADERS = frozenset(
    [
        "/metrics",
        "/readiness",
        "/liveness",
    ]
)


if config.dev_config.enable_dev_ui:
    # Gradio depends on many packages like Matplotlib, Pillow etc.
    # it does not make much sense to import all these packages for
    # regular deployment
    from ols.src.ui.gradio_ui import GradioUI

    if config.dev_config.disable_tls:
        app = GradioUI().mount_ui(app)
    else:
        app = GradioUI(ols_url="https://127.0.0.1:8443/v1/query").mount_ui(app)
else:
    logger.info(
        "Embedded Gradio UI is disabled. To enable set `enable_dev_ui: true` "
        "in the `dev_config` section of the configuration file."
    )


# update provider and model as soon as possible so the metrics will be visible
# even for first scraping
metrics.setup_model_metrics(config)


@app.middleware("")
async def rest_api_counter(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Middleware with REST API counter update logic and security headers."""
    path = request.url.path

    if path not in app_routes_paths:
        return await call_next(request)

    # measure time to handle duration + update histogram
    with metrics.response_duration_seconds.labels(path).time():
        response = await call_next(request)

    # Add security headers to all endpoints except health checks and metrics

    # NOTE: These are RESPONSE headers (server -> client). Clients do not need to send them.
    # Clients can either ignore these headers or leverage them for additional security validation.
    # Browsers will enforce these policies; HTTP libraries typically ignore them.

    # This ensures new endpoints get security headers by default
    if path not in ENDPOINTS_WITHOUT_SECURITY_HEADERS:
        # X-Content-Type-Options prevents MIME-sniffing attacks
        response.headers["X-Content-Type-Options"] = "nosniff"

        # HTTP Strict Transport Security (HSTS) - only when TLS is enabled
        if not config.dev_config.disable_tls:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Additional security headers (currently commented out)
        # Uncomment these if you want additional protections:

        # Prevent clickjacking by disallowing the page to be embedded in iframes
        # response.headers["X-Frame-Options"] = "DENY"

        # Control referrer information to prevent leaking sensitive data in URLs
        # response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Legacy XSS protection (modern browsers use CSP instead)
        # response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy - restricts resource loading (requires careful tuning)
        # response.headers["Content-Security-Policy"] = "default-src 'self'"

    # ignore /metrics endpoint that will be called periodically
    if not path.endswith("/metrics"):
        # just update metrics
        metrics.rest_api_calls_total.labels(path, response.status_code).inc()
    return response


def _log_headers(headers: Headers, to_redact: frozenset[str]) -> str:
    """Serialize headers into a string while redacting sensitive values."""
    pairs = []
    for h, v in headers.items():
        value = "XXXXX" if h.lower() in to_redact else v
        pairs.append(f'"{h}":"{value}"')
    return "Headers({" + ", ".join(pairs) + "})"


@app.middleware("")
async def log_requests_responses(
    request: Request, call_next: Callable[[Request], Awaitable[StreamingResponse]]
) -> StreamingResponse:
    """Middleware for logging of HTTP requests and responses, at debug level."""
    # Bail out early if not logging or Prometheus metrics logging is suppressed
    if not logger.isEnabledFor(logging.DEBUG) or (
        config.ols_config.logging_config.suppress_metrics_in_log
        and request.url.path == "/metrics"
    ):
        return await call_next(request)

    # retrieve client host and port if provided in request object
    host = "not specified"
    port = -1
    if request.client is None:
        logger.debug("Client info is not present in request object")
    else:
        host = request.client.host
        port = request.client.port

    request_log_message = f"Request from {host}:{port} "
    request_log_message += _log_headers(
        request.headers, constants.HTTP_REQUEST_HEADERS_TO_REDACT
    )
    request_log_message += ", Body: "

    request_body = await request.body()
    if request_body:
        request_log_message += f"{request_body.decode('utf-8')}"
    else:
        request_log_message += "None"

    logger.debug(request_log_message)

    response = await call_next(request)

    response_headers_log_message = f"Response to {host}:{port} "
    response_headers_log_message += _log_headers(
        response.headers, constants.HTTP_RESPONSE_HEADERS_TO_REDACT
    )
    logger.debug(response_headers_log_message)

    async def stream_response_body(
        response_body: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[bytes, None]:
        async for chunk in response_body:
            logger.debug(
                "Response to %s:%d Body chunk: %s}",
                host,
                port,
                chunk.decode("utf-8", errors="ignore"),
            )
            yield chunk

    # current version of Starlette pass instance of _StreamingResponse class that is
    # private. Thus we need to check if the body_iterator attribute exists
    if hasattr(response, "body_iterator"):
        # The response is already a streaming response
        response.body_iterator = stream_response_body(response.body_iterator)
    else:
        # Convert non-streaming response to a streaming response to log its body
        response_body = response.body
        response = StreamingResponse(stream_response_body(iter([response_body])))

    return response


routers.include_routers(app)

app_routes_paths = [
    route.path
    for route in app.routes
    if isinstance(route, (Mount, Route, WebSocketRoute))
]

if config.ols_config.user_data_collection.config_status_enabled:
    try:
        config_status = extract_config_status(config.config)
        store_config_status(
            config.ols_config.user_data_collection.config_status_storage,
            config_status,
        )
    except Exception as e:
        logger.warning("Failed to store config status: %s", e)
