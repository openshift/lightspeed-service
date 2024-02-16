"""Prometheus metrics that are exposed by REST API."""

from prometheus_client import Counter, Histogram, make_asgi_app

rest_api_calls_total = Counter(
    "rest_api_calls_total", "REST API calls counter", ["path", "status_code"]
)

response_duration_seconds = Histogram(
    "response_duration_seconds", "Response durations", ["path"]
)

llm_calls_total = Counter("llm_calls_total", "LLM calls counter")
llm_calls_failures_total = Counter("llm_calls_failures_total", "LLM calls failures")
llm_calls_validation_errors_total = Counter(
    "llm_validation_errors_total", "LLM validation errors"
)

# register metric
metrics_app = make_asgi_app()
