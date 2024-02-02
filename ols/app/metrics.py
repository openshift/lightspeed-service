"""Prometheus metrics that are exposed by REST API."""

from prometheus_client import Counter, make_asgi_app

# just an example
llm_calls_total = Counter("llm_calls_total", "LLM calls counter")
llm_calls_failures_total = Counter("llm_calls_failures_total", "LLM calls failures")
llm_calls_validation_errors_total = Counter(
    "llm_validation_errors_total", "LLM validation errors"
)

# register metric
metrics_app = make_asgi_app()
