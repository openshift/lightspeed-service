"""Prometheus metrics that are exposed by REST API."""

from typing import Any

from fastapi import APIRouter, Depends, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

import ols.app.models.config as config_model
from ols.utils.auth_dependency import AuthDependency

router = APIRouter(tags=["metrics"])
auth_dependency = AuthDependency(virtual_path="/ols-metrics-access")

rest_api_calls_total = Counter(
    "rest_api_calls_total", "REST API calls counter", ["path", "status_code"]
)

response_duration_seconds = Histogram(
    "response_duration_seconds", "Response durations", ["path"]
)

llm_calls_total = Counter("llm_calls_total", "LLM calls counter", ["provider", "model"])
llm_calls_failures_total = Counter("llm_calls_failures_total", "LLM calls failures")
llm_calls_validation_errors_total = Counter(
    "llm_validation_errors_total", "LLM validation errors"
)

llm_token_sent_total = Counter(
    "llm_token_sent_total", "LLM tokens sent", ["provider", "model"]
)
llm_token_received_total = Counter(
    "llm_token_received_total", "LLM tokens received", ["provider", "model"]
)

# expose selected provider and model
# (these are represented by counters, but the only meaning is presence of label)
selected_model = Info("selected_model", "Selected model")

# metric that indicates what provider + model customers are using so we can
# understand what is popular/important
model_enabled = Gauge("model_enabled", "Enabled LLM models", ["provider", "model"])


@router.get("/metrics")
def get_metrics(auth: Any = Depends(auth_dependency)) -> Response:
    """Metrics Endpoint.

    Args:
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.

    Returns:
        Response containing the latest metrics.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_model_metrics(config: config_model.Config) -> None:
    """Perform setup of all metrics related to LLM model and provider."""
    selected_model.info(
        {
            "model": config.ols_config.default_model,
            "provider": config.ols_config.default_provider,
        }
    )

    for _, provider in config.llm_config.providers.items():
        provider_type = provider.type
        for model_name, _ in provider.models.items():
            model_enabled.labels(provider_type, model_name).inc()
