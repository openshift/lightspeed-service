"""Prometheus metrics that are exposed by REST API."""

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)

from ols import config
from ols.app.metrics.quota_metrics_service import (
    get_quota_metrics_collector,
    update_quota_metrics_on_request,
)
from ols.src.auth.auth import get_auth_dependency
from ols.utils.config import AppConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])
auth_dependency = get_auth_dependency(
    config.ols_config, virtual_path="/ols-metrics-access"
)

disable_created_metrics()  # type: ignore [no-untyped-call]

rest_api_calls_total = Counter(
    "ols_rest_api_calls_total", "REST API calls counter", ["path", "status_code"]
)

response_duration_seconds = Histogram(
    "ols_response_duration_seconds", "Response durations", ["path"]
)

llm_calls_total = Counter(
    "ols_llm_calls_total", "LLM calls counter", ["provider", "model"]
)
llm_calls_failures_total = Counter("ols_llm_calls_failures_total", "LLM calls failures")
llm_calls_validation_errors_total = Counter(
    "ols_llm_validation_errors_total", "LLM validation errors"
)

llm_token_sent_total = Counter(
    "ols_llm_token_sent_total", "LLM tokens sent", ["provider", "model"]
)
llm_token_received_total = Counter(
    "ols_llm_token_received_total", "LLM tokens received", ["provider", "model"]
)

# metric that indicates what provider + model customers are using so we can
# understand what is popular/important
provider_model_configuration = Gauge(
    "ols_provider_model_configuration",
    "LLM provider/models combinations defined in configuration",
    ["provider", "model"],
)


def get_quota_metrics_dependency() -> Optional[Any]:
    """FastAPI dependency to provide quota metrics collector.

    Returns:
        QuotaMetricsCollector instance or None if not configured or failed to initialize
    """
    try:
        # Check if quota handlers are configured
        if (
            config.ols_config.quota_handlers is None
            or config.ols_config.quota_handlers.storage is None
        ):
            logger.debug("Quota handlers not configured, skipping quota metrics")
            return None

        return get_quota_metrics_collector(config.ols_config.quota_handlers.storage)

    except Exception as e:
        logger.error("Failed to initialize quota metrics collector: %s", e)
        # Return None to gracefully degrade - metrics endpoint should still work
        return None


@router.get("/metrics", response_class=PlainTextResponse)
def get_metrics(
    auth: Annotated[Any, Depends(auth_dependency)],
    quota_collector: Annotated[Optional[Any], Depends(get_quota_metrics_dependency)],
) -> PlainTextResponse:
    """Metrics Endpoint.

    Args:
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.
        quota_collector: The quota metrics collector dependency (optional)

    Returns:
        Response containing the latest metrics.
    """
    # Update quota metrics if collector is available
    update_quota_metrics_on_request(quota_collector)

    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_model_metrics(config: AppConfig) -> None:
    """Perform setup of all metrics related to LLM model and provider."""
    # Set to track which provider/model combinations are set to 1, to
    # avoid setting provider/model to 0 in case it is already in metric
    # with value 1 - case when there are more "same" providers/models
    # combinations, but with the different names and other parameters.
    model_metrics_set = set()

    for provider in config.llm_config.providers.values():
        for model_name in provider.models:
            label_key = (provider.type, model_name)
            if (
                provider.name == config.ols_config.default_provider
                and model_name == config.ols_config.default_model
            ):
                provider_model_configuration.labels(*label_key).set(1)
                model_metrics_set.add(label_key)
            elif label_key not in model_metrics_set:
                provider_model_configuration.labels(*label_key).set(0)
