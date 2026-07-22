"""Prometheus metrics that are exposed by REST API."""

from typing import Annotated, Any

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
from ols.src.auth.auth import get_auth_dependency
from ols.utils.config import AppConfig

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

llm_token_sent_total = Counter(
    "ols_llm_token_sent_total", "LLM tokens sent", ["provider", "model"]
)
llm_token_received_total = Counter(
    "ols_llm_token_received_total", "LLM tokens received", ["provider", "model"]
)
llm_reasoning_token_total = Counter(
    "ols_llm_reasoning_token_total",
    "LLM reasoning summary tokens received",
    ["provider", "model"],
)

gen_ai_client_token_usage = Histogram(
    "gen_ai_client_token_usage",
    "Per-request token usage distribution (OTel GenAI semantic conventions)",
    [
        "gen_ai_operation_name",
        "gen_ai_token_type",
        "gen_ai_request_model",
        "gen_ai_provider_name",
    ],
    buckets=(
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
    ),
)

gen_ai_client_operation_duration_seconds = Histogram(
    "gen_ai_client_operation_duration_seconds",
    "LLM inference call duration (OTel GenAI semantic conventions)",
    ["gen_ai_request_model", "gen_ai_provider_name", "gen_ai_operation_name"],
)

gen_ai_execute_tool_duration_seconds = Histogram(
    "gen_ai_execute_tool_duration_seconds",
    "Tool execution duration (OTel GenAI semantic conventions)",
    ["gen_ai_tool_name"],
)

# metric that indicates what provider + model customers are using so we can
# understand what is popular/important
provider_model_configuration = Gauge(
    "ols_provider_model_configuration",
    "LLM provider/models combinations defined in configuration",
    ["provider", "model"],
)


@router.get("/metrics", response_class=PlainTextResponse)
def get_metrics(auth: Annotated[Any, Depends(auth_dependency)]) -> PlainTextResponse:
    """Metrics Endpoint.

    Args:
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.

    Returns:
        Response containing the latest metrics.
    """
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
