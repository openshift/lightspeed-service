"""REST API routers."""

import logging

from fastapi import FastAPI

from ols.app.endpoints import (
    authorized,
    conversations,
    feedback,
    health,
    mcp_apps,
    mcp_client_headers,
    ols,
    streaming_ols,
    tool_approvals,
)
from ols.app.metrics import metrics

logger = logging.getLogger(__name__)


def _mount_a2a_routes(app: FastAPI) -> None:
    """Mount A2A protocol routes onto the FastAPI app.

    Args:
        app: The `FastAPI` app instance.
    """
    # pylint: disable=import-outside-toplevel
    from a2a.server.apps import A2AFastAPIApplication
    from a2a.server.events import InMemoryQueueManager
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore

    from ols.src.a2a.server import OLSAgentExecutor, build_agent_card

    # pylint: enable=import-outside-toplevel

    server_url = str(app.root_path or "http://localhost:8443")
    agent_card = build_agent_card(server_url)

    request_handler = DefaultRequestHandler(
        agent_executor=OLSAgentExecutor(),
        task_store=InMemoryTaskStore(),
        queue_manager=InMemoryQueueManager(),
    )

    a2a_app = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    a2a_app.add_routes_to_app(
        app,
        agent_card_url="/.well-known/agent-card.json",
        rpc_url="/a2a",
    )
    logger.info("A2A protocol routes mounted at /a2a")


def include_routers(app: FastAPI) -> None:
    """Include FastAPI routers for different endpoints.

    Args:
        app: The `FastAPI` app instance.
    """
    app.include_router(ols.router, prefix="/v1")
    app.include_router(streaming_ols.router, prefix="/v1")
    app.include_router(mcp_client_headers.router, prefix="/v1")
    app.include_router(mcp_apps.router, prefix="/v1")
    app.include_router(tool_approvals.router, prefix="/v1")
    app.include_router(feedback.router, prefix="/v1")
    app.include_router(conversations.router, prefix="/v1")
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(authorized.router)
    _mount_a2a_routes(app)
