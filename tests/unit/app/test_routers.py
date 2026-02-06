"""Unit tests for routers.py."""

from ols import config

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints import (  # noqa:E402
    authorized,
    conversations,
    feedback,
    health,
    mcp_client_headers,
    ols,
    streaming_ols,
)
from ols.app.metrics import metrics  # noqa:E402
from ols.app.routers import include_routers  # noqa:E402


class MockFastAPI:
    """Mock class for FastAPI."""

    def __init__(self):
        """Initialize mock class."""
        self.routers = []

    def include_router(self, router, prefix=None):
        """Register new router."""
        self.routers.append(router)


def test_include_routers():
    """Test the function include_routers."""
    app = MockFastAPI()
    include_routers(app)

    # are all routers added?
    assert len(app.routers) == 8
    assert authorized.router in app.routers
    assert conversations.router in app.routers
    assert feedback.router in app.routers
    assert health.router in app.routers
    assert mcp_client_headers.router in app.routers
    assert metrics.router in app.routers
    assert ols.router in app.routers
    assert streaming_ols.router in app.routers
