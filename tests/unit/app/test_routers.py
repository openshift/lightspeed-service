"""Unit tests for routers."""

from ols.app import routers
from ols.app.endpoints import feedback, health, ols
from tests.mock_classes.mock_app import MockApp


def test_include_routers():
    """Tests if function include_routers add all routers into FastAPI application."""
    # Alternativelly it would be possible to use MagicMock and then check calls etc.,
    # but it is a bit fragile and less readable in the end.
    app = MockApp()
    routers.include_routers(app)

    assert len(app.mounts) == 1 and app.mounts[0] == "/metrics"
    assert len(app.prefixes) == 3
    assert "/v1" in app.prefixes

    # we don't expect the calls to be ordered
    assert ols.router in app.routers
    assert feedback.router in app.routers
    assert health.router in app.routers
