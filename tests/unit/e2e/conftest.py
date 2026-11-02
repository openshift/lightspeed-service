"""Keep unit tests of cluster helpers independent of real cluster credentials."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from tests.e2e.utils import cluster, openshift_client


@pytest.fixture(autouse=True)
def prevent_cluster_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail on unmocked cluster access rather than loading developer credentials."""
    openshift_client.get_client.cache_clear()
    monkeypatch.setattr(cluster, "_SESSION_TOKEN_SECRETS", set())
    for loader in ("load_kube_config", "load_incluster_config"):
        monkeypatch.setattr(
            openshift_client.config,
            loader,
            Mock(side_effect=AssertionError("Unit tests must mock cluster API access")),
        )
    try:
        yield
    finally:
        openshift_client.get_client.cache_clear()
