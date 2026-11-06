"""Test authenticated, lazily loaded cluster API access without a live cluster."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.e2e.utils import openshift_client


@pytest.mark.parametrize("in_cluster", [False, True])
def test_get_client_loads_credentials_lazily(
    monkeypatch: pytest.MonkeyPatch, in_cluster: bool
) -> None:
    """Load credentials once and reuse the authenticated client."""
    monkeypatch.delenv("KUBECONFIG", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    if in_cluster:
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "cluster.example")
    openshift_client.get_client.cache_clear()
    try:
        with (
            patch.object(openshift_client.config, "load_kube_config") as kube,
            patch.object(
                openshift_client.config, "load_incluster_config"
            ) as service_account,
            patch.object(openshift_client, "ClusterDynamicClient") as dynamic,
            patch.object(openshift_client, "ws_client") as websocket_client,
        ):
            assert openshift_client.get_client() is dynamic.return_value
            assert openshift_client.get_client() is dynamic.return_value
            assert kube.call_count == int(not in_cluster)
            assert service_account.call_count == int(in_cluster)
            dynamic.assert_called_once()
            assert websocket_client.WebSocket is openshift_client.BoundedWebSocket
            loader = service_account if in_cluster else kube
            configuration = loader.call_args.kwargs["client_configuration"]
            assert dynamic.call_args.args[0].configuration is configuration
    finally:
        openshift_client.get_client.cache_clear()


def test_get_client_honors_explicit_kubeconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefer an explicit kubeconfig even when running inside a cluster."""
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "cluster.example")
    monkeypatch.setenv("KUBECONFIG", "/test/config")
    openshift_client.get_client.cache_clear()
    try:
        with (
            patch.object(openshift_client.config, "load_kube_config") as kube,
            patch.object(
                openshift_client.config, "load_incluster_config"
            ) as service_account,
            patch.object(openshift_client, "ClusterDynamicClient"),
            patch.object(openshift_client, "ws_client"),
        ):
            openshift_client.get_client()
            kube.assert_called_once()
            assert kube.call_args.kwargs["config_file"] == "/test/config"
            service_account.assert_not_called()
    finally:
        openshift_client.get_client.cache_clear()


def test_discovery_and_resource_requests_have_timeouts(tmp_path: Path) -> None:
    """Exercise real discovery and preserve explicit resource timeout overrides."""
    configuration = openshift_client.client.Configuration()
    configuration.host = "https://unit-test.invalid"
    replies = {
        "/version": {"gitVersion": "v1.34.0"},
        "/version/openshift": {},
        "/apis": {"kind": "APIGroupList", "apiVersion": "v1", "groups": []},
        "/api/v1": {
            "kind": "APIResourceList",
            "apiVersion": "v1",
            "groupVersion": "v1",
            "resources": [
                {"name": "pods", "kind": "Pod", "namespaced": True, "verbs": ["list"]}
            ],
        },
        "/api/v1/namespaces/test/pods": {
            "kind": "PodList",
            "apiVersion": "v1",
            "items": [],
        },
    }
    with openshift_client.client.ApiClient(configuration) as transport:
        with patch.object(transport, "call_api") as request:
            request.side_effect = lambda path, *_a, **_k: SimpleNamespace(
                data=json.dumps(replies[path]).encode()
            )
            dynamic = openshift_client.ClusterDynamicClient(
                transport, cache_file=str(tmp_path / "discovery.json")
            )
            pods = dynamic.resources.get(api_version="v1", kind="Pod")
            pods.get(namespace="test")
            pods.get(namespace="test", _request_timeout=600)
            pods.get(namespace="test", _request_timeout=None)
    calls = request.call_args_list
    assert {call.args[0] for call in calls} == set(replies)
    assert [call.kwargs["_request_timeout"] for call in calls] == [120] * 5 + [600, 120]
