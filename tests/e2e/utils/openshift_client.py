"""Provide authenticated OpenShift API access for cluster tests."""

import os
from functools import lru_cache
from typing import Any

from kubernetes import client, config
from kubernetes.stream import ws_client
from openshift.dynamic import DynamicClient
from websocket import WebSocket

REQUEST_TIMEOUT = 120


class BoundedWebSocket(WebSocket):
    """Supply the connection timeout omitted by Kubernetes' WebSocket factory."""

    def connect(self, url: str, **options: Any) -> None:
        """Bound connection setup without changing process-wide socket defaults."""
        options.setdefault("timeout", REQUEST_TIMEOUT)
        super().connect(url, **options)


class ClusterDynamicClient(DynamicClient):
    """Bound resource and discovery requests with the former CLI timeout."""

    def request(self, method: str, path: str, body: Any = None, **params: Any) -> Any:
        """Supply a default timeout, preserving explicit operation timeouts."""
        if params.get("_request_timeout") is None:
            params["_request_timeout"] = REQUEST_TIMEOUT
        return super().request(method, path, body=body, **params)


@lru_cache(maxsize=1)
def get_client() -> DynamicClient:
    """Load credentials and bound Kubernetes WebSockets in the E2E process."""
    ws_client.WebSocket = BoundedWebSocket
    configuration = client.Configuration()
    if os.getenv("KUBERNETES_SERVICE_HOST") and not os.getenv("KUBECONFIG"):
        config.load_incluster_config(client_configuration=configuration)
    else:
        config.load_kube_config(
            config_file=os.getenv("KUBECONFIG") or None,
            client_configuration=configuration,
        )
    return ClusterDynamicClient(client.ApiClient(configuration))
