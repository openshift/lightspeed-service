"""MCPConfigBuilder for building MultiServerMCPClient configuration."""

import logging
import os
from typing import Any

from ols.app.models.config import MCPServerConfig

logger = logging.getLogger(__name__)


# Additional header containing user token for k8s/ocp authentication
# for SSE MCP servers.
K8S_AUTH_HEADER = "kubernetes-authorization"


class MCPConfigBuilder:
    """Builds MCP config for MultiServerMCPClient."""

    def __init__(
        self, user_token: str, mcp_server_configs: list[MCPServerConfig]
    ) -> None:
        """Initialize the MCPConfigBuilder with user token and server config list."""
        self.user_token = user_token
        self.mcp_server_configs = mcp_server_configs

    @staticmethod
    def include_auth_header(user_token: str, config: dict[str, Any]) -> dict[str, Any]:
        """Include user token in the config headers."""
        if "headers" not in config:
            config["headers"] = {}
        if K8S_AUTH_HEADER in config["headers"]:
            logger.warning(
                "Kubernetes auth header is already set, overriding with actual user token."
            )
        config["headers"][K8S_AUTH_HEADER] = f"Bearer {user_token}"
        return config

    def include_auth_to_stdio(self, server_envs: dict[str, str]) -> dict[str, str]:
        """Resolve OpenShift stdio env config."""
        logger.debug("Updating env configuration of openshift stdio mcp server")
        env = {**server_envs}

        if "OC_USER_TOKEN" in env:
            logger.warning("OC_USER_TOKEN is set, overriding with actual user token.")
        env["OC_USER_TOKEN"] = self.user_token

        if "KUBECONFIG" not in env:
            if "KUBECONFIG" in os.environ:
                logger.info("Using KUBECONFIG from environment.")
                env["KUBECONFIG"] = os.environ["KUBECONFIG"]
            elif (
                "KUBERNETES_SERVICE_HOST" in os.environ
                and "KUBERNETES_SERVICE_PORT" in os.environ
            ):
                logger.info("Using KUBERNETES_SERVICE_* from environment.")
                env["KUBERNETES_SERVICE_HOST"] = os.environ["KUBERNETES_SERVICE_HOST"]
                env["KUBERNETES_SERVICE_PORT"] = os.environ["KUBERNETES_SERVICE_PORT"]
            else:
                logger.error("Missing necessary KUBECONFIG/KUBERNETES_SERVICE_* envs.")
        return env

    def dump_client_config(self) -> dict[str, Any]:
        """Convert server configs to MultiServerMCPClient config format."""
        servers_conf: dict[str, Any] = {}

        for server_conf in self.mcp_server_configs:
            servers_conf[server_conf.name] = {
                "transport": server_conf.transport,
            }

            if server_conf.stdio:
                stdio_conf = server_conf.stdio.model_dump()
                if server_conf.name == "openshift":
                    stdio_conf["env"] = self.include_auth_to_stdio(
                        server_conf.stdio.env
                    )
                servers_conf[server_conf.name].update(stdio_conf)

            if server_conf.sse:
                sse_conf = server_conf.sse.model_dump()
                self.include_auth_header(self.user_token, sse_conf)
                servers_conf[server_conf.name].update(sse_conf)

        return servers_conf
