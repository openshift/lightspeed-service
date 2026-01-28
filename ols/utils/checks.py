"""Checks that are performed to configuration options."""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

from pydantic import AnyHttpUrl, FilePath

from ols import constants


class InvalidConfigurationError(Exception):
    """OLS Configuration is invalid."""


def is_valid_http_url(url: AnyHttpUrl) -> bool:
    """Check is a string is a well-formed http or https URL."""
    result = urlparse(str(url))
    return all([result.scheme, result.netloc]) and result.scheme in {
        "http",
        "https",
    }


def get_attribute_from_file(data: dict, file_name_key: str) -> Optional[str]:
    """Retrieve value of an attribute from a file."""
    file_path = data.get(file_name_key)
    if file_path is not None:
        with open(file_path, encoding="utf-8") as f:
            return f.read().rstrip()
    return None


def read_secret(
    data: dict,
    path_key: str,
    default_filename: str,
    raise_on_error: bool = True,
    directory_name_expected: bool = False,
) -> Optional[str]:
    """Read secret from file on given path or from filename if path points to directory."""
    path = data.get(path_key)

    if path is None:
        return None

    filename = path
    if os.path.isdir(path):
        filename = os.path.join(path, default_filename)
    elif directory_name_expected:
        msg = "Improper credentials_path specified: it must contain path to directory with secrets."
        # no logging configured yet
        print(msg)
        return None

    try:
        with open(filename, encoding="utf-8") as f:
            return f.read().rstrip()
    except OSError as e:
        # some files with secret must exist, so for such cases it is time
        # to inform about improper configuration
        if raise_on_error:
            raise
        # no logging configured yet
        print(f"Problem reading secret from file {filename}:", e)
        print(f"Verify the provider secret contains {default_filename}")
        return None


def dir_check(path: FilePath, desc: str) -> None:
    """Check that path is a readable directory."""
    if not os.path.exists(path):
        raise InvalidConfigurationError(f"{desc} '{path}' does not exist")
    if not os.path.isdir(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a directory")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


def file_check(path: FilePath, desc: str) -> None:
    """Check that path is a readable regular file."""
    if not os.path.isfile(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a file")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


def get_log_level(value: str) -> int:
    """Get log level from string."""
    if not isinstance(value, str):
        raise InvalidConfigurationError(
            f"'{value}' log level must be string, got {type(value)}"
        )
    log_level = logging.getLevelName(value.upper())
    if not isinstance(log_level, int):
        raise InvalidConfigurationError(
            f"'{value}' is not valid log level, valid levels are "
            f"{[k.lower() for k in logging.getLevelNamesMapping()]}"
        )
    return log_level


def resolve_headers(
    headers: dict[str, str],
    auth_module: Optional[str] = None,
) -> dict[str, str]:
    """Resolve authorization headers by reading secret files or preserving special values.

    Args:
        headers: Map of header names to secret locations or special keywords.
            - If value is "kubernetes": preserved unchanged for later substitution during request.
              Only valid when authentication module is "k8s" or "noop-with-token".
              "noop-with-token" is for testing only - the real k8s token must be passed at
              request time.
              If used with other auth modules, a warning is logged and the server is skipped.
            - If value is "client": preserved unchanged for later substitution during request.
            - Otherwise: Treated as file path and read the secret from that file.
        auth_module: The authentication module being used (e.g., "k8s", "noop-with-token").
            Used to validate that "kubernetes" placeholder is only used with appropriate auth
            modules.

    Returns:
        Map of header names to resolved header values or special keywords.
        Returns empty dict if any header fails to resolve (kubernetes placeholder
        with non-k8s/non-noop-with-token auth, or secret file cannot be read).

    Examples:
        >>> # With file paths
        >>> resolve_headers({"Authorization": "/var/secrets/token"})
        {"Authorization": "secret-value-from-file"}

        >>> # With kubernetes special case (kept as-is, requires k8s or noop-with-token auth)
        >>> resolve_authorization_headers(
        ...     {"Authorization": "kubernetes"},
        ...     auth_module="k8s"
        ... )
        {"Authorization": "kubernetes"}

        >>> # With client special case (kept as-is)
        >>> resolve_headers({"Authorization": "client"})
        {"Authorization": "client"}
    """
    logger = logging.getLogger(__name__)
    resolved: dict[str, str] = {}

    for header_name, header_value in headers.items():
        match header_value.strip():
            case constants.MCP_KUBERNETES_PLACEHOLDER:
                # Validate that kubernetes placeholder is only used with k8s or noop-with-token auth
                # (noop-with-token is allowed for testing purposes)
                if auth_module not in (
                    constants.DEFAULT_AUTHENTICATION_MODULE,
                    constants.NOOP_WITH_TOKEN_AUTHENTICATION_MODULE,
                ):
                    logger.warning(
                        "MCP server authorization header '%s' uses '%s' placeholder, but "
                        "authentication module is '%s'. "
                        "The 'kubernetes' placeholder requires authentication module to be "
                        "'%s' or '%s'. This MCP server will be skipped.",
                        header_name,
                        constants.MCP_KUBERNETES_PLACEHOLDER,
                        auth_module,
                        constants.DEFAULT_AUTHENTICATION_MODULE,
                        constants.NOOP_WITH_TOKEN_AUTHENTICATION_MODULE,
                    )
                    return {}  # Return empty dict to signal server should be skipped
                resolved[header_name] = constants.MCP_KUBERNETES_PLACEHOLDER
                logger.debug(
                    "Header %s will use Kubernetes token (resolved at request time)",
                    header_name,
                )

            case constants.MCP_CLIENT_PLACEHOLDER:
                resolved[header_name] = constants.MCP_CLIENT_PLACEHOLDER
                logger.debug(
                    "Header %s will use client-provided token (resolved at request time)",
                    header_name,
                )

            case _:
                # Read secret from file path
                secret_value = read_secret(
                    data={"path": header_value},
                    path_key="path",
                    default_filename="",
                    raise_on_error=False,
                )
                if secret_value:
                    resolved[header_name] = secret_value
                    logger.debug(
                        "Resolved header %s from secret file %s",
                        header_name,
                        header_value,
                    )
                else:
                    logger.warning(
                        "MCP server authorization header '%s' failed to read secret file '%s'. "
                        "This MCP server will be skipped.",
                        header_name,
                        header_value,
                    )
                    return {}  # Return empty dict to signal server should be skipped

    return resolved


def validate_mcp_servers(
    servers: list,
    auth_module: Optional[str],
) -> list:
    """Validate and filter MCP servers, resolving their authorization headers.

    Args:
        servers: List of MCPServerConfig objects to validate.
        auth_module: The authentication module being used (e.g., "k8s", "noop").

    Returns:
        List of valid MCPServerConfig objects with resolved authorization headers.
        Servers are excluded if any authorization header cannot be resolved.
    """
    logger = logging.getLogger(__name__)
    valid_servers = []

    for server in servers:
        if server.headers:
            # Resolve headers with auth module context
            resolved = resolve_headers(
                server.headers,
                auth_module=auth_module,
            )
            if not resolved:
                # Already logged in resolve_headers
                logger.debug(
                    "MCP server '%s' excluded due to unresolvable authorization headers",
                    server.name,
                )
                continue
            # Store the resolved headers
            server._resolved_headers = resolved
        valid_servers.append(server)

    return valid_servers
