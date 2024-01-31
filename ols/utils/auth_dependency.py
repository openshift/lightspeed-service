"""Handles authentication in a FastAPI app,.

integrating with Kubernetes SSAR and RBAC for auth checks.
"""

import json
import logging

import kubernetes.client
import kubernetes.utils
from fastapi import HTTPException, Request
from urllib3.exceptions import MaxRetryError

from ols.utils import config

logger = logging.getLogger(__name__)


async def auth_dependency(request: Request):
    """Authenticate API requests using Kubernetes SSAR.

    Validates the authorization header and bearer token in the request. If authentication
    is disabled in the configuration, it's skipped.

    Args:
        request (Request): The incoming request object from FastAPI.

    Raises:
        HTTPException: If authentication fails or headers are missing.
    """
    if config.dev_config.disable_auth:
        logger.info("Auth checks disabled, skipping")
        return
    else:
        # Validate the presence and format of the authorization header
        authorization_header = request.headers.get("Authorization")
        if not authorization_header:
            raise HTTPException(
                status_code=401, detail="Unauthorized: No auth header found"
            )
        # Split the header to extract the token and validate its presence
        token = _extract_bearer_token(authorization_header)
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Bearer token not found or invalid",
            )
        # Perform authentication with Kubernetes
        if not _k8s_auth(token):
            raise HTTPException(
                status_code=403, detail="Forbidden: Authentication failed"
            )


def _k8s_auth(api_key) -> bool:
    try:
        configuration = kubernetes.client.Configuration()
        configuration.api_key["authorization"] = api_key
        configuration.api_key_prefix["authorization"] = "Bearer"
        configuration.host = config.ols_config.authentication_config.k8s_cluster_api
        configuration.verify_ssl = True

        if config.ols_config.authentication_config.k8s_ca_cert_path:
            configuration.ssl_ca_cert = (
                config.ols_config.authentication_config.k8s_ca_cert_path
            )
        if config.ols_config.authentication_config.skip_tls_verification:
            configuration.verify_ssl = False

        k8s_client = kubernetes.client.ApiClient(configuration)
        jd = json.loads(
            '{"apiVersion":"authorization.k8s.io/v1","kind":"SelfSubjectAccessReview","spec":{"nonResourceAttributes":{"path":"/ols-access","verb":"get"}}}'
        )
        response = kubernetes.utils.create_from_dict(k8s_client, jd)
        if response[0].status.allowed:
            logger.info("passed authorization check")
            return True
        logger.info("failed authorization check (Unauthorized)")
        return False
    except kubernetes.client.exceptions.ApiException as e:
        logger.error(f"Kubernetes API exception: {e}")
        return False
    except MaxRetryError as e:
        logger.error(f"Kubernetes connection exception: {e}")
        return False
    except Exception as e:
        logger.error(f"Authentication/Athorization fail: {e}")
        return False


def _extract_bearer_token(header: str) -> str:
    """Extract the bearer token from the authorization header.

    Returns the token if present, else returns an empty string.
    """
    try:
        scheme, token = header.split(" ", 1)
        return token if scheme.lower() == "bearer" else ""
    except ValueError:
        return ""
