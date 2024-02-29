"""Manage authentication flow for FastAPI endpoints with K8S/OCP."""

import logging
from typing import Optional, Self

import kubernetes.client
from fastapi import HTTPException, Request
from kubernetes.client.rest import ApiException
from kubernetes.config import ConfigException

from ols.constants import DEFAULT_USER_NAME, DEFAULT_USER_UID
from ols.utils import config

logger = logging.getLogger(__name__)


class K8sClientSingleton:
    """Return the Kubernetes client instances.

    Ensures we initialize the k8s client only once per application life cycle.
    manage the initialization and config loading.
    """

    _instance = None
    _api_client = None
    _authn_api: kubernetes.client.AuthenticationV1Api = None
    _authz_api: kubernetes.client.AuthorizationV1Api = None

    def __new__(cls: type[Self]) -> Self:
        """Create a new instance of the singleton, or returns the existing instance.

        This method initializes the Kubernetes API clients the first time it is called.
        and ensures that subsequent calls return the same instance.
        """
        if cls._instance is None:
            cls._instance = super(K8sClientSingleton, cls).__new__(cls)
            configuration = kubernetes.client.Configuration()

            try:
                if config.ols_config.authentication_config.k8s_cluster_api not in [
                    None,
                    "None",
                    "",
                ] and config.dev_config.k8s_auth_token not in [None, "None", ""]:
                    logger.info("loading kubeconfig from app Config config")
                    configuration.api_key["authorization"] = (
                        config.dev_config.k8s_auth_token
                    )
                    configuration.api_key_prefix["authorization"] = "Bearer"
                else:
                    logger.debug(
                        "no Auth Token Override was provided,\
                            procceeding with in-cluster config load"
                    )
                    try:
                        logger.info("loading in-cluster config")
                        kubernetes.config.load_incluster_config(
                            client_configuration=configuration
                        )
                    except ConfigException as e:
                        logger.debug(f"unable to load in-cluster config: {e}")
                        try:
                            logger.info("loading config from kube-config file")
                            kubernetes.config.load_kube_config(
                                client_configuration=configuration
                            )
                        except ConfigException as e:
                            logger.error(
                                f"failed to load kubeconfig, in-cluster config\
                                  and no override token was provided: {e}"
                            )

                configuration.host = (
                    config.ols_config.authentication_config.k8s_cluster_api
                    if config.ols_config.authentication_config.k8s_cluster_api
                    not in [None, "None", ""]
                    else configuration.host
                )
                configuration.verify_ssl = (
                    not config.ols_config.authentication_config.skip_tls_verification
                )
                configuration.ssl_ca_cert = (
                    config.ols_config.authentication_config.k8s_ca_cert_path
                    if config.ols_config.authentication_config.k8s_ca_cert_path
                    not in [None, "None", ""]
                    else configuration.ssl_ca_cert
                )
                api_client = kubernetes.client.ApiClient(configuration)
                cls._api_client = api_client
                cls._authn_api = kubernetes.client.AuthenticationV1Api(api_client)
                cls._authz_api = kubernetes.client.AuthorizationV1Api(api_client)
            except Exception as e:
                logger.info(f"Failed to initialize Kubernetes client: {e}")
                raise
        return cls._instance

    @classmethod
    def get_authn_api(cls) -> kubernetes.client.AuthenticationV1Api:
        """Return the Authentication API client instance.

        Ensures the singleton is initialized before returning the Authentication API client.
        """
        if cls._instance is None or cls._authn_api is None:
            cls()
        return cls._authn_api

    @classmethod
    def get_authz_api(cls) -> kubernetes.client.AuthorizationV1Api:
        """Return the Authorization API client instance.

        Ensures the singleton is initialized before returning the Authorization API client.
        """
        if cls._instance is None or cls._authz_api is None:
            cls()
        return cls._authz_api


def get_user_info(token: str) -> Optional[kubernetes.client.V1TokenReview]:
    """Perform a Kubernetes TokenReview to validate a given token.

    Parameters:
        token (str): The bearer token to be validated.

    Returns:
        The user information if the token is valid, None otherwise.
    """
    auth_api = K8sClientSingleton.get_authn_api()
    token_review = kubernetes.client.V1TokenReview(
        spec=kubernetes.client.V1TokenReviewSpec(token=token)
    )
    try:
        response = auth_api.create_token_review(token_review)
        if response.status.authenticated:
            return response.status
        else:
            return None
    except ApiException as e:
        logger.error(f"API exception during TokenReview: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during TokenReview - Unauthorized: {e}")
        raise HTTPException(status_code=500, detail="Forbidden: Unable to Review Token")


def _extract_bearer_token(header: str) -> str:
    """Extract the bearer token from an HTTP authorization header.

    Parameters:
        header (str): The authorization header containing the token.

    Returns:
        The extracted token if present, else an empty string.
    """
    try:
        scheme, token = header.split(" ", 1)
        return token if scheme.lower() == "bearer" else ""
    except ValueError:
        return ""


async def auth_dependency(request: Request) -> tuple[str, str]:
    """Validate FastAPI Requests for authentication and authorization.

    Validates the bearer token from the request,
    performs access control checks using Kubernetes TokenReview and SubjectAccessReview.

    Parameters:
        request (Request): The FastAPI request object.

    Returns:
        The user's UID and username if authentication and authorization succeed.

    Raises:
        HTTPException: If authentication fails or the user does not have access.
    """
    if config.dev_config.disable_auth:
        logger.warn("Auth checks disabled, skipping")
        # TODO: replace with constants for default identity
        return DEFAULT_USER_UID, DEFAULT_USER_NAME
    authorization_header = request.headers.get("Authorization")
    if not authorization_header:
        raise HTTPException(
            status_code=401, detail="Unauthorized: No auth header found"
        )
    token = _extract_bearer_token(authorization_header)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Bearer token not found or invalid",
        )
    user_info = get_user_info(token)
    if user_info is None:
        raise HTTPException(
            status_code=403, detail="Forbidden: Invalid or expired token"
        )
    authorization_api = K8sClientSingleton.get_authz_api()
    # TODO: we can support Groups here also admins, developers etc..
    sar = kubernetes.client.V1SubjectAccessReview(
        spec=kubernetes.client.V1SubjectAccessReviewSpec(
            user=user_info.user.username,
            non_resource_attributes=kubernetes.client.V1NonResourceAttributes(
                path="/ols-access", verb="get"
            ),
        )
    )
    try:
        response = authorization_api.create_subject_access_review(sar)
        if not response.status.allowed:
            raise HTTPException(
                status_code=403, detail="Forbidden: User does not have access"
            )
    except ApiException as e:
        logger.error(f"API exception during SubjectAccessReview: {e}")
        raise HTTPException(status_code=403, detail="Internal server error")

    return user_info.user.uid, user_info.user.username
