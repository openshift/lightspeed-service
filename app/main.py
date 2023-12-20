import json
import requests

from fastapi import FastAPI, Request, JSONResponse

from app.endpoints import feedback, ols
from src.ui.gradio_ui import gradioUI
from utils.config import Config

app = FastAPI()

config = Config()
logger = config.logger

if config.enable_ui:
    app = gradioUI(logger=logger).mount_ui(app)
else:
    logger.info("Embedded Gradio UI is disabled. To enable set OLS_ENABLE_UI=True")


import os

import kubernetes.client
import kubernetes.utils


@app.middleware("auth")
async def auth_middleware(request: Request, call_next):
    logger.info(f"performing auth check")
    logger.info(f"request.url: {request.url}")

    # TODO: this should not be done in the future! probably not even in debug mode.
    logger.info(f"request.headers: {request.headers}")

    # check env to see if auth checking should be performed
    # TODO: move to config, properly boolean-ize
    if os.getenv("OLS_AUTH_CHECK") == "False":
        logger.info(f"auth checks disabled, skipping")
        return await call_next(request)

    # ignore auth for default routes and docs
    # TODO: is there a way to programmatically determine which paths/routes we want to skip auth on?
    if request.url.path in [
        "/",
        "/readyz",
        "/healthz",
        "/status",
        "/docs",
        "/openapi.json",
    ]:
        logger.info(f"ignoring default route for auth check")
        return await call_next(request)

    # not a default, so check if the user is authorized to use ols via the k8s auth

    # if there's no authorization header, fail with noauth http error
    if "authorization" not in request.headers:
        logger.info(f"no auth header found")
        return JSONResponse(
            status_code=requests.codes.unauthorized,
            content={"message": "no auth header found"},
        )

    logger.info(f"auth header found, checking k8s auth")

    if not k8s_auth(request.headers["authorization"]):
        return JSONResponse(
            status_code=requests.codes.unauthorized,
            content={"message": "not authorized to use ols service"},
        )

    # if we made it this far, they are authorized, so continue
    logger.info(f"passed authorization check")
    return await call_next(request)


def k8s_auth(authorization) -> bool:
    configuration = kubernetes.client.Configuration()

    # strip the token from the header
    # Configure API key authorization: BearerToken
    configuration.api_key["authorization"] = authorization.split("Bearer ")[1]

    # Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
    configuration.api_key_prefix["authorization"] = "Bearer"

    # Defining host is optional and default to http://localhost
    configuration.host = os.getenv("K8S_CLUSTER_API")

    # TODO: make configurable
    # cluster is using a self-signed cert -- would need to inject CA information
    configuration.verify_ssl = False

    # create a k8s client using the supplied authentication information
    k8s_client = kubernetes.client.ApiClient(configuration)

    # create a self subject access review to determine if the user can access the lightspeed service
    # TODO: this is currently checking a non-resource attribute path of `/ols` which is totally generic.
    #       in the future we might want to restrict users to specific ols endpoints even and make this much
    #       fancier
    jsonstr = '{"apiVersion":"authorization.k8s.io/v1","kind":"SelfSubjectAccessReview","spec":{"nonResourceAttributes":{"path":"/ols","verb":"get"}}}'

    # make a request to create the SSAR, whose return immediately contains the status of the check
    jd = json.loads(jsonstr)

    # TODO: if the client itself is unauthorized, this explodes. we probably want to separate creating and checking
    #       the k8s client into its own method which can itself cause an unauthorized error
    response = kubernetes.utils.create_from_dict(k8s_client, jd)

    if response[0].status.allowed:
        logger.info(f"passed authorization check")
        return True
    else:
        logger.info(f"failed authorization check")
        return False


def include_routers(app: FastAPI):
    """
    Include FastAPI routers for different endpoints.

    Args:
        app (FastAPI): The FastAPI app instance.
    """
    app.include_router(ols.router)
    app.include_router(feedback.router)


include_routers(app)


# TODO
# Still to be decided on their functionality
@app.get("/healthz")
@app.get("/readyz")
def read_root():
    return {"status": "1"}


# TODO
# Still to be decided on their functionality
@app.get("/")
@app.get("/status")
def root(request: Request):
    """
    TODO: In the future should respond
    """
    return {"message": "This is the default endpoint for OLS", "status": "running"}
