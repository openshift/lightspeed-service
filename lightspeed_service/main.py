from fastapi import FastAPI
from fastapi import Request

from lightspeed_service.routers import feedback
from lightspeed_service.routers import ols
from lightspeed_service.ui.gradio_ui import gradioUI
from lightspeed_service.utils.config import Config


app = FastAPI()

config = Config()

logger = config.logger

if config.enable_ui:
    app = gradioUI().mount_ui(app)
else:
    logger.info(
        "Embedded Gradio UI is disabled. To enable set OLS_ENABLE_UI=True"
    )


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
    return {
        "message": "This is the default endpoint for OLS",
        "status": "running",
    }
