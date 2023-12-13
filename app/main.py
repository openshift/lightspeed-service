import gradio as gr
from fastapi import FastAPI, Request
from src.ui.gradio_ui import gradioUI
from utils.config import Config
from app.endpoints import ols, feedback

app = FastAPI()

config = Config()
logger = config.logger

if config.enable_ui:
    app = gradioUI(logger=logger).mount_ui(app)
else:
    logger.info("Embedded Gradio UI is disabled. To enable set OLS_ENABLE_UI=True")


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
