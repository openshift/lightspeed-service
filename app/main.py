from fastapi import FastAPI, Request

from app.endpoints import feedback, ols
from src.ui.gradio_ui import gradioUI
from utils import config

app = FastAPI()

# config = load_config(os.environ.get("OLS_CONFIG_FILE","olsconfig.yaml"))
config.load_config_from_env()
if config.ols_config.enable_debug_ui:
    app = gradioUI(logger=config.default_logger).mount_ui(app)
else:
    config.default_logger.info(
        "Embedded Gradio UI is disabled. To enable set ENABLE_DEV_UI to True"
    )


def include_routers(app: FastAPI):
    """Include FastAPI routers for different endpoints.

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
    """TODO: In the future should respond"""
    return {"message": "This is the default endpoint for OLS", "status": "running"}
