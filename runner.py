"""Script that is able to start Uvicorn-based REST API service."""

import os

import uvicorn

if __name__ == "__main__":
    # TODO syedriko consolidate this env var into a central location as per OLS-345.
    # disable Gradio analytics, which calls home to https://api.gradio.app
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

    uvicorn.run("ols.app.main:app", host="127.0.0.1", port=8080, reload=True)
