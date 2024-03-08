"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import os
import tempfile

import uvicorn


def configure_hugging_face_envs(ols_config) -> None:
    """Configure HuggingFace library environment variables."""
    if (
        ols_config
        and hasattr(ols_config, "reference_content")
        and hasattr(ols_config.reference_content, "embeddings_model_path")
        and ols_config.reference_content.embeddings_model_path
    ):
        os.environ["TRANSFORMERS_CACHE"] = (
            ols_config.reference_content.embeddings_model_path
        )
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def configure_gradio_ui_envs() -> None:
    """Configure GradioUI framework environment variables."""
    # disable Gradio analytics, which calls home to https://api.gradio.app
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

    # Setup config directory for Matplotlib. It will be used to store info
    # about fonts (usually one JSON file) and it really is just temporary
    # storage that can be deleted at any time and recreated later.
    # Fixes: https://issues.redhat.com/browse/OLS-301
    tempdir = os.path.join(tempfile.gettempdir(), "matplotlib")
    os.environ["MPLCONFIGDIR"] = tempdir


if __name__ == "__main__":
    configure_gradio_ui_envs()

    # NOTE: We import config here to avoid triggering import of anything
    # else via our code before other envs are set (mainly the gradio).
    from ols.utils import config

    cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
    config.init_config(cfg_file)

    configure_hugging_face_envs(config.ols_config)

    uvicorn.run("ols.app.main:app", host="127.0.0.1", port=8080, reload=True)
