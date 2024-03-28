"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import logging
import os
import tempfile
from pathlib import Path

import uvicorn

from ols.utils import config
from ols.utils.logging import configure_logging


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

    cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")

    config.init_config(cfg_file)

    configure_logging(config.ols_config.logging_config)
    logger = logging.getLogger(__name__)
    logger.info(f"Config loaded from {Path(cfg_file).resolve()}")

    configure_gradio_ui_envs()

    # NOTE: We import config here to avoid triggering import of anything
    # else via our code before other envs are set (mainly the gradio).
    from ols.utils import config

    configure_hugging_face_envs(config.ols_config)
    config.init_query_filter()
    config.init_vector_index()

    host = "localhost" if config.dev_config.run_on_localhost else None
    log_level = config.ols_config.logging_config.uvicorn_log_level

    # use workers=1 so config loaded can be accessed from other modules
    if config.dev_config.disable_tls:
        # TLS is disabled, run without SSL configuration
        uvicorn.run(
            "ols.app.main:app", host=host, port=8080, log_level=log_level, workers=1
        )
    else:
        uvicorn.run(
            "ols.app.main:app",
            host=host,
            port=8443,
            workers=1,
            log_level=log_level,
            ssl_keyfile=config.ols_config.tls_config.tls_key_path,
            ssl_certfile=config.ols_config.tls_config.tls_certificate_path,
            ssl_keyfile_password=config.ols_config.tls_config.tls_key_password,
        )
