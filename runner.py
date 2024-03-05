"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import os

import uvicorn


def configure_hugging_face_envs() -> None:
    """Configure HuggingFace library environment variables."""
    # NOTE: We import config here to avoid triggering import of anything
    # else via our code before other envs are set (mainly the gradio).
    from ols.utils import config

    cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
    config.init_config(cfg_file)
    if (
        config.ols_config
        and hasattr(config.ols_config, "reference_content")
        and hasattr(config.ols_config.reference_content, "embeddings_model_path")
        and config.ols_config.reference_content.embeddings_model_path
    ):
        os.environ["TRANSFORMERS_CACHE"] = (
            config.ols_config.reference_content.embeddings_model_path
        )
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


if __name__ == "__main__":
    # disable Gradio analytics, which calls home to https://api.gradio.app
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

    configure_hugging_face_envs()

    uvicorn.run("ols.app.main:app", host="127.0.0.1", port=8080, reload=True)
