"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import logging
import os
import threading
from pathlib import Path

import ols.app.models.config as config_model
from ols.runners.uvicorn import start_uvicorn
from ols.src.auth.auth import use_k8s_auth
from ols.utils.certificates import generate_certificates_file
from ols.utils.environments import configure_gradio_ui_envs
from ols.utils.logging_configurator import configure_logging


def configure_hugging_face_envs(ols_config: config_model.OLSConfig) -> None:
    """Configure HuggingFace library environment variables."""
    if (
        ols_config
        and hasattr(ols_config, "reference_content")
        and hasattr(ols_config.reference_content, "embeddings_model_path")
        and ols_config.reference_content.embeddings_model_path
    ):
        os.environ["TRANSFORMERS_CACHE"] = str(
            ols_config.reference_content.embeddings_model_path
        )
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_index():
    """Load the index."""
    # accessing the config's rag_index property will trigger the loading
    # of the index
    config.rag_index


if __name__ == "__main__":
    # First of all, configure environment variables for Gradio before
    # import config and initializing config module.
    configure_gradio_ui_envs()

    # NOTE: We import config here to avoid triggering import of anything
    # else via our code before other envs are set (mainly the gradio).
    from ols import config

    cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
    config.reload_from_yaml_file(cfg_file)

    configure_logging(config.ols_config.logging_config)
    logger = logging.getLogger("ols")
    logger.info("Config loaded from %s", Path(cfg_file).resolve())

    configure_hugging_face_envs(config.ols_config)

    # generate certificates file from all certificates from certifi package
    # merged with explicitly specified certificates
    generate_certificates_file(logger, config.ols_config)

    if use_k8s_auth(config.ols_config):
        logger.info("Initializing k8s auth")
        from ols.src.auth.k8s import K8sClientSingleton

        # Initialize the K8sClientSingleton with cluster id during module load.
        # We want the application to fail early if the cluster ID is not available.
        cluster_id = K8sClientSingleton.get_cluster_id()
        logger.info("running on cluster with ID '%s'", cluster_id)

    # init loading of query redactor
    config.query_redactor

    # create and start the rag_index_thread - allows loading index in
    # parallel with starting the Uvicorn server
    rag_index_thread = threading.Thread(target=load_index)
    rag_index_thread.start()

    # start the Uvicorn server
    start_uvicorn(config)
