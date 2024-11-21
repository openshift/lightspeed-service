"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import logging
import os
import threading
from pathlib import Path

import uvicorn

import ols.app.models.config as config_model
from ols.src.auth.auth import use_k8s_auth
from ols.utils import ssl
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


def start_uvicorn():
    """Start Uvicorn-based REST API service."""
    # use workers=1 so config loaded can be accessed from other modules
    host = (
        "localhost"
        if config.dev_config.run_on_localhost
        else "0.0.0.0"  # noqa: S104 # nosec: B104
    )
    port = 8080 if config.dev_config.disable_tls else 8443
    log_level = config.ols_config.logging_config.uvicorn_log_level

    # The tls fields can be None, which means we will pass those values as None to uvicorn.run
    ssl_keyfile = config.ols_config.tls_config.tls_key_path
    ssl_certfile = config.ols_config.tls_config.tls_certificate_path
    ssl_keyfile_password = config.ols_config.tls_config.tls_key_password

    # setup SSL version and allowed SSL ciphers based on service configuration
    # when TLS security profile is not specified, default values will be used
    # that default values are based on default SSL package settings
    sec_profile = config.ols_config.tls_security_profile
    ssl_version = ssl.get_ssl_version(sec_profile)
    ssl_ciphers = ssl.get_ciphers(sec_profile)

    uvicorn.run(
        "ols.app.main:app",
        host=host,
        port=port,
        workers=config.ols_config.max_workers,
        log_level=log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_version=ssl_version,
        ssl_ciphers=ssl_ciphers,
        access_log=log_level < logging.INFO,
    )


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
    start_uvicorn()
