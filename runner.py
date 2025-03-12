"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import logging
import os
import sys
import threading
from pathlib import Path

from ols.constants import (
    CONFIGURATION_FILE_NAME_ENV_VARIABLE,
    DEFAULT_CONFIGURATION_FILE,
)
from ols.runners.quota_scheduler import start_quota_scheduler
from ols.runners.uvicorn import start_uvicorn
from ols.src.auth.auth import use_k8s_auth
from ols.utils.certificates import generate_certificates_file
from ols.utils.environments import configure_gradio_ui_envs, configure_hugging_face_envs
from ols.utils.logging_configurator import configure_logging
from ols.utils.pyroscope import start_with_pyroscope_enabled
from ols.version import __version__


def load_index():
    """Load the index."""
    # accessing the config's rag_index property will trigger the loading
    # of the index
    config.rag_index  # pylint: disable=W0104, E0606


if __name__ == "__main__":
    if "--version" in sys.argv:
        print(__version__)
        sys.exit()

    # First of all, configure environment variables for Gradio before
    # import config and initializing config module.
    configure_gradio_ui_envs()

    # NOTE: We import config here to avoid triggering import of anything
    # else via our code before other envs are set (mainly the gradio).
    from ols import config

    cfg_file = os.environ.get(
        CONFIGURATION_FILE_NAME_ENV_VARIABLE, DEFAULT_CONFIGURATION_FILE
    )
    config.reload_from_yaml_file(cfg_file)

    if "--dump-config" in sys.argv:
        with open("olsconfig.json", "w", encoding="utf-8") as fout:
            fout.write(config.config.json())
        sys.exit()

    logger = logging.getLogger("ols")
    configure_logging(config.ols_config.logging_config)
    logger.info("Config loaded from %s", Path(cfg_file).resolve())
    logger.info("Running on Python version %s", sys.version)
    configure_hugging_face_envs(config.ols_config)

    # generate certificates file from all certificates from certifi package
    # merged with explicitly specified certificates
    generate_certificates_file(logger, config.ols_config)

    if use_k8s_auth(config.ols_config):
        logger.info("Initializing k8s auth")
        from ols.src.auth.k8s import K8sClientSingleton

        # Initialize the K8sClientSingleton with cluster id during module load.
        # We want the application to fail early if the cluster ID is not available.
        CLUSTER_ID = K8sClientSingleton.get_cluster_id()
        logger.info("running on cluster with ID '%s'", CLUSTER_ID)

    # init loading of query redactor
    config.query_redactor  # pylint: disable=W0104

    if config.dev_config.pyroscope_url:
        start_with_pyroscope_enabled(config, logger)
    else:
        logger.info(
            "Pyroscope url is not specified. To enable profiling please set `pyroscope_url` "
            "in the `dev_config` section of the configuration file."
        )
    # create and start the rag_index_thread - allows loading index in
    # parallel with starting the Uvicorn server
    rag_index_thread = threading.Thread(target=load_index)
    rag_index_thread.start()

    # start the quota scheduler
    start_quota_scheduler(config)

    # start the Uvicorn server
    start_uvicorn(config)
