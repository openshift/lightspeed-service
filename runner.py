"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytz

from ols.constants import (
    CONFIGURATION_DUMP_FILE_NAME,
    CONFIGURATION_FILE_NAME_ENV_VARIABLE,
    DEFAULT_CONFIGURATION_FILE,
)
from ols.runners.quota_scheduler import start_quota_scheduler
from ols.runners.uvicorn import start_uvicorn
from ols.src.auth.auth import use_k8s_auth
from ols.utils import suid
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


def store_config(cfg_file: str, logger: logging.Logger, config) -> None:
    """Store service configuration in the local filesystem.

    This function stores the original configuration file content once at startup.
    Since the configuration is immutable for a single service deployment,
    this avoids duplicating the same config data in every transcript/feedback.

    Args:
        cfg_file: Path to the original configuration file.
        logger: Logger instance to use for logging.
        config: The configuration object.
    """
    # snyk:ignore:a5ed58a6-47dc-4ff3-a430-5a040df0dc12
    with open(cfg_file, "r", encoding="utf-8") as f:
        config_content = f.read()

    data_to_store = {
        "metadata": {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "service_version": __version__,
            "config_file_path": cfg_file,
            # Identifier for current backend, can be omitted in lcore version
            "backend": "lightspeed-service",
        },
        "configuration": config_content,
    }

    # Store the data in the local filesystem
    storage_path = Path(config.ols_config.user_data_collection.config_storage)
    storage_path.mkdir(parents=True, exist_ok=True)
    config_file_path = storage_path / f"{suid.get_suid()}.json"

    with open(config_file_path, "w", encoding="utf-8") as config_file:
        json.dump(data_to_store, config_file, indent=2)

    logger.info("service configuration stored in '%s'", config_file_path)


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
        print(f"Dumping configuration into {CONFIGURATION_DUMP_FILE_NAME}")
        with open(CONFIGURATION_DUMP_FILE_NAME, "w", encoding="utf-8") as fout:
            fout.write(config.config.model_dump_json(indent=4))
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

    # store service configuration if enabled
    if not config.ols_config.user_data_collection.config_disabled:
        store_config(cfg_file, logger, config)
    else:
        logger.debug("config collection is disabled in configuration")

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
