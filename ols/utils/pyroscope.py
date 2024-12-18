"""Pyroscope handling utility functions."""

import logging
import threading
from typing import Any

import requests

from ols.runners.uvicorn import start_uvicorn


def start_with_pyroscope_enabled(
    config: Any,
    logger: logging.Logger,
) -> None:
    """Start the application using pyroscope."""
    try:

        response = requests.get(config.dev_config.pyroscope_url, timeout=60)
        if response.status_code == requests.codes.ok:
            logger.info(
                "Pyroscope server is reachable at %s", config.dev_config.pyroscope_url
            )
            import pyroscope

            pyroscope.configure(
                application_name="lightspeed-service",
                server_address=config.dev_config.pyroscope_url,
                oncpu=True,
                gil_only=True,
                enable_logging=True,
            )
            with pyroscope.tag_wrapper({"main": "main_method"}):
                # create and start the rag_index_thread
                rag_index_thread = threading.Thread(target=config.rag_index)
                rag_index_thread.start()

                # start the Uvicorn server
                start_uvicorn(config)
        else:
            logger.info(
                "Failed to reach Pyroscope server. Status code: %d",
                response.status_code,
            )
    except requests.exceptions.RequestException as e:
        logger.info("Error connecting to Pyroscope server: %s", str(e))
