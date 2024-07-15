"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import logging
import os
import tempfile
import threading
from pathlib import Path

import certifi
import uvicorn
from cryptography import x509

from ols import config
from ols.utils.auth_dependency import K8sClientSingleton
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


def load_index():
    """Load the index."""
    # accessing the config's rag_index property will trigger the loading
    # of the index
    config.rag_index


# NOTE: Be aware this is irreversible step, as it modifies the certifi
# store. To restore original certs, the certifi lib needs to be
# reinstalled (or extra cert manually removed from the file).
def add_ca_to_certifi(cert_path: str, certifi_cert_location=certifi.where()) -> None:
    """Add a certificate to the certifi store."""
    print(f"Certifi store location: '{certifi_cert_location}'")
    print(f"Adding certificate '{cert_path}' to certifi store")
    with open(cert_path, "rb") as certificate_file:
        new_certificate_data = certificate_file.read()
    new_cert = x509.load_pem_x509_certificate(new_certificate_data)

    # load certifi certs
    with open(certifi_cert_location, "rb") as certifi_store:
        certifi_certs_data = certifi_store.read()
    certifi_certs = x509.load_pem_x509_certificates(certifi_certs_data)

    # append the certificate to the certifi store
    if new_cert in certifi_certs:
        print(f"Certificate '{cert_path}' is already in certifi store")
    else:
        with open(certifi_cert_location, "ab") as certifi_store:
            certifi_store.write(new_certificate_data)


def start_uvicorn():
    """Start Uvicorn-based REST API service."""
    # use workers=1 so config loaded can be accessed from other modules
    host = (
        "localhost"
        if config.dev_config.run_on_localhost
        else "0.0.0.0"  # noqa: S104 # nosec: B104
    )
    log_level = config.ols_config.logging_config.uvicorn_log_level

    if config.dev_config.disable_tls:
        # TLS is disabled, run without SSL configuration
        uvicorn.run(
            "ols.app.main:app",
            host=host,
            port=8080,
            log_level=log_level,
            workers=1,
            access_log=log_level < logging.INFO,
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
            access_log=log_level < logging.INFO,
        )


if __name__ == "__main__":

    cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
    config.reload_from_yaml_file(cfg_file)

    configure_logging(config.ols_config.logging_config)
    logger = logging.getLogger(__name__)
    logger.info(f"Config loaded from {Path(cfg_file).resolve()}")

    configure_gradio_ui_envs()

    # NOTE: We import config here to avoid triggering import of anything
    # else via our code before other envs are set (mainly the gradio).
    from ols import config

    configure_hugging_face_envs(config.ols_config)

    # add extra certificates if defined
    for certificate_path in config.ols_config.extra_ca:
        add_ca_to_certifi(certificate_path)

    # Initialize the K8sClientSingleton with cluster id during module load.
    # We want the application to fail early if the cluster ID is not available.
    cluster_id = K8sClientSingleton.get_cluster_id()
    logger.info(f"running on cluster with ID '{cluster_id}'")

    # init loading of query redactor
    config.query_redactor

    # create and start the rag_index_thread - allows loading index in
    # parallel with starting the Uvicorn server
    rag_index_thread = threading.Thread(target=load_index)
    rag_index_thread.start()

    # start the Uvicorn server
    start_uvicorn()
