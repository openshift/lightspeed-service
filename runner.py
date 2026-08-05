"""Main app entrypoint. Starts Uvicorn-based REST API service."""

import logging
import os
import sys
from pathlib import Path

from ols.constants import (
    CONFIGURATION_DUMP_FILE_NAME,
    CONFIGURATION_FILE_NAME_ENV_VARIABLE,
    DEFAULT_CONFIGURATION_FILE,
)
from ols.runners.quota_scheduler import start_quota_scheduler
from ols.runners.uvicorn import start_uvicorn
from ols.src.auth.auth import use_k8s_auth
from ols.utils.environments import configure_gradio_ui_envs, configure_hugging_face_envs
from ols.utils.logging_configurator import configure_logging
from ols.utils.otel import init_tracer
from ols.utils.pyroscope import start_with_pyroscope_enabled
from ols.version import __version__

_SYSTEM_CA_BUNDLE = Path("/etc/pki/tls/certs/ca-bundle.crt")

logger = logging.getLogger("ols")


def _ensure_cert_bundle() -> None:
    """Build the CA trust bundle at the path given by SSL_CERT_FILE.

    The operator mounts service CA certs (RHOKP, OTEL, Postgres, MCP, etc.)
    into separate read-only directories under /etc/certs/ and creates an
    empty writable emptyDir for the combined bundle.  This function gathers:

    1. System root CAs (/etc/pki/tls/certs/ca-bundle.crt) for public endpoints.
    2. All .crt/.pem files from sibling directories under the cert root.

    The combined bundle is written to SSL_CERT_FILE so that Python's ssl
    module (and httpx) automatically picks it up for TLS verification.
    """
    cert_file = os.environ.get("SSL_CERT_FILE")
    if not cert_file:
        raise RuntimeError("SSL_CERT_FILE is not set — cannot build CA trust bundle")
    cert_path = Path(cert_file).resolve()
    bundle_dir = cert_path.parent

    parts: list[bytes] = [_SYSTEM_CA_BUNDLE.read_bytes()]
    logger.info("Added system CA bundle from %s", _SYSTEM_CA_BUNDLE)

    cert_root = bundle_dir.parent
    for d in sorted(cert_root.iterdir()):
        if d == bundle_dir or not d.is_dir():
            continue
        parts.extend(
            f.read_bytes()
            for f in sorted(d.iterdir())
            if f.is_file() and f.suffix in (".crt", ".pem")
        )
        logger.info("Added CA certs from %s", d.name)

    cert_path.write_bytes(b"\n".join(parts))


def load_index():
    """Resolve Solr hybrid search and load RAG indexes before accepting requests."""
    config.solr_hybrid_search  # pylint: disable=W0104, E0606
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
        print(f"Dumping configuration into {CONFIGURATION_DUMP_FILE_NAME}")
        with open(CONFIGURATION_DUMP_FILE_NAME, "w", encoding="utf-8") as fout:
            fout.write(config.config.model_dump_json(indent=4))
        sys.exit()

    logger = logging.getLogger("ols")
    configure_logging(config.ols_config.logging_config)
    logger.info("Config loaded from %s", Path(cfg_file).resolve())
    logger.info("Running on Python version %s", sys.version)
    configure_hugging_face_envs()
    _ensure_cert_bundle()

    if use_k8s_auth(config.ols_config):
        logger.info("Initializing k8s auth")
        from ols.src.auth.k8s import K8sClientSingleton

        # Initialize the K8sClientSingleton with cluster id during module load.
        # We want the application to fail early if the cluster ID is not available.
        CLUSTER_ID = K8sClientSingleton.get_cluster_id()
        logger.info("running on cluster with ID '%s'", CLUSTER_ID)

    # init loading of query redactor
    config.query_redactor  # pylint: disable=W0104

    # Let gRPC pick up the same CA bundle as Python ssl via SSL_CERT_FILE
    if ssl_cert := os.environ.get("SSL_CERT_FILE"):
        os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", ssl_cert)

    # Initialize OTEL tracer for audit spans
    audit_cfg = config.ols_config.audit
    init_tracer(
        otel_endpoint=audit_cfg.otel.endpoint if audit_cfg and audit_cfg.otel else None,
        audit_enabled=bool(audit_cfg and audit_cfg.enabled),
    )

    if config.dev_config.pyroscope_url:
        start_with_pyroscope_enabled(config, logger)
    else:
        logger.info(
            "Pyroscope url is not specified. To enable profiling please set `pyroscope_url` "
            "in the `dev_config` section of the configuration file."
        )

    load_index()

    # start the quota scheduler
    start_quota_scheduler(config)

    # start the Uvicorn server
    start_uvicorn(config)
