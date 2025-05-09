"""Certificates handling utility functions."""

import logging
import os
import shutil
from pathlib import Path

import certifi
from cryptography import x509

import ols.app.models.config as config_model
from ols import constants


def add_ca_to_certificates_store(
    logger: logging.Logger, cert_path: Path, cert_location: str
) -> None:
    """Add a certificate to the certifi store."""
    logger.debug("Certifications store location: %s", cert_location)
    logger.info("Adding certificate '%s' to certificates store", cert_path)

    # load certificate file that needs to be added into store
    with open(cert_path, "rb") as certificate_file:
        new_certificate_data = certificate_file.read()
    new_cert = x509.load_pem_x509_certificate(new_certificate_data)

    # load existing certificates
    with open(cert_location, "rb") as certifi_store:
        certifi_certs_data = certifi_store.read()
    certifi_certs = x509.load_pem_x509_certificates(certifi_certs_data)

    # append the certificate to the certificates store
    if new_cert in certifi_certs:
        logger.warning("Certificate '%s' is already in certificates store", cert_path)
    else:
        with open(cert_location, "ab") as certifi_store:
            certifi_store.write(new_certificate_data)
            logger.debug(
                "Written certificate with length %d bytes", len(new_certificate_data)
            )


def generate_certificates_file(
    logger: logging.Logger, ols_config: config_model.OLSConfig
) -> None:
    """Generate certificates by merging certificates from certify with defined certificates."""
    certificate_directory = ols_config.certificate_directory

    if certificate_directory is None:
        logger.warning(
            "Cannot generate certificate file: certificate directory is not specified"
        )
        return

    logger.info("Generating certificates file into directory %s", certificate_directory)

    # file where all certificates will be stored
    destination_file = os.path.join(
        certificate_directory, constants.CERTIFICATE_STORAGE_FILENAME
    )

    certifi_cert_location = certifi.where()
    logger.debug(
        "Copying certifi certificates file from %s into %s",
        certifi_cert_location,
        destination_file,
    )

    shutil.copyfile(certifi_cert_location, destination_file)

    for certificate_path in ols_config.extra_ca:
        add_ca_to_certificates_store(logger, certificate_path, destination_file)
