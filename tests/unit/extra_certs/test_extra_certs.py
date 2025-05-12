"""Unit tests for extra certs handling."""

import logging
import os
from pathlib import Path

import pytest
from cryptography import x509

import ols.app.models.config as config_model
import ols.constants as constants
from ols.utils.certificates import (
    add_ca_to_certificates_store,
    generate_certificates_file,
)

cert_in_certificates_store_path = "tests/unit/extra_certs/sample_cert_1.crt"
extra_cert_path = Path("tests/unit/extra_certs/sample_cert_2.crt")


@pytest.fixture
def fake_certifi_store(tmpdir):
    """Create a fake certifi store."""
    cert_store_path = tmpdir / "cacerts.crt"
    with open(cert_store_path, "wb") as cert_store:
        with open(cert_in_certificates_store_path, "rb") as cert_file:
            cert_store.write(cert_file.read())
    return cert_store_path.strpath


def load_certs(cert_path):
    """Load a certificates from a file."""
    with open(cert_path, "rb") as cert_file:
        cert_data = cert_file.read()
    return x509.load_pem_x509_certificates(cert_data)


@pytest.fixture
def logger():
    """Logger to be used indirectly in unit tests."""
    return logging.getLogger("ols")


def test_add_ca_to_certificates_store(logger, fake_certifi_store):
    """Test if the certificate is added to the certifi store."""
    add_ca_to_certificates_store(
        logger, extra_cert_path, cert_location=fake_certifi_store
    )

    extra_cert = load_certs(extra_cert_path)[0]
    cert_store_certs = load_certs(fake_certifi_store)

    assert len(cert_store_certs) == 2
    assert extra_cert in cert_store_certs


def test_add_ca_to_certificates_store_no_cert_multiplication(
    logger, fake_certifi_store
):
    """Test if the certificate is not added multiple times to the certifi store."""
    # add the same cert twice
    add_ca_to_certificates_store(
        logger, extra_cert_path, cert_location=fake_certifi_store
    )
    add_ca_to_certificates_store(
        logger, extra_cert_path, cert_location=fake_certifi_store
    )

    extra_cert = load_certs(extra_cert_path)[0]
    cert_store_certs = load_certs(fake_certifi_store)

    assert len(cert_store_certs) == 2
    assert extra_cert in cert_store_certs


def test_add_ca_to_certificates_store_only_appends(logger, fake_certifi_store):
    """Test if the certificate is only appended to the certifi store."""
    add_ca_to_certificates_store(
        logger, extra_cert_path, cert_location=fake_certifi_store
    )

    cert_in_certificates_store = load_certs(cert_in_certificates_store_path)[0]
    extra_cert = load_certs(extra_cert_path)[0]
    cert_store_certs = load_certs(fake_certifi_store)

    assert len(cert_store_certs) == 2
    assert cert_store_certs[0] == cert_in_certificates_store
    assert cert_store_certs[1] == extra_cert


def test_generate_certificates_file(logger, fake_certifi_store, tmpdir):
    """Test the generation of certificates file."""
    ols_config = config_model.OLSConfig()
    ols_config.certificate_directory = tmpdir
    generate_certificates_file(logger, ols_config)

    # check that cert store is created
    final_filename = os.path.join(tmpdir, constants.CERTIFICATE_STORAGE_FILENAME)
    assert os.path.exists(
        final_filename
    ), f"Certificate file {final_filename} was not created"


def test_generate_certificates_file_append_custom_certificate(
    logger, fake_certifi_store, tmpdir
):
    """Test the generation of certificates file with appending custom certificate into it."""
    ols_config = config_model.OLSConfig()
    ols_config.certificate_directory = tmpdir
    ols_config.extra_ca = [extra_cert_path]
    generate_certificates_file(logger, ols_config)

    # check that cert store is created
    final_filename = os.path.join(tmpdir, constants.CERTIFICATE_STORAGE_FILENAME)
    assert os.path.exists(
        final_filename
    ), f"Certificate file {final_filename} was not created"

    # check that extra cert has been added
    cert_store_certs = load_certs(final_filename)
    extra_cert = load_certs(extra_cert_path)[0]

    # our certificate needs to be appended as last one in the store
    assert cert_store_certs[-1] == extra_cert
