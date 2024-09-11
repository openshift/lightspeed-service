"""Unit tests for extra certs handling."""

import logging

import pytest
from cryptography import x509

from runner import add_ca_to_certifi

cert_in_certitifi_store_path = "tests/unit/extra_certs/sample_cert_1.crt"
extra_cert_path = "tests/unit/extra_certs/sample_cert_2.crt"


@pytest.fixture()
def fake_certifi_store(tmpdir):
    """Create a fake certifi store."""
    cert_store_path = tmpdir / "cacerts.crt"
    with open(cert_store_path, "wb") as cert_store:
        with open(cert_in_certitifi_store_path, "rb") as cert_file:
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


def test_add_ca_to_certifi(logger, fake_certifi_store):
    """Test if the certificate is added to the certifi store."""
    add_ca_to_certifi(logger, extra_cert_path, certifi_cert_location=fake_certifi_store)

    extra_cert = load_certs(extra_cert_path)[0]
    cert_store_certs = load_certs(fake_certifi_store)

    assert len(cert_store_certs) == 2
    assert extra_cert in cert_store_certs


def test_add_ca_to_certifi_no_cert_multiplication(logger, fake_certifi_store):
    """Test if the certificate is not added multiple times to the certifi store."""
    # add the same cert twice
    add_ca_to_certifi(logger, extra_cert_path, certifi_cert_location=fake_certifi_store)
    add_ca_to_certifi(logger, extra_cert_path, certifi_cert_location=fake_certifi_store)

    extra_cert = load_certs(extra_cert_path)[0]
    cert_store_certs = load_certs(fake_certifi_store)

    assert len(cert_store_certs) == 2
    assert extra_cert in cert_store_certs


def test_add_ca_to_certifi_only_appends(logger, fake_certifi_store):
    """Test if the certificate is only appended to the certifi store."""
    add_ca_to_certifi(logger, extra_cert_path, certifi_cert_location=fake_certifi_store)

    cert_in_certitifi_store = load_certs(cert_in_certitifi_store_path)[0]
    extra_cert = load_certs(extra_cert_path)[0]
    cert_store_certs = load_certs(fake_certifi_store)

    assert len(cert_store_certs) == 2
    assert cert_store_certs[0] == cert_in_certitifi_store
    assert cert_store_certs[1] == extra_cert
