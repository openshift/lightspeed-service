"""Unit tests for TLS security profiles manipulation."""

import pytest

from ols import constants
from ols.app.models.config import TLSSecurityProfile
from ols.utils import ssl, tls


def test_get_ssl_version_no_security_profile():
    """Check the function to get SSL version when security profile is not provided."""
    assert ssl.get_ssl_version(None) == constants.DEFAULT_SSL_VERSION


def test_get_ssl_version_no_security_profile_type():
    """Check the function to get SSL version when security profile type is not provided."""
    security_profile = TLSSecurityProfile()
    security_profile.profile_type = None
    assert ssl.get_ssl_version(security_profile) == constants.DEFAULT_SSL_VERSION


tls_profile_to_min_version = (
    ("OldType", "VersionTLS10"),
    ("IntermediateType", "VersionTLS12"),
    ("ModernType", "VersionTLS13"),
)


@pytest.mark.parametrize("tls_profile_to_min_version", tls_profile_to_min_version)
def test_get_ssl_version_with_proper_security_profile(tls_profile_to_min_version):
    """Check the function to get SSL version when security profile type is provided."""
    security_profile = TLSSecurityProfile()
    security_profile.profile_type = tls_profile_to_min_version[0]
    ssl_version = ssl.get_ssl_version(security_profile)
    assert ssl_version is not None
    assert ssl_version == tls.ssl_tls_version(tls_profile_to_min_version[1])


def test_get_ciphers_no_security_profile():
    """Check the function to get SSL ciphers when security profile is not provided."""
    assert ssl.get_ciphers(None) == constants.DEFAULT_SSL_CIPHERS


def test_get_ciphers_no_security_profile_type():
    """Check the function to get SSL ciphers when security profile type is not provided."""
    security_profile = TLSSecurityProfile()
    security_profile.profile_type = None
    assert ssl.get_ciphers(security_profile) == constants.DEFAULT_SSL_CIPHERS


tls_profile_names = (
    "OldType",
    "IntermediateType",
    "ModernType",
)


@pytest.mark.parametrize("tls_profile_name", tls_profile_names)
def test_get_ciphers_with_proper_security_profile(tls_profile_name):
    """Check the function to get SSL ciphers when security profile type is provided."""
    security_profile = TLSSecurityProfile()
    security_profile.profile_type = tls_profile_name
    security_profile.ciphers = None
    allowed_ciphers = ssl.get_ciphers(security_profile)
    assert allowed_ciphers is not None
    assert allowed_ciphers == tls.ciphers_for_tls_profile(tls_profile_name)
