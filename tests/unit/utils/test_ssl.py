"""Unit tests for TLS security profiles manipulation."""

import ssl as stdlib_ssl

import pytest
from psycopg2 import extensions

from ols import constants
from ols.app.models.config import TLSSecurityProfile
from ols.utils import ssl as ssl_utils
from ols.utils import tls
from ols.utils.ssl import SplitCiphers, split_ciphers


def test_postgres_ssl_mode_default_is_require():
    """Verify the default Postgres SSL mode is 'require', not 'prefer'."""
    assert constants.POSTGRES_CACHE_SSL_MODE == "require"


def test_get_ssl_version_returns_protocol_constant():
    """Check the function to get SSL version."""
    assert ssl_utils.get_ssl_version(None) == constants.DEFAULT_SSL_VERSION


def test_get_min_tls_version_no_security_profile():
    """Check the function to get minimum TLS version when profile is absent."""
    assert ssl_utils.get_min_tls_version(None) is None


def test_get_min_tls_version_no_security_profile_type():
    """Check the function to get minimum TLS version when profile type is absent."""
    security_profile = TLSSecurityProfile()
    security_profile.profile_type = None
    assert ssl_utils.get_min_tls_version(security_profile) is None


tls_profile_to_min_version = (
    ("OldType", stdlib_ssl.TLSVersion.TLSv1),
    ("IntermediateType", stdlib_ssl.TLSVersion.TLSv1_2),
    ("ModernType", stdlib_ssl.TLSVersion.TLSv1_3),
)


@pytest.mark.parametrize("tls_profile_to_min_version", tls_profile_to_min_version)
def test_get_min_tls_version_with_proper_security_profile(tls_profile_to_min_version):
    """Check the function to get minimum TLS version for each security profile."""
    security_profile = TLSSecurityProfile()
    security_profile.profile_type = tls_profile_to_min_version[0]
    ssl_version = ssl_utils.get_min_tls_version(security_profile)
    assert ssl_version == tls_profile_to_min_version[1]


def test_get_ciphers_no_security_profile():
    """Check the function to get SSL ciphers when security profile is not provided."""
    assert ssl_utils.get_ciphers(None) == constants.DEFAULT_SSL_CIPHERS


def test_get_ciphers_no_security_profile_type():
    """Check the function to get SSL ciphers when security profile type is not provided."""
    security_profile = TLSSecurityProfile()
    security_profile.profile_type = None
    assert ssl_utils.get_ciphers(security_profile) == constants.DEFAULT_SSL_CIPHERS


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
    allowed_ciphers = ssl_utils.get_ciphers(security_profile)
    assert allowed_ciphers is not None
    assert allowed_ciphers == tls.ciphers_for_tls_profile(tls_profile_name)


class TestLibpqTlsParams:
    """Tests for the libpq_tls_params helper."""

    def test_returns_empty_when_profile_is_none(self):
        """Return empty dict when no TLS security profile is provided."""
        assert ssl_utils.libpq_tls_params(None) == {}

    def test_returns_empty_when_profile_type_is_none(self):
        """Return empty dict when profile exists but profile_type is unset."""
        profile = TLSSecurityProfile()
        profile.profile_type = None
        assert ssl_utils.libpq_tls_params(profile) == {}

    @pytest.mark.parametrize(
        ("profile_type", "expected_version"),
        [
            ("IntermediateType", "TLSv1.2"),
            ("ModernType", "TLSv1.3"),
        ],
    )
    def test_maps_profile_to_libpq_version(self, profile_type, expected_version):
        """Verify the profile maps to the correct libpq version string."""
        profile = TLSSecurityProfile()
        profile.profile_type = profile_type
        params = ssl_utils.libpq_tls_params(profile)
        assert params == {"ssl_min_protocol_version": expected_version}

    def test_result_can_be_merged_into_connect_kwargs(self):
        """Verify the returned dict produces a valid libpq DSN."""
        profile = TLSSecurityProfile()
        profile.profile_type = "IntermediateType"
        params = ssl_utils.libpq_tls_params(profile)
        dsn = extensions.make_dsn(
            host="127.0.0.1", dbname="test", sslmode="require", **params
        )
        assert "ssl_min_protocol_version=TLSv1.2" in dsn


class TestSplitCiphers:
    """Tests for the split_ciphers helper."""

    def test_none_input(self):
        """Return both fields as None when input is None."""
        result = split_ciphers(None)
        assert result == SplitCiphers(tls12=None, tls13=None)

    def test_empty_string(self):
        """Return both fields as None when input is empty."""
        result = split_ciphers("")
        assert result == SplitCiphers(tls12=None, tls13=None)

    def test_tls13_only(self):
        """ModernType: all ciphers are TLS 1.3 ciphersuites."""
        cipher_str = (
            "TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384, "
            "TLS_CHACHA20_POLY1305_SHA256"
        )
        result = split_ciphers(cipher_str)
        assert result.tls12 is None
        assert result.tls13 == (
            "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:"
            "TLS_CHACHA20_POLY1305_SHA256"
        )

    def test_tls12_only(self):
        """Custom profile with only TLS 1.2 ciphers."""
        cipher_str = "ECDHE-RSA-AES128-GCM-SHA256, DHE-RSA-AES256-GCM-SHA384"
        result = split_ciphers(cipher_str)
        assert result.tls12 == "ECDHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384"
        assert result.tls13 is None

    def test_mixed_ciphers(self):
        """IntermediateType: mix of TLS 1.2 and TLS 1.3."""
        cipher_str = (
            "TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384, "
            "TLS_CHACHA20_POLY1305_SHA256, "
            "ECDHE-ECDSA-AES128-GCM-SHA256, ECDHE-RSA-AES128-GCM-SHA256"
        )
        result = split_ciphers(cipher_str)
        assert result.tls12 == (
            "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256"
        )
        assert result.tls13 == (
            "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:"
            "TLS_CHACHA20_POLY1305_SHA256"
        )

    def test_colon_separated_input(self):
        """Accept colon-separated input (OpenSSL native format)."""
        cipher_str = "TLS_AES_128_GCM_SHA256:ECDHE-RSA-AES128-GCM-SHA256"
        result = split_ciphers(cipher_str)
        assert result.tls12 == "ECDHE-RSA-AES128-GCM-SHA256"
        assert result.tls13 == "TLS_AES_128_GCM_SHA256"
