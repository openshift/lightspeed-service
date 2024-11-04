"""Unit tests for TLS security profiles manipulation."""

import ssl

import pytest

from ols.utils import tls

tls_versions = (
    ("VersionTLS10", ssl.TLSVersion.TLSv1),
    ("VersionTLS11", ssl.TLSVersion.TLSv1_1),
    ("VersionTLS12", ssl.TLSVersion.TLSv1_2),
    ("VersionTLS13", ssl.TLSVersion.TLSv1_3),
)


@pytest.mark.parametrize("tls_version", tls_versions)
def test_ssl_tls_version(tls_version):
    """Check the function ssl_tls_version."""
    assert tls.ssl_tls_version(tls_version[0]) == tls_version[1]


tls_unknown_versions = (
    "VersionTLS",
    "VersionTLS9",
    "VersionTLS14",
    "foo",
    "bar",
    "None",
)


@pytest.mark.parametrize("tls_version", tls_unknown_versions)
def test_ssl_tls_version_unknown_input(tls_version):
    """Check the function ssl_tls_version."""
    assert tls.ssl_tls_version(tls_version) is None


def test_ssl_tls_version_null_input():
    """Check the function ssl_tls_version."""
    assert tls.ssl_tls_version(None) is None


tls_profile_to_min_version = (
    ("OldType", "VersionTLS10"),
    ("IntermediateType", "VersionTLS12"),
    ("ModernType", "VersionTLS13"),
)


@pytest.mark.parametrize("tls_profile_to_min_version", tls_profile_to_min_version)
def test_min_tls_version_tls_profile(tls_profile_to_min_version):
    """Check the function min_tls_version."""
    assert (
        tls.min_tls_version(None, tls_profile_to_min_version[0])
        == tls_profile_to_min_version[1]
    )


@pytest.mark.parametrize("tls_version", tls_versions)
def test_min_tls_version(tls_version):
    """Check the function min_tls_version."""
    assert tls.min_tls_version(tls_version, None) == tls_version


def test_ciphers_from_list():
    """Check the function ciphers_from_list."""
    ciphers = [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
    ]
    expected = (
        "TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256"
    )
    assert tls.ciphers_from_list(ciphers) == expected


def test_ciphers_from_list_empty_input():
    """Check the function ciphers_from_list for empty input."""
    ciphers = []
    expected = ""
    assert tls.ciphers_from_list(ciphers) == expected


def test_ciphers_from_list_null_input():
    """Check the function ciphers_from_list for null input."""
    ciphers = None
    assert tls.ciphers_from_list(ciphers) is None


ciphers_for_tls_profile = (
    (
        "OldType",
        (
            "TLS_AES_128_GCM_SHA256",
            "ECDHE-ECDSA-AES128-GCM-SHA256",
            "AES128-GCM-SHA256",
        ),
    ),
    (
        "IntermediateType",
        (
            "TLS_AES_128_GCM_SHA256",
            "ECDHE-ECDSA-CHACHA20-POLY1305",
            "DHE-RSA-AES256-GCM-SHA384",
        ),
    ),
    (
        "ModernType",
        (
            "TLS_AES_128_GCM_SHA256",
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
        ),
    ),
)


@pytest.mark.parametrize("ciphers_for_tls_profile", ciphers_for_tls_profile)
def test_ciphers_for_tls_profile_known_profile(ciphers_for_tls_profile):
    """Check the function ciphers_for_tls_profile."""
    ciphers = tls.ciphers_for_tls_profile(ciphers_for_tls_profile[0])
    assert ciphers is not None

    expected_ciphers = ciphers_for_tls_profile[1]
    for expected_cipher in expected_ciphers:
        assert expected_cipher in ciphers


def test_ciphers_for_tls_profile_custom_profile():
    """Check the function ciphers_for_tls_profile."""
    ciphers = tls.ciphers_for_tls_profile("Custom")
    assert ciphers is None


def test_ciphers_for_tls_profile_unknown_profile():
    """Check the function ciphers_for_tls_profile."""
    ciphers = tls.ciphers_for_tls_profile("UnknownProfile")
    assert ciphers is None


@pytest.mark.parametrize("ciphers_for_tls_profile", ciphers_for_tls_profile)
def test_ciphers_as_string_no_default(ciphers_for_tls_profile):
    """Check the function ciphers_as_string."""
    ciphers = tls.ciphers_as_string(None, ciphers_for_tls_profile[0])
    assert ciphers is not None

    expected_ciphers = ciphers_for_tls_profile[1]
    for expected_cipher in expected_ciphers:
        assert expected_cipher in ciphers


@pytest.mark.parametrize("ciphers_for_tls_profile", ciphers_for_tls_profile)
def test_ciphers_as_string_with_default(ciphers_for_tls_profile):
    """Check the function ciphers_as_string."""
    ciphers = [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
    ]
    ciphers = tls.ciphers_as_string(ciphers, ciphers_for_tls_profile[0])
    assert ciphers is not None

    expected_ciphers = ciphers
    for expected_cipher in expected_ciphers:
        assert expected_cipher in ciphers
