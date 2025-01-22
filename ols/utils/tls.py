"""TLS-related data structures and constants."""

# For further information please look at TLS security profiles source:
# wokeignore:rule=master
# https://github.com/openshift/api/blob/master/config/v1/types_tlssecurityprofile.go

import logging
import ssl
from enum import StrEnum
from typing import Optional

logger = logging.getLogger(__name__)


class TLSProfiles(StrEnum):
    """TLS profile names."""

    OLD_TYPE = "OldType"
    INTERMEDIATE_TYPE = "IntermediateType"
    MODERN_TYPE = "ModernType"
    CUSTOM_TYPE = "Custom"


class TLSProtocolVersion(StrEnum):
    """TLS protocol versions."""

    # version 1.0 of the TLS security protocol.
    VERSION_TLS_10 = "VersionTLS10"
    # version 1.1 of the TLS security protocol.
    VERSION_TLS_11 = "VersionTLS11"
    # version 1.2 of the TLS security protocol.
    VERSION_TLS_12 = "VersionTLS12"
    # version 1.3 of the TLS security protocol.
    VERSION_TLS_13 = "VersionTLS13"


# Minimal TLS versions required for each TLS profile
MIN_TLS_VERSIONS = {
    TLSProfiles.OLD_TYPE: TLSProtocolVersion.VERSION_TLS_10,
    TLSProfiles.INTERMEDIATE_TYPE: TLSProtocolVersion.VERSION_TLS_12,
    TLSProfiles.MODERN_TYPE: TLSProtocolVersion.VERSION_TLS_13,
}

# TLS ciphers defined for each TLS profile
TLS_CIPHERS = {
    TLSProfiles.OLD_TYPE: [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "ECDHE-RSA-CHACHA20-POLY1305",
        "DHE-RSA-AES128-GCM-SHA256",
        "DHE-RSA-AES256-GCM-SHA384",
        "DHE-RSA-CHACHA20-POLY1305",
        "ECDHE-ECDSA-AES128-SHA256",
        "ECDHE-RSA-AES128-SHA256",
        "ECDHE-ECDSA-AES128-SHA",
        "ECDHE-RSA-AES128-SHA",
        "ECDHE-ECDSA-AES256-SHA384",
        "ECDHE-RSA-AES256-SHA384",
        "ECDHE-ECDSA-AES256-SHA",
        "ECDHE-RSA-AES256-SHA",
        "DHE-RSA-AES128-SHA256",
        "DHE-RSA-AES256-SHA256",
        "AES128-GCM-SHA256",
        "AES256-GCM-SHA384",
        "AES128-SHA256",
        "AES256-SHA256",
        "AES128-SHA",
        "AES256-SHA",
        "DES-CBC3-SHA",
    ],
    TLSProfiles.INTERMEDIATE_TYPE: [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "ECDHE-RSA-CHACHA20-POLY1305",
        "DHE-RSA-AES128-GCM-SHA256",
        "DHE-RSA-AES256-GCM-SHA384",
    ],
    TLSProfiles.MODERN_TYPE: [
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
    ],
}


def ssl_tls_version(
    tls_protocol_version: Optional[TLSProtocolVersion],
) -> Optional[ssl.TLSVersion]:
    """Convert script with TLS protocol version into its numeric code."""
    tls_versions = {
        TLSProtocolVersion.VERSION_TLS_10: ssl.TLSVersion.TLSv1,
        TLSProtocolVersion.VERSION_TLS_11: ssl.TLSVersion.TLSv1_1,
        TLSProtocolVersion.VERSION_TLS_12: ssl.TLSVersion.TLSv1_2,
        TLSProtocolVersion.VERSION_TLS_13: ssl.TLSVersion.TLSv1_3,
    }
    return tls_versions.get(tls_protocol_version, None)


def min_tls_version(
    specified_tls_version: Optional[str],
    tls_profile: TLSProfiles,
) -> Optional[ssl.TLSVersion]:
    """Retrieve minimal TLS version for the profile or for the current profile configuration."""
    min_tls_version_specified = specified_tls_version
    if min_tls_version_specified is None:
        return MIN_TLS_VERSIONS[tls_profile]
    return min_tls_version_specified


def ciphers_from_list(ciphers: Optional[list[str]]) -> Optional[str]:
    """Convert list of ciphers into one string to be consumable by SSL context."""
    if ciphers is None:
        return None
    return ", ".join(ciphers)


def ciphers_for_tls_profile(tls_profile: TLSProfiles) -> Optional[str]:
    """Retrieve list of ciphers for specified TLS profile."""
    ciphers = TLS_CIPHERS.get(tls_profile, None)
    return ciphers_from_list(ciphers)


def ciphers_as_string(
    ciphers: Optional[list[str]], tls_profile: TLSProfiles
) -> Optional[str]:
    """Retrieve ciphers as one string for custom list of TLS profile-based list."""
    ciphers_as_str = ciphers_from_list(ciphers)
    if ciphers_as_str is None:
        return ciphers_for_tls_profile(tls_profile)
    return ciphers_as_str
