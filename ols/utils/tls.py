"""TLS-related data structures and constants."""

# For further information please look at TLS security profiles source:
# wokeignore:rule=master
# https://github.com/openshift/api/blob/master/config/v1/types_tlssecurityprofile.go

from enum import StrEnum


class TLSProfiles(StrEnum):
    """TLS profile names."""

    OLD_TYPE = "OldType"
    INTERMEDIATE_TYPE = "IntermediateType"
    MODERN_TYPE = "ModernType"


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
