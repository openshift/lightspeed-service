"""Utility functions for TLS security profile enforcement."""

import logging
import ssl
from dataclasses import dataclass
from typing import Any, Optional

from ols import constants
from ols.app.models.config import TLSSecurityProfile
from ols.utils import tls

logger = logging.getLogger(__name__)


_LIBPQ_TLS_VERSION_MAP: dict[ssl.TLSVersion, str] = {
    ssl.TLSVersion.TLSv1: "TLSv1",
    ssl.TLSVersion.TLSv1_1: "TLSv1.1",
    ssl.TLSVersion.TLSv1_2: "TLSv1.2",
    ssl.TLSVersion.TLSv1_3: "TLSv1.3",
}


def libpq_tls_params(
    sec_profile: Optional[TLSSecurityProfile],
) -> dict[str, Any]:
    """Return extra libpq connection kwargs enforcing the TLS security profile.

    Maps the OpenShift TLS security profile to libpq's
    ``ssl_min_protocol_version`` parameter.  Returns an empty dict when no
    profile is configured so the caller can simply ``**``-merge it.

    Cipher enforcement is not supported by libpq on the client side —
    cipher negotiation is controlled by the PostgreSQL server's
    ``ssl_ciphers`` setting.
    """
    if sec_profile is None or sec_profile.profile_type is None:
        return {}

    min_version = get_min_tls_version(sec_profile)
    if min_version is None:
        return {}

    libpq_value = _LIBPQ_TLS_VERSION_MAP.get(min_version)
    if libpq_value is None:
        logger.warning("Unmapped TLS version %s, skipping enforcement", min_version)
        return {}

    logger.info("Enforcing Postgres ssl_min_protocol_version=%s", libpq_value)
    return {"ssl_min_protocol_version": libpq_value}


def get_ssl_version(sec_profile: Optional[TLSSecurityProfile]) -> int:
    """Get SSL protocol constant for TLS context creation."""
    logger.info("Using SSL protocol version: %s", ssl.PROTOCOL_TLS_SERVER)
    return ssl.PROTOCOL_TLS_SERVER


def get_min_tls_version(
    sec_profile: Optional[TLSSecurityProfile],
) -> Optional[ssl.TLSVersion]:
    """Get minimum TLS version to enforce on the SSL context."""
    if sec_profile is None or sec_profile.profile_type is None:
        return None

    min_tls_version = tls.min_tls_version(
        sec_profile.min_tls_version, sec_profile.profile_type
    )
    logger.info("min TLS version: %s", min_tls_version)

    resolved_min_tls_version = tls.ssl_tls_version(min_tls_version)
    logger.info("Using minimum TLS version: %s", resolved_min_tls_version)
    return resolved_min_tls_version


def get_ciphers(sec_profile: Optional[TLSSecurityProfile]) -> str:
    """Get allowed ciphers to be used. It can be configured in tls_security_profile section."""
    # if security profile is not set, use default ciphers
    # as specified in SSL library
    if sec_profile is None or sec_profile.profile_type is None:
        logger.info("Allowing default ciphers: %s", constants.DEFAULT_SSL_CIPHERS)
        return constants.DEFAULT_SSL_CIPHERS

    # security profile is set -> we need to retrieve ciphers to be allowed
    ciphers = tls.ciphers_as_string(sec_profile.ciphers, sec_profile.profile_type)
    logger.info("Allowing following ciphers: %s", ciphers)
    return ciphers


@dataclass(frozen=True, slots=True)
class SplitCiphers:
    """TLS 1.2 ciphers and TLS 1.3 ciphersuites separated for OpenSSL APIs."""

    tls12: Optional[str]
    tls13: Optional[str]


def split_ciphers(cipher_string: Optional[str]) -> SplitCiphers:
    """Separate a cipher string into TLS 1.2 ciphers and TLS 1.3 ciphersuites.

    TLS 1.3 ciphersuites are identified by the 'TLS_' prefix per RFC 8446.
    Output uses colon separation as required by OpenSSL APIs.
    Accepts comma-separated, colon-separated, or mixed input.
    """
    if not cipher_string:
        return SplitCiphers(tls12=None, tls13=None)

    tls12: list[str] = []
    tls13: list[str] = []
    for cipher in (c.strip() for c in cipher_string.replace(":", ",").split(",")):
        if not cipher:
            continue
        if cipher.startswith("TLS_"):
            tls13.append(cipher)
        else:
            tls12.append(cipher)

    return SplitCiphers(
        tls12=":".join(tls12) if tls12 else None,
        tls13=":".join(tls13) if tls13 else None,
    )
