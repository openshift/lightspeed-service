"""Utility function for retrieving SSL version and list of ciphers for TLS secutiry profile."""

import logging
from typing import Optional

from ols import constants
from ols.app.models.config import TLSSecurityProfile
from ols.utils import tls

logger = logging.getLogger(__name__)


def get_ssl_version(sec_profile: Optional[TLSSecurityProfile]) -> str:
    """Get SSL version to be used. It can be configured in tls_security_profile section."""
    # if security profile is not set, use default SSL version
    # as specified in SSL library
    if sec_profile is None or sec_profile.profile_type is None:
        logger.info("Using default SSL version: %s", constants.DEFAULT_SSL_VERSION)
        return constants.DEFAULT_SSL_VERSION

    # security profile is set -> we need to retrieve SSL version and list of allowed ciphers
    min_tls_version = tls.min_tls_version(
        sec_profile.min_tls_version, sec_profile.profile_type
    )
    logger.info("min TLS version: %s", min_tls_version)

    ssl_version = tls.ssl_tls_version(min_tls_version)
    logger.info("Using SSL version: %s", ssl_version)
    return ssl_version


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
