"""Checks that are performed to configuration options."""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

from pydantic import (
    AnyHttpUrl,
    FilePath,
)


class InvalidConfigurationError(Exception):
    """OLS Configuration is invalid."""


def is_valid_http_url(url: AnyHttpUrl) -> bool:
    """Check is a string is a well-formed http or https URL."""
    result = urlparse(str(url))
    return all([result.scheme, result.netloc]) and result.scheme in {
        "http",
        "https",
    }


def get_attribute_from_file(data: dict, file_name_key: str) -> Optional[str]:
    """Retrieve value of an attribute from a file."""
    file_path = data.get(file_name_key)
    if file_path is not None:
        with open(file_path, encoding="utf-8") as f:
            return f.read().rstrip()
    return None


def read_secret(
    data: dict,
    path_key: str,
    default_filename: str,
    raise_on_error: bool = True,
    directory_name_expected: bool = False,
) -> Optional[str]:
    """Read secret from file on given path or from filename if path points to directory."""
    path = data.get(path_key)

    if path is None:
        return None

    filename = path
    if os.path.isdir(path):
        filename = os.path.join(path, default_filename)
    elif directory_name_expected:
        msg = "Improper credentials_path specified: it must contain path to directory with secrets."
        # no logging configured yet
        print(msg)
        return None

    try:
        with open(filename, encoding="utf-8") as f:
            return f.read().rstrip()
    except OSError as e:
        # some files with secret must exist, so for such cases it is time
        # to inform about improper configuration
        if raise_on_error:
            raise
        # no logging configured yet
        print(f"Problem reading secret from file {filename}:", e)
        print(f"Verify the provider secret contains {default_filename}")
        return None


def dir_check(path: FilePath, desc: str) -> None:
    """Check that path is a readable directory."""
    if not os.path.exists(path):
        raise InvalidConfigurationError(f"{desc} '{path}' does not exist")
    if not os.path.isdir(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a directory")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


def file_check(path: FilePath, desc: str) -> None:
    """Check that path is a readable regular file."""
    if not os.path.isfile(path):
        raise InvalidConfigurationError(f"{desc} '{path}' is not a file")
    if not os.access(path, os.R_OK):
        raise InvalidConfigurationError(f"{desc} '{path}' is not readable")


def get_log_level(value: str) -> int:
    """Get log level from string."""
    if not isinstance(value, str):
        raise InvalidConfigurationError(
            f"'{value}' log level must be string, got {type(value)}"
        )
    log_level = logging.getLevelName(value.upper())
    if not isinstance(log_level, int):
        raise InvalidConfigurationError(
            f"'{value}' is not valid log level, valid levels are "
            f"{[k.lower() for k in logging.getLevelNamesMapping()]}"
        )
    return log_level
