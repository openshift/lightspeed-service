"""Config status collection module."""

from ols.src.config_status.config_status import (
    ConfigStatus,
    extract_config_status,
    store_config_status,
)

__all__ = ["ConfigStatus", "extract_config_status", "store_config_status"]
