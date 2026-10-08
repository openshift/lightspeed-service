"""Create or apply e2e OLSConfig resources with inspection set by suite suffix."""

import os
from typing import Literal

import yaml

from tests.e2e.utils import cluster as cluster_utils


def inspection_enabled_for_e2e(suffix: str) -> bool:
    """Return whether the e2e suffix enables tool-result inspection."""
    return suffix == "mcp_inspection"


def apply_e2e_olsconfig(
    path: str,
    operation: Literal["apply", "create"],
    *,
    suffix: str,
    substitute_environment: bool,
    ignore_existing_resource: bool,
) -> None:
    """Create or apply an OLSConfig CR with the e2e inspection setting."""
    with open(path, encoding="utf-8") as config_file:
        config_yaml = config_file.read()
    if substitute_environment:
        config_yaml = os.path.expandvars(config_yaml)

    olsconfig = yaml.safe_load(config_yaml)
    inspection_config = (
        olsconfig.setdefault("spec", {})
        .setdefault("ols", {})
        .setdefault("guardrails", {})
        .setdefault("toolResultInspection", {})
    )
    inspection_config["enabled"] = inspection_enabled_for_e2e(suffix)

    cluster_utils.run_oc(
        [operation, "-f", "-"],
        command=yaml.safe_dump(olsconfig),
        ignore_existing_resource=ignore_existing_resource,
    )
