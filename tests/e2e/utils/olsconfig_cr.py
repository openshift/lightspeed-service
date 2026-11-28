"""Create or apply e2e OLSConfig resources with inspection set by suite suffix."""

import os
from typing import Literal

import yaml
from openshift.dynamic.exceptions import NotFoundError

from tests.e2e.utils import cluster_api


def inspection_enabled_for_e2e(suffix: str) -> bool:
    """Return whether the e2e suffix enables tool-result inspection."""
    return suffix == "mcp_inspection" or suffix.startswith("mcp_inspection_")


def apply_e2e_olsconfig(
    path: str,
    operation: Literal["apply", "create", "replace_spec"],
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
    olsconfig.setdefault("spec", {}).setdefault("ols", {}).setdefault(
        "logLevel", "INFO"
    )
    inspection_config = (
        olsconfig.setdefault("spec", {})
        .setdefault("ols", {})
        .setdefault("guardrails", {})
        .setdefault("toolResultInspection", {})
    )
    inspection_config["enabled"] = inspection_enabled_for_e2e(suffix)

    if operation == "apply":
        cluster_api.apply_resource(olsconfig)
    elif operation == "replace_spec":
        try:
            cluster_api.patch_resource(
                "olsconfig",
                olsconfig["metadata"]["name"],
                [{"op": "replace", "path": "/spec", "value": olsconfig["spec"]}],
                patch_type="json",
            )
        except NotFoundError:
            cluster_api.create_resource(olsconfig, ignore_existing=False)
    else:
        cluster_api.create_resource(olsconfig, ignore_existing=ignore_existing_resource)
