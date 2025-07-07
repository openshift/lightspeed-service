"""Functionality for updating OLS_CONFIG to support multiproviders"""

import os
from tests.e2e.utils import cluster
from tests.e2e.utils import wait_for_ols
from ols.constants import DEFAULT_CONFIGURATION_FILE
import yaml

def adapt_ols_config() -> str :
    """
        Helper function to adapt to multiple providers during test job by dynamically replacing
        the existing OLS provider config (olsconfig) with one matching the environment variables during test job execution.
    """
    try:

        provider = os.environ["PROVIDER"]
        model = os.environ["MODEL"]
        key_path = os.environ["PROVIDER_KEY_PATH"]
        provider_url = os.environ["OLS_PROVIDER_RL"]
        configmap_yaml = cluster.run_oc(["get", "cm/olsconfig", "-o", "yaml"]).stdout
        configmap = yaml.safe_load(configmap_yaml)
        olsconfig = yaml.safe_load(configmap["data"][DEFAULT_CONFIGURATION_FILE])

        olsconfig["ols_config"]["logging_config"]["lib_log_level"] = "INFO"

        olsconfig["ols_config"]["providers"] = [{
            "name": provider,
            "type": provider,
            "url": provider_url,
            "credentials_path": key_path,
            "models": [{"name": model}]
        }]

        configmap["data"][DEFAULT_CONFIGURATION_FILE] = yaml.dump(olsconfig)
        updated_configmap = yaml.dump(configmap)

        cluster.run_oc(["delete", "configmap", "olsconfig"])
        cluster.run_oc(["apply", "-f", "-"], command=updated_configmap)

        url = cluster.run_oc(
            ["get", "route", "ols", "-o", "jsonpath='{.spec.host}'"]
        ).stdout.strip("'")

        ols_url = f"https://{url}"
        wait_for_ols.wait_for_ols(ols_url)
        return "OLS operator now available with the updated provider set-up."
    except Exception as e:
        print(f"[ERROR] Failed to adapt OLS config: {e}")
        raise
