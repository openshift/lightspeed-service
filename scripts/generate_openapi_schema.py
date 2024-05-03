"""Utility script to generate OpenAPI schema."""

import json
import os.path
import sys

from fastapi.openapi.utils import get_openapi

# we need to import OLS app from directory above, so it is needed to update
# search path accordingly
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from ols.utils.config import ConfigManager

# it is needed to read proper configuration in order to start the app to generate schema
cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
config_manager = ConfigManager()
config_manager.init_config(cfg_file)

from ols.app.main import app  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_openapi_schema.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    # retrieve OpenAPI schema via initialized app
    open_api = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    # dump the schema into file
    with open(filename, "w") as fout:
        json.dump(open_api, fout, indent=4)
