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

from ols import config  # pylint: disable=C0413
from ols.constants import (
    CONFIGURATION_FILE_NAME_ENV_VARIABLE,
    DEFAULT_CONFIGURATION_FILE,
)

# it is needed to read proper configuration in order to start the app to generate schema
cfg_file = os.environ.get(
    CONFIGURATION_FILE_NAME_ENV_VARIABLE, DEFAULT_CONFIGURATION_FILE
)
config.reload_from_yaml_file(cfg_file)

from ols.app.main import app  # noqa: E402  pylint: disable=C0413

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
    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(open_api, fout, indent=4)
