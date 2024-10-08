"""Test that the project version is set consistently."""

import json
import tomllib

import semantic_version

from ols import config


def read_version_from_openapi():
    """Read version from OpenAPI.json file."""
    # retrieve pre-generated OpenAPI schema
    with open("docs/openapi.json") as fin:
        pre_generated_schema = json.load(fin)
        assert pre_generated_schema is not None
        assert "info" in pre_generated_schema, "node 'info' not found in openapi.json"
        info = pre_generated_schema["info"]
        assert "version" in info, "node 'version' not found in 'info'"
        return info["version"]


def read_version_from_pyproject():
    """Read version from pyproject.toml file."""
    with open("pyproject.toml", "rb") as fin:
        pyproject = tomllib.load(fin)
        assert pyproject is not None
        assert "project" in pyproject, "section [project] is missing in pyproject.toml"
        project = pyproject["project"]
        assert (
            "version" in project
        ), "attribute 'version' is missing in section [project]"
        return project["version"]


def read_version_from_app():
    """Read version from app object."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    # app.main need to be imported after the configuration is read
    from ols.app.main import app

    assert app.version is not None
    return app.version


def check_semantic_version(value):
    """Check that the value contains semantic version."""
    # we just need to parse the value, that's all
    semantic_version.Version(value)


def test_project_version_consistency():
    """Test than the project version is set consistently."""
    openapi_version = read_version_from_openapi()
    check_semantic_version(openapi_version)

    project_version = read_version_from_pyproject()
    check_semantic_version(project_version)

    app_version = read_version_from_app()
    check_semantic_version(app_version)

    assert openapi_version == project_version
    assert project_version == app_version
