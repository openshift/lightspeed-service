"""Test that the project version is set consistently."""

import json
import subprocess

import semantic_version

from ols import config, version


def read_version_from_openapi():
    """Read version from OpenAPI.json file."""
    # retrieve pre-generated OpenAPI schema
    with open("docs/openapi.json", encoding="utf-8") as fin:
        pre_generated_schema = json.load(fin)
        assert pre_generated_schema is not None
        assert "info" in pre_generated_schema, "node 'info' not found in openapi.json"
        info = pre_generated_schema["info"]
        assert "version" in info, "node 'version' not found in 'info'"
        return info["version"]


def read_version_from_pyproject():
    """Read version from pyproject.toml file."""
    # it is not safe to just try to read version from pyproject.toml file directly
    # the uv tool itself is able to retrieve the version, even if the version
    # is generated dynamically
    completed = subprocess.run(
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('ols'))",
        ],
        capture_output=True,
        check=True,
    )
    return completed.stdout.decode("utf-8").strip()


def read_version_from_app():
    """Read version from app object."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    # app.main need to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    assert app.version is not None
    return app.version


def check_semantic_version(value):
    """Check that the value contains semantic version."""
    # we just need to parse the value, that's all
    semantic_version.Version(value)


def test_project_version_consistency():
    """Test than the project version is set consistently."""
    # read the "true" version defined in sources
    version_from_sources = version.__version__
    check_semantic_version(version_from_sources)

    # OpenAPI endpoint should contain version number
    openapi_version = read_version_from_openapi()
    check_semantic_version(openapi_version)

    # version is dynamically put into pyproject.toml
    project_version = read_version_from_pyproject()
    check_semantic_version(project_version)

    # version is set into app object
    app_version = read_version_from_app()
    check_semantic_version(app_version)

    # compare all versions for equality
    assert version_from_sources == openapi_version
    assert openapi_version == project_version
    assert project_version == app_version
