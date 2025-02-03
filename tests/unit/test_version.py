"""Test that the project version is set."""

import semantic_version

from ols import version


def check_semantic_version(value):
    """Check that the value contains semantic version."""
    # we just need to parse the value, that's all
    semantic_version.Version(value)


def test_project_version_consistency():
    """Test than the project version is set consistently."""
    # read the "true" version defined in sources
    version_from_sources = version.__version__
    check_semantic_version(version_from_sources)
