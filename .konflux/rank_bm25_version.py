"""Missing version.py for rank-bm25==0.2.2 sdist; extracted from upstream repo."""

__all__ = "get_version"

import re
from os.path import dirname, join

version_re = re.compile("^Version: (.+)$", re.MULTILINE)


def get_version():
    """Extract version from PKG-INFO."""
    with open(join(dirname(__file__), "PKG-INFO")) as f:
        return version_re.search(f.read()).group(1)
