"""Unit Test for the Utils class."""

import re

from ols.utils import suid


def test_get_suid():
    """Test get_suid method."""
    uid = suid.get_suid()
    assert isinstance(uid, str)
    assert re.compile("^[a-f0-9]{32}$").match(uid) is not None
