"""Unit Test for the Utils class."""

import re

from ols.app.utils import Utils


def test_get_suid():
    """Test get_suid method."""
    suid = Utils.get_suid()
    assert isinstance(suid, str)
    assert re.compile("^[a-f0-9]{32}$").match(suid) is not None
