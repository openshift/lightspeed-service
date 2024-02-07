"""Unit Test for the Utils class."""

import uuid

from ols.utils import suid


def test_get_suid():
    """Test get_suid method."""
    uid = suid.get_suid()
    assert isinstance(uid, str)
    assert uuid.UUID(uid)
