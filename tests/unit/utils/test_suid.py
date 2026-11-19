"""Unit Test for the Utils class."""

import uuid

import pytest

from ols.utils import suid


def test_get_suid():
    """Test get_suid method."""
    uid = suid.get_suid()
    assert isinstance(uid, str)
    assert uuid.UUID(uid)


proper_user_ids = [
    "00000000-0000-0000-0000-000000000000",  # UUID-formatted user ID
    "305070630980550817",  # snowflake ID (OIDC sub mapped to user.uid)
    "sha256~P1lT3example45",  # OpenShift internal OAuth subject format
    "a" * suid.USER_ID_MAX_LENGTH,  # maximal length is accepted
]

improper_user_ids = [
    None,
    "",
    " ",
    "\t",
    "\x00",
    "foo bar",
    "foo\nbar",
    ":",
    "foo:bar",
    # service account UID: contains the compound-key separator (rejected
    # by the previous UUID-based check too, so this is not a regression)
    "system:serviceaccount:openshift-lightspeed:ols",
    "00000000-0000-0000-0000-000000000000:extra",
    "a" * (suid.USER_ID_MAX_LENGTH + 1),
    42,
]


@pytest.mark.parametrize("user_id", proper_user_ids)
def test_check_user_id_proper(user_id):
    """Test that proper user IDs are accepted."""
    assert suid.check_user_id(user_id) is True


@pytest.mark.parametrize("user_id", improper_user_ids)
def test_check_user_id_improper(user_id):
    """Test that improper user IDs are rejected."""
    assert suid.check_user_id(user_id) is False
