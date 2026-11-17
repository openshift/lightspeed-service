"""Unit tests for suid utilities."""

import uuid

from ols.utils import suid

ENTRA_LIKE_SUB = "Pt6XrojL4L5iRK_MN8XCbpQNG5A-AUlhAM8SghjnI7E"


def test_normalize_user_uid_preserves_uuid():
    """UUID-shaped uids must not change (kube:admin cluster id, Entra oid, etc.)."""
    existing = "550e8400-e29b-41d4-a716-446655440000"
    assert suid.normalize_user_uid(existing) == existing


def test_normalize_user_uid_maps_opaque_subject_to_uuid():
    """Opaque TokenReview uid becomes a stable RFC-4122 string."""
    normalized = suid.normalize_user_uid(ENTRA_LIKE_SUB)
    assert suid.check_suid(normalized)
    assert suid.normalize_user_uid(ENTRA_LIKE_SUB) == normalized


def test_normalize_user_uid_uses_uuid5_namespace():
    """Mapping is deterministic and namespaced."""
    namespace = uuid.uuid5(
        uuid.NAMESPACE_URL, "https://redhat.com/openshift-lightspeed/user-uid/v1"
    )
    expected = str(uuid.uuid5(namespace, ENTRA_LIKE_SUB))
    assert suid.normalize_user_uid(ENTRA_LIKE_SUB) == expected
