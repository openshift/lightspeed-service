"""Mocked K8S SAR,TR Responces, Mocked K8S SAR,TR Requests/Clients."""


class MockK8sResponse:
    """Mock Kubernetes API Response.

    This class is designed to mock Kubernetes API responses for testing purposes.

    """

    def __init__(self, authenticated=None, allowed=None, username=None, uid=None):
        """Init function."""
        self.status = MockK8sResponseStatus(authenticated, allowed, username, uid)


class MockK8sUser:
    """Mock Kubernetes User.

    Represents a user in the mocked Kubernetes environment.
    """

    def __init__(self, username=None, uid=None):
        """Init function."""
        self.username = username
        self.uid = uid


class MockK8sResponseStatus:
    """Mock Kubernetes Response Status.

    Holds the status of a mocked Kubernetes API response,
    including authentication and authorization details,
    and user information if authenticated.
    """

    def __init__(self, authenticated, allowed, username=None, uid=None):
        """Init function."""
        self.authenticated = authenticated
        self.allowed = allowed
        if authenticated:
            self.user = MockK8sUser(username, uid)
        else:
            self.user = None


def mock_token_review_response(token_review):
    """Mock TokenReview Response.

    Simulates a response to a Kubernetes TokenReview request,
    returning authenticated user information for a valid token.

    Parameters:
    - token_review: The TokenReview object being simulated.

    Returns:
    - A MockK8sResponse object with authentication status and user details.
    """
    if token_review.spec.token == "valid-token":  # noqa: S105
        return MockK8sResponse(True, username="valid-user", uid="valid-uid")
    else:
        return MockK8sResponse(False)


def mock_subject_access_review_response(sar):
    """Mock SubjectAccessReview Response.

    Simulates a response to a Kubernetes SubjectAccessReview request,
    determining if a user has authorization for a given action.

    Parameters:
    - sar: The SubjectAccessReview object being simulated.

    Returns:
    - A MockK8sResponse object with authorization status.
    """
    if sar.spec.user == "valid-user":
        return MockK8sResponse(allowed=True)
    else:
        return MockK8sResponse(allowed=False)
