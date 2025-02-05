"""Integration tests for basic OLS REST API endpoints with no-op auth."""

from fastapi import Request

from ols import config, constants
from ols.src.auth.auth import get_auth_dependency


def test_authorized():
    """Check authorization based on no-op module."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    config.ols_config.authentication_config.module = "noop"
    config.dev_config.disable_auth = True
    from ols.app.endpoints import authorized  # pylint: disable=C0415

    # auth is disabled
    request = Request(scope={"type": "http", "headers": [], "query_string": ""})
    response = authorized.is_user_authorized(request)
    assert response is not None
    assert response.user_id == constants.DEFAULT_USER_UID
    assert response.username == constants.DEFAULT_USER_NAME

    config.dev_config.disable_auth = False
    authorized.auth_dependency = get_auth_dependency(
        config.ols_config, virtual_path="/ols-access"
    )

    # auth is enabled
    request = Request(scope={"type": "http", "headers": [], "query_string": ""})
    response = authorized.is_user_authorized(request)
    assert response is not None
    assert response.user_id == constants.DEFAULT_USER_UID
    assert response.username == constants.DEFAULT_USER_NAME

    # Simulate a request with user_id specified as optional parameter
    user_id_in_request = "00000000-1234-1234-1234-000000000000"
    request = Request(
        scope={
            "type": "http",
            "headers": [],
            "query_string": f"user_id={user_id_in_request}",
        }
    )

    response = authorized.is_user_authorized(request)
    assert response is not None
    assert response.user_id == user_id_in_request
    assert response.username == constants.DEFAULT_USER_NAME
    assert response.skip_user_id_check is True
