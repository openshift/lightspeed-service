"""Checks for responses from the service."""

import requests


def check_content_type(response, content_type, message=""):
    """Check if response content-type is set to defined value."""
    assert response.headers["content-type"].startswith(content_type), message


def check_missing_field_response(response, field_name):
    """Check if 'Field required' error is returned by the service."""
    # error should be detected on Pydantic level
    assert response.status_code == requests.codes.unprocessable

    # the resonse payload should be valid JSON
    check_content_type(response, "application/json")
    json_response = response.json()

    # check payload details
    assert (
        "detail" in json_response
    ), "Improper response format: 'detail' node is missing"
    assert json_response["detail"][0]["msg"] == "Field required"
    assert json_response["detail"][0]["loc"][0] == "body"
    assert json_response["detail"][0]["loc"][1] == field_name
