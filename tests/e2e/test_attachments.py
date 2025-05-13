"""End to end tests for the REST API endpoint /query when attachment(s) are send in request."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import pytest
import requests

from tests.e2e.utils import metrics as metrics_utils
from tests.e2e.utils import response as response_utils
from tests.e2e.utils.decorators import retry

from . import test_api


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_empty_attachment_list() -> None:
    """Check the REST API /v1/query with POST HTTP method using empty attachment list."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, endpoint, status_code=requests.codes.ok
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # HTTP OK should be returned
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_one_attachment() -> None:
    """Check the REST API /v1/query with POST HTTP method using one attachment."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, endpoint, status_code=requests.codes.ok
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [
                    {
                        "attachment_type": "log",
                        "content_type": "text/plain",
                        "content": "Kubernetes is a core component of OpenShift.",
                    },
                ],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # HTTP OK should be returned
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_more_attachments() -> None:
    """Check the REST API /v1/query with POST HTTP method using two attachments."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, endpoint, status_code=requests.codes.ok
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [
                    {
                        "attachment_type": "log",
                        "content_type": "text/plain",
                        "content": "Kubernetes is a core component of OpenShift.",
                    },
                    {
                        "attachment_type": "configuration",
                        "content_type": "application/json",
                        "content": "{'foo': 'bar'}",
                    },
                ],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # HTTP OK should be returned
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_wrong_attachment_format_unknown_field() -> None:
    """Check the REST API /v1/query with POST HTTP method using attachment with wrong format."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        endpoint,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [
                    {
                        "xyzzy": "log",  # unknown field
                        "content_type": "text/plain",
                        "content": "this is attachment",
                    },
                ],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # the attachment should not be processed correctly
        assert response.status_code == requests.codes.unprocessable_entity

        json_response = response.json()
        details = json_response["detail"][0]
        assert details["msg"] == "Field required"
        assert details["type"] == "missing"


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_wrong_attachment_format_missing_fields() -> None:
    """Check the REST API /v1/query with POST HTTP method using attachment with wrong format."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        endpoint,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [  # missing fields
                    {},
                ],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # the attachment should not be processed correctly
        assert response.status_code == requests.codes.unprocessable_entity

        json_response = response.json()
        details = json_response["detail"][0]
        assert details["msg"] == "Field required"
        assert details["type"] == "missing"


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_wrong_attachment_format_field_of_different_type() -> None:
    """Check the REST API /v1/query with POST HTTP method using attachment with wrong value type."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        endpoint,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [
                    {
                        "attachment_type": 42,  # not a string
                        "content_type": "application/json",
                        "content": "{'foo': 'bar'}",
                    },
                ],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # the attachment should not be processed correctly
        assert response.status_code == requests.codes.unprocessable_entity

        json_response = response.json()
        details = json_response["detail"][0]
        assert details["msg"] == "Input should be a valid string"
        assert details["type"] == "string_type"


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_wrong_attachment_format_unknown_attachment_type() -> None:
    """Check the REST API /v1/query with POST HTTP method using attachment with wrong type."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        endpoint,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [
                    {
                        "attachment_type": "unknown_type",
                        "content_type": "text/plain",
                        "content": "this is attachment",
                    },
                ],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # the attachment should not be processed correctly
        assert response.status_code == requests.codes.unprocessable_entity

        json_response = response.json()
        expected_response = {
            "detail": {
                "response": "Unable to process this request",
                "cause": "Attachment with improper type unknown_type detected",
            }
        }
        assert json_response == expected_response


@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_with_wrong_attachment_format_unknown_content_type() -> None:
    """Check the REST API /v1/query with POST HTTP method: attachment with wrong content type."""
    endpoint = "/v1/query"

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        endpoint,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "attachments": [
                    {
                        "attachment_type": "log",
                        "content_type": "unknown/type",
                        "content": "this is attachment",
                    },
                ],
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )

        # the attachment should not be processed correctly
        assert response.status_code == requests.codes.unprocessable_entity

        json_response = response.json()
        expected_response = {
            "detail": {
                "response": "Unable to process this request",
                "cause": "Attachment with improper content type unknown/type detected",
            }
        }
        assert json_response == expected_response
