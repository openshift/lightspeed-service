"""Integration tests for basic OLS REST API endpoints."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import logging
from unittest.mock import AsyncMock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from ols import config, constants
from ols.app.models.config import (
    LoggingConfig,
    ProviderConfig,
    QueryFilter,
)
from ols.customize import prompts
from ols.utils import suid
from ols.utils.errors_parsing import DEFAULT_ERROR_MESSAGE, DEFAULT_STATUS_CODE
from ols.utils.logging_configurator import configure_logging
from tests.mock_classes.mock_langchain_interface import mock_langchain_interface
from tests.mock_classes.mock_llm_loader import mock_llm_loader
from tests.mock_classes.mock_tools import mock_tools_map

INVALID_QUERY_RESP = prompts.INVALID_QUERY_RESP


@pytest.fixture(scope="function")
def _setup():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    pytest.client = TestClient(app)


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_on_unexpected_payload(_setup, endpoint):
    """Check the REST API /v1/query when unexpected payload is posted."""
    response = pytest.client.post(endpoint, json="this is really not proper payload")
    assert response.status_code == requests.codes.unprocessable

    # try to deserialize payload
    response_json = response.json()

    # remove attribute that strongly depends on Pydantic version
    if "url" in response_json["detail"][0]:
        del response_json["detail"][0]["url"]

    assert response_json == {
        "detail": [
            {
                "input": "this is really not proper payload",
                "loc": ["body"],
                "msg": "Input should be a valid dictionary or object to extract fields from",
                "type": "model_attributes_type",
            }
        ],
    }


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_without_payload(_setup, endpoint):
    """Check the REST API query endpoints when no payload is posted."""
    # perform POST request without any payload
    response = pytest.client.post(endpoint)
    assert response.status_code == requests.codes.unprocessable

    # check the response payload
    json = response.json()
    assert "detail" in json, "Missing 'detail' node in response payload"
    detail = json["detail"][0]
    assert detail["input"] is None
    assert "Field required" in detail["msg"]


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_on_invalid_question(_setup, endpoint):
    """Check the REST API /v1/query for invalid question."""
    # let's pretend the question is invalid without even asking LLM
    with patch("ols.app.endpoints.ols.validate_question", return_value=False):
        conversation_id = suid.get_suid()
        response = pytest.client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "test query"},
        )
        assert response.status_code == requests.codes.ok

        if response.headers["content-type"] == "application/json":
            # non-streaming responses return JSON
            expected_response = {
                "conversation_id": conversation_id,
                "response": INVALID_QUERY_RESP,
                "referenced_documents": [],
                "truncated": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "available_quotas": {},
                "tool_calls": [],
                "tool_results": [],
            }
            actual_response = response.json()
        else:
            # streaming_query returns bytes
            expected_response = INVALID_QUERY_RESP
            actual_response = response.text

        assert actual_response == expected_response


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_on_generic_response_type_summarize_error(_setup, endpoint):
    """Check the REST API query endpoints when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = True
    with (
        patch(
            "ols.src.query_helpers.question_validator.QuestionValidator.validate_question",
            return_value=answer,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer.create_response",
            side_effect=Exception("summarizer error"),
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer.generate_response",
            side_effect=Exception("summarizer error"),
        ),
    ):
        conversation_id = suid.get_suid()
        response = pytest.client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "test query"},
        )
        assert response.status_code == DEFAULT_STATUS_CODE
        expected_json = {
            "detail": {
                "response": DEFAULT_ERROR_MESSAGE,
                "cause": "summarizer error",
            }
        }

        assert response.json() == expected_json


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_that_is_not_validated(_setup, endpoint):
    """Check the REST API query endpoints for question that is not validated."""
    # let's pretend the question can not be validated
    with (
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=Exception("can not validate"),
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        conversation_id = suid.get_suid()
        response = pytest.client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "test query"},
        )

        # error should be returned
        assert response.status_code == requests.codes.internal_server_error
        expected_details = {
            "detail": {
                "cause": "can not validate",
                "response": "Error while validating question",
            }
        }
        assert response.json() == expected_details


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_with_provider_but_not_model(_setup, endpoint):
    """Check how missing model is detected in request."""
    conversation_id = suid.get_suid()
    response = pytest.client.post(
        endpoint,
        json={
            "conversation_id": conversation_id,
            "query": "test query",
            "provider": constants.PROVIDER_BAM,
        },
    )
    assert response.status_code == requests.codes.unprocessable
    assert len(response.json()["detail"]) == 1
    assert response.json()["detail"][0]["type"] == "value_error"
    assert (
        response.json()["detail"][0]["msg"]
        == "Value error, LLM model must be specified when the provider is specified."
    )


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_with_model_but_not_provider(_setup, endpoint):
    """Check how missing provider is detected in request."""
    conversation_id = suid.get_suid()
    response = pytest.client.post(
        endpoint,
        json={
            "conversation_id": conversation_id,
            "query": "test query",
            "model": "model-name",
        },
    )
    assert response.status_code == requests.codes.unprocessable
    assert len(response.json()["detail"]) == 1
    assert response.json()["detail"][0]["type"] == "value_error"
    assert (
        response.json()["detail"][0]["msg"]
        == "Value error, LLM provider must be specified when the model is specified."
    )


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_unknown_provider_in_post(_setup, endpoint):
    """Check the REST API query endpoints with POST method when unknown provider is requested."""
    # empty config - no providers
    config.llm_config.providers = {}
    response = pytest.client.post(
        endpoint,
        json={
            "query": "hello?",
            "provider": "some-provider",
            "model": "model-name",
        },
    )

    assert response.status_code == requests.codes.unprocessable
    expected_json = {
        "detail": {
            "cause": "Provider 'some-provider' is not a valid provider. "
            "Valid providers are: []",
            "response": "Unable to process this request",
        }
    }

    assert response.json() == expected_json


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_unsupported_model_in_post(_setup, endpoint):
    """Check the REST API query endpoints with POST method when unsupported model is requested."""
    test_provider = "test-provider"
    provider_config = ProviderConfig()
    provider_config.models = {}  # no models configured
    config.llm_config.providers = {test_provider: provider_config}

    with patch(
        "ols.app.endpoints.ols.config.ols_config.query_validation_method",
        constants.QueryValidationMethod.LLM,
    ):
        response = pytest.client.post(
            endpoint,
            json={
                "query": "hello?",
                "provider": test_provider,
                "model": "model-name",
            },
        )

        assert response.status_code == requests.codes.unprocessable
        expected_json = {
            "detail": {
                "cause": "Model 'model-name' is not a valid model for "
                "provider 'test-provider'. Valid models are: []",
                "response": "Unable to process this request",
            }
        }
        assert response.json() == expected_json


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_improper_conversation_id(_setup, endpoint) -> None:
    """Check the REST API query endpoints with improper conversation ID."""
    assert config.dev_config is not None
    config.dev_config.disable_auth = True
    answer = True
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        conversation_id = "not-correct-uuid"
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "test query",
            },
        )
        # error should be returned
        assert response.status_code == requests.codes.internal_server_error
        expected_details = {
            "detail": {
                "cause": "Invalid conversation ID not-correct-uuid",
                "response": "Error retrieving conversation history",
            }
        }
        assert response.json() == expected_details


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_on_noyaml_response_type(_setup, endpoint) -> None:
    """Check the REST API query endpoints when call is success."""
    ml = mock_langchain_interface("test response")
    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(ml()),
        ),
    ):
        conversation_id = suid.get_suid()
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "test query",
            },
        )
        print(response)
        assert response.status_code == requests.codes.ok


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_with_keyword(_setup, endpoint) -> None:
    """Check the REST API /v1/query with keyword validation."""
    query = "What is Openshift ?"

    ml = mock_langchain_interface(None)
    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(ml()),
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.KEYWORD,
        ),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question"
        ) as mock_llm_validation,
    ):
        conversation_id = suid.get_suid()
        response = pytest.client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": query},
        )
        assert response.status_code == requests.codes.ok

        if response.headers["content-type"] == "application/json":
            # non-streaming responses return JSON
            actual_response = response.json()["response"]
        else:
            # streaming_query returns bytes
            actual_response = response.text

        # Currently mock invoke passes same query as response text.
        assert query in actual_response
        assert mock_llm_validation.call_count == 0


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_query_with_query_filters_response_type(_setup, endpoint) -> None:
    """Check the REST API query endpoints with query filters."""
    answer = True

    query_filters = [
        QueryFilter(
            {
                "name": "test",
                "pattern": r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
                "replace_with": "redacted_ip",
            }
        )
    ]
    config.ols_config.query_filters = query_filters

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.src.query_helpers.question_validator.QuestionValidator.validate_question",
            return_value=answer,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query with 9.25.33.67 will be replaced with redacted_ip",
                },
            )

            assert response.status_code == requests.codes.ok

            if response.headers["content-type"] == "application/json":
                # non-streaming responses return JSON
                actual_response = response.json()["response"]
            else:
                # streaming_query returns bytes
                actual_response = response.text

            assert (
                "test query with redacted_ip will be replaced with redacted_ip"
                in actual_response
            )


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_query_for_conversation_history(_setup, endpoint) -> None:
    """Check the REST API query endpoints with same conversation_id for conversation history."""
    # we need to import it here because these modules triggers config
    # load too -> causes exception in auth module because of missing config
    # values
    from ols.app.endpoints.ols import retrieve_previous_input  # pylint: disable=C0415
    from ols.app.models.models import CacheEntry  # pylint: disable=C0415

    actual_returned_history = []

    def capture_return_value(*args, **kwargs):
        nonlocal actual_returned_history
        actual_returned_history = retrieve_previous_input(*args, **kwargs)
        return actual_returned_history

    ml = mock_langchain_interface("test response")
    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(ml()),
        ),
        patch(
            "ols.app.endpoints.ols.retrieve_previous_input",
            side_effect=capture_return_value,
        ),
    ):
        conversation_id = suid.get_suid()
        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "Query1",
            },
        )
        assert response.status_code == requests.codes.ok
        assert actual_returned_history == []  # pylint: disable=C1803

        response = pytest.client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "Query2",
            },
        )
        assert response.status_code == requests.codes.ok
        chat_history_expected = [
            CacheEntry(
                query=HumanMessage("Query1"),
                response=AIMessage("Query1"),
                attachments=[],
            )
        ]
        # cannot test exact timestamp, test the existence
        assert (
            actual_returned_history[0].query.content
            == chat_history_expected[0].query.content
        )
        assert (
            actual_returned_history[0].response.content
            == chat_history_expected[0].response.content
        )


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_question_without_attachments(_setup, endpoint) -> None:
    """Check the REST API query endpoints without attachments."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                },
            )
            assert response.status_code == requests.codes.ok
    assert query_passed == "test query"


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_empty_list_of_attachments(_setup, endpoint) -> None:
    """Check the REST API query endpoints with empty list of attachments."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [],
                },
            )
            assert response.status_code == requests.codes.ok
    assert query_passed == "test query"


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_one_plaintext_attachment(_setup, endpoint) -> None:
    """Check the REST API query endpoints with one attachment."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [
                        {
                            "attachment_type": "log",
                            "content": "this is attachment",
                            "content_type": "text/plain",
                        },
                    ],
                },
            )
            assert response.status_code == requests.codes.ok
    expected = """test query


```
this is attachment
```
"""
    assert query_passed == expected


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_one_yaml_attachment(_setup, endpoint) -> None:
    """Check the REST API query endpoints with YAML attachment."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            yaml = """
kind: Pod
metadata:
     name: private-reg
"""
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [
                        {
                            "attachment_type": "configuration",
                            "content": yaml,
                            "content_type": "application/yaml",
                        },
                    ],
                },
            )
            assert response.status_code == requests.codes.ok
    expected = """test query

For reference, here is the full resource YAML for Pod 'private-reg':
```yaml

kind: Pod
metadata:
     name: private-reg

```
"""
    assert query_passed == expected


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_two_yaml_attachments(_setup, endpoint) -> None:
    """Check the REST API query endpoints with two YAML attachments."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            yaml1 = """
kind: Pod
metadata:
     name: private-reg
"""
            yaml2 = """
kind: Deployment
metadata:
     name: foobar-deployment
"""
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [
                        {
                            "attachment_type": "configuration",
                            "content": yaml1,
                            "content_type": "application/yaml",
                        },
                        {
                            "attachment_type": "configuration",
                            "content": yaml2,
                            "content_type": "application/yaml",
                        },
                    ],
                },
            )
            assert response.status_code == requests.codes.ok
    expected = """test query

For reference, here is the full resource YAML for Pod 'private-reg':
```yaml

kind: Pod
metadata:
     name: private-reg

```


For reference, here is the full resource YAML for Deployment 'foobar-deployment':
```yaml

kind: Deployment
metadata:
     name: foobar-deployment

```
"""
    assert query_passed == expected


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_one_yaml_without_kind_attachment(_setup, endpoint) -> None:
    """Check the REST API query endpoints with one YAML without kind attachment."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            yaml = """
metadata:
     name: private-reg
"""
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [
                        {
                            "attachment_type": "configuration",
                            "content": yaml,
                            "content_type": "application/yaml",
                        },
                    ],
                },
            )
            assert response.status_code == requests.codes.ok
    expected = """test query

For reference, here is the full resource YAML:
```yaml

metadata:
     name: private-reg

```
"""
    assert query_passed == expected


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_one_yaml_without_name_attachment(_setup, endpoint) -> None:
    """Check the REST API query endpoints with one YAML without name attachment."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            yaml = """
kind: Deployment
metadata:
     foo: bar
"""
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [
                        {
                            "attachment_type": "configuration",
                            "content": yaml,
                            "content_type": "application/yaml",
                        },
                    ],
                },
            )
            assert response.status_code == requests.codes.ok
    expected = """test query

For reference, here is the full resource YAML:
```yaml

kind: Deployment
metadata:
     foo: bar

```
"""
    assert query_passed == expected


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_one_invalid_yaml_attachment(_setup, endpoint) -> None:
    """Check the REST API query endpoints with one invalid YAML attachment."""
    answer = True
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            side_effect=validate_question,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            yaml = """
kind: Pod
*metadata:
     name: private-reg
"""
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [
                        {
                            "attachment_type": "configuration",
                            "content": yaml,
                            "content_type": "application/yaml",
                        },
                    ],
                },
            )
            assert response.status_code == requests.codes.ok
    expected = """test query

For reference, here is the full resource YAML:
```yaml

kind: Pod
*metadata:
     name: private-reg

```
"""
    assert query_passed == expected


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.attachment
def test_post_question_with_large_attachment(_setup, endpoint) -> None:
    """Check the REST API query endpoints with large attachment."""
    answer = True

    def validate_question(_conversation_id, _query):
        """Closure called indirectly to validate the question."""
        return answer

    # generate large YAML content that exceeds token limit
    yaml = """
kind: Pod
metadata:
     name: private-reg
logs:
"""
    for i in range(10000):
        yaml += f"    log{i}: 'this is log message #{i}"

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()

            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [
                        {
                            "attachment_type": "configuration",
                            "content": yaml,
                            "content_type": "application/yaml",
                        },
                    ],
                },
            )
            if response.headers["content-type"] == "application/json":
                # non-streaming responses return JSON
                assert response.status_code == requests.codes.request_entity_too_large
            else:
                # streaming_query returns bytes
                error_response = response.text
                assert "Prompt is too long" in error_response
                assert "exceeds LLM available context window limit" in error_response


@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
def test_post_too_long_query(_setup, endpoint):
    """Check the REST API query endpoints for query that is too long."""
    query = "test query" * 1000
    conversation_id = suid.get_suid()
    response = pytest.client.post(
        endpoint,
        json={"conversation_id": conversation_id, "query": query},
    )

    if response.headers["content-type"] == "application/json":
        # non-streaming responses return JSON
        assert response.status_code == requests.codes.request_entity_too_large
        error_response = response.json()["detail"]
        assert error_response["response"] == "Prompt is too long"
        assert "exceeds" in error_response["cause"]
    else:
        # streaming_query returns bytes
        error_response = response.text
        assert "Prompt is too long" in error_response
        assert "exceeds LLM available context window limit" in error_response


def _post_with_system_prompt_override(_setup, caplog, query, system_prompt):
    """Invoke the POST /v1/query API with a system prompt override."""
    logging_config = LoggingConfig(app_log_level="debug")

    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=lambda x, y: True,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = pytest.client.post(
                "/v1/query",
                json={
                    "conversation_id": conversation_id,
                    "query": query,
                    "system_prompt": system_prompt,
                },
            )
            assert response.status_code == requests.codes.ok

    # Specified system prompt should appear twice in query_helper outputs:
    # One is from question_validator and another from docs_summarizer.
    assert response.status_code == requests.codes.ok


def test_post_with_system_prompt_override(_setup, caplog):
    """Check the POST /v1/query API with a system prompt."""
    query = "test query"
    system_prompt = "You are an expert in something marvelous."

    with (
        patch(
            "ols.app.endpoints.ols.config.dev_config.enable_system_prompt_override",
            True,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        _post_with_system_prompt_override(_setup, caplog, query, system_prompt)

    # Specified system prompt should appear twice in query_helper debug log outputs.
    # One is from question_validator and another is from docs_summarizer.
    assert caplog.text.count("System prompt: " + system_prompt) == 2


def test_post_with_system_prompt_override_disabled(_setup, caplog):
    """Check the POST /v1/query API with a system prompt when overriding is disabled."""
    query = "test query"
    system_prompt = "You are an expert in something marvelous."
    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.app.endpoints.ols.config.dev_config.enable_system_prompt_override",
            False,
        ),
        patch(
            "ols.app.endpoints.ols.config.ols_config.query_validation_method",
            constants.QueryValidationMethod.LLM,
        ),
    ):
        _post_with_system_prompt_override(_setup, caplog, query, system_prompt)

    # Specified system prompt should NOT appear in query_helper debug log outputs
    # as enable_system_prompt_override is set to False.
    assert caplog.text.count("System prompt: " + system_prompt) == 0


# TODO: consolidate this and the same function in test_docs_summarizer.py
# into some reusable fixture
async def fake_invoke_llm(*args, **kwargs):
    """Fake invoke_llm function to simulate LLM behavior.

    Yields depends on the number of calls
    """
    # use an attribute on the function to track calls
    if not hasattr(fake_invoke_llm, "call_count"):
        fake_invoke_llm.call_count = 0
    fake_invoke_llm.call_count += 1

    if fake_invoke_llm.call_count == 1:
        # first call yields a message that requests tool calls
        yield AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "tool_calls"},
            tool_calls=[
                {"name": "get_namespaces_mock", "args": {}, "id": "call_id1"},
            ],
        )
    elif fake_invoke_llm.call_count == 2:
        # second call yields the final message.
        yield AIMessageChunk(
            content="You have 1 namespace.",
        )
    else:
        # the end event
        yield AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "stop"},
        )


def test_tool_calling(_setup, caplog) -> None:
    """Check the REST API query endpoints when tool calling is enabled."""
    endpoint = "/v1/query"
    caplog.set_level(10)
    mcp_servers = {"fake-server": {"transport": "http", "url": "http://fake-server"}}

    with (
        patch("ols.customize.prompts.QUERY_SYSTEM_INSTRUCTION", "System Instruction"),
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._build_mcp_config",
            return_value=mcp_servers,
        ),
        patch(
            "ols.src.query_helpers.docs_summarizer.MultiServerMCPClient"
        ) as mock_mcp_client_cls,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer._invoke_llm",
            new=fake_invoke_llm,
        ) as mock_invoke,
    ):
        # Create mock tools map
        mock_mcp_client_instance = AsyncMock()
        mock_mcp_client_instance.get_tools.return_value = mock_tools_map
        mock_mcp_client_cls.return_value = mock_mcp_client_instance

        with (
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(None),
            ),
        ):
            conversation_id = suid.get_suid()
            response = pytest.client.post(
                endpoint,
                json={
                    "conversation_id": conversation_id,
                    "query": "How many namespaces are there in my cluster?",
                },
            )
            assert mock_invoke.call_count == 3

            assert "Tool: get_namespaces_mock" in caplog.text
            tool_output = mock_tools_map[0].invoke({})
            assert f"Output: {tool_output}" in caplog.text

            assert response.status_code == requests.codes.ok
            assert response.json()["response"] == "You have 1 namespace."
