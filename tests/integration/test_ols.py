"""Integration tests for basic OLS REST API endpoints."""

from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient
from langchain.schema import AIMessage, HumanMessage

from ols import config, constants
from ols.app.models.config import (
    ProviderConfig,
    QueryFilter,
)
from ols.utils import suid
from ols.utils.errors_parsing import DEFAULT_ERROR_MESSAGE, DEFAULT_STATUS_CODE
from tests.mock_classes.mock_langchain_interface import mock_langchain_interface
from tests.mock_classes.mock_llm_chain import mock_llm_chain
from tests.mock_classes.mock_llm_loader import mock_llm_loader


@pytest.fixture(scope="function")
def _setup():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    global client

    # app.main need to be imported after the configuration is read
    from ols.app.main import app

    client = TestClient(app)


def test_post_question_on_unexpected_payload(_setup):
    """Check the REST API /v1/query with POST HTTP method when unexpected payload is posted."""
    response = client.post("/v1/query", json="this is really not proper payload")
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


def test_post_question_without_payload(_setup):
    """Check the REST API /v1/query with POST HTTP method when no payload is posted."""
    # perform POST request without any payload
    response = client.post("/v1/query")
    assert response.status_code == requests.codes.unprocessable

    # check the response payload
    json = response.json()
    assert "detail" in json, "Missing 'detail' node in response payload"
    detail = json["detail"][0]
    assert detail["input"] is None
    assert "Field required" in detail["msg"]


def test_post_question_on_invalid_question(_setup):
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    # let's pretend the question is invalid without even asking LLM
    with patch("ols.app.endpoints.ols.validate_question", return_value=False):
        conversation_id = suid.get_suid()
        response = client.post(
            "/v1/query",
            json={"conversation_id": conversation_id, "query": "test query"},
        )
        assert response.status_code == requests.codes.ok

        expected_json = {
            "conversation_id": conversation_id,
            "response": constants.INVALID_QUERY_RESP,
            "referenced_documents": [],
            "truncated": False,
        }
        assert response.json() == expected_json


def test_post_question_on_generic_response_type_summarize_error(_setup):
    """Check the REST API /v1/query with POST HTTP method when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = constants.SUBJECT_ALLOWED
    with (
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            return_value=answer,
        ),
        patch(
            "ols.app.endpoints.ols.DocsSummarizer.summarize",
            side_effect=Exception("summarizer error"),
        ),
    ):
        conversation_id = suid.get_suid()
        response = client.post(
            "/v1/query",
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


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_that_is_not_validated(_setup):
    """Check the REST API /v1/query with POST HTTP method for question that is not validated."""
    # let's pretend the question can not be validated
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=Exception("can not validate"),
    ):
        conversation_id = suid.get_suid()
        response = client.post(
            "/v1/query",
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


def test_post_question_with_provider_but_not_model(_setup):
    """Check how missing model is detected in request."""
    conversation_id = suid.get_suid()
    response = client.post(
        "/v1/query",
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


def test_post_question_with_model_but_not_provider(_setup):
    """Check how missing provider is detected in request."""
    conversation_id = suid.get_suid()
    response = client.post(
        "/v1/query",
        json={
            "conversation_id": conversation_id,
            "query": "test query",
            "model": constants.GRANITE_13B_CHAT_V2,
        },
    )
    assert response.status_code == requests.codes.unprocessable
    assert len(response.json()["detail"]) == 1
    assert response.json()["detail"][0]["type"] == "value_error"
    assert (
        response.json()["detail"][0]["msg"]
        == "Value error, LLM provider must be specified when the model is specified."
    )


def test_unknown_provider_in_post(_setup):
    """Check the REST API /v1/query with POST method when unknown provider is requested."""
    # empty config - no providers
    config.llm_config.providers = {}
    response = client.post(
        "/v1/query",
        json={
            "query": "hello?",
            "provider": "some-provider",
            "model": "some-model",
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


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_unsupported_model_in_post(_setup):
    """Check the REST API /v1/query with POST method when unsupported model is requested."""
    test_provider = "test-provider"
    provider_config = ProviderConfig()
    provider_config.models = {}  # no models configured
    config.llm_config.providers = {test_provider: provider_config}

    response = client.post(
        "/v1/query",
        json={
            "query": "hello?",
            "provider": test_provider,
            "model": "some-model",
        },
    )

    assert response.status_code == requests.codes.unprocessable
    expected_json = {
        "detail": {
            "cause": "Model 'some-model' is not a valid model for "
            "provider 'test-provider'. Valid models are: []",
            "response": "Unable to process this request",
        }
    }
    assert response.json() == expected_json


def test_post_question_improper_conversation_id(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with improper conversation ID."""
    assert config.dev_config is not None
    config.dev_config.disable_auth = True
    answer = constants.SUBJECT_ALLOWED
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):

        conversation_id = "not-correct-uuid"
        response = client.post(
            "/v1/query",
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


def test_post_question_on_noyaml_response_type(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method when call is success."""
    answer = constants.SUBJECT_ALLOWED
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                },
            )
            print(response)
            assert response.status_code == requests.codes.ok


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.KEYWORD,
)
@patch("ols.app.endpoints.ols.QuestionValidator.validate_question")
def test_post_question_with_keyword(mock_llm_validation, _setup) -> None:
    """Check the REST API /v1/query with keyword validation."""
    query = "What is Openshift ?"

    ml = mock_langchain_interface(None)
    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.LLMChain",
            new=mock_llm_chain(None),
        ),
        patch(
            "ols.src.query_helpers.query_helper.load_llm",
            new=mock_llm_loader(ml()),
        ),
    ):
        conversation_id = suid.get_suid()
        response = client.post(
            "/v1/query",
            json={"conversation_id": conversation_id, "query": query},
        )
        assert response.status_code == requests.codes.ok
        # Currently mock invoke passes same query as response text.
        assert query in response.json()["response"]
        assert mock_llm_validation.call_count == 0


def test_post_query_with_query_filters_response_type(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with query filters."""
    answer = constants.SUBJECT_ALLOWED

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

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
                json={
                    "conversation_id": conversation_id,
                    "query": "test query with 9.25.33.67 will be replaced with redacted_ip",
                },
            )
            print(response.json())
            assert response.status_code == requests.codes.ok
            assert (
                "test query with redacted_ip will be replaced with redacted_ip"
                in response.json()["response"]
            )


def test_post_query_for_conversation_history(_setup) -> None:
    """Check the REST API /v1/query with same conversation_id for conversation history."""
    answer = constants.SUBJECT_ALLOWED
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):

        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain.invoke",
                return_value={"text": "some response"},
            ) as invoke,
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
            patch(
                "ols.app.metrics.token_counter.TokenMetricUpdater.__enter__",
            ) as token_counter,
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
                json={
                    "conversation_id": conversation_id,
                    "query": "Query1",
                },
            )
            assert response.status_code == requests.codes.ok
            invoke.assert_called_once_with(
                input={
                    "query": "Query1",
                },
                config={"callbacks": [token_counter.return_value]},
            )
            invoke.reset_mock()

            response = client.post(
                "/v1/query",
                json={
                    "conversation_id": conversation_id,
                    "query": "Query2",
                },
            )
            chat_history_expected = f"human: Query1\nai: {response.json()['response']}"
            chat_history_expected = [
                HumanMessage(content="Query1"),
                AIMessage(content=response.json()["response"]),
            ]
            invoke.assert_called_once_with(
                input={
                    "query": "Query2",
                    "chat_history": chat_history_expected,
                },
                config={"callbacks": [token_counter.return_value]},
            )


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_without_attachments(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method without attachments."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                },
            )
            assert response.status_code == requests.codes.ok
    assert query_passed == "test query"


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
@pytest.mark.attachment()
def test_post_question_with_empty_list_of_attachments(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with empty list of attachments."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
                json={
                    "conversation_id": conversation_id,
                    "query": "test query",
                    "attachments": [],
                },
            )
            assert response.status_code == requests.codes.ok
    assert query_passed == "test query"


@pytest.mark.attachment()
@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_with_one_plaintext_attachment(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with one attachment."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
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


@pytest.mark.attachment()
@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_with_one_yaml_attachment(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with YAML attachment."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
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
            response = client.post(
                "/v1/query",
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


@pytest.mark.attachment()
@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_with_two_yaml_attachments(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with two YAML attachments."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
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
            response = client.post(
                "/v1/query",
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


@pytest.mark.attachment()
@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_with_one_yaml_without_kind_attachment(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with one YAML without kind attachment."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
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
            response = client.post(
                "/v1/query",
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


@pytest.mark.attachment()
@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_with_one_yaml_without_name_attachment(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with one YAML without name attachment."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
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
            response = client.post(
                "/v1/query",
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


@pytest.mark.attachment()
@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.LLM,
)
def test_post_question_with_one_invalid_yaml_attachment(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with one invalid YAML attachment."""
    answer = constants.SUBJECT_ALLOWED
    query_passed = None

    def validate_question(_conversation_id, query):
        """Closure called indirectly to validate the question."""
        nonlocal query_passed
        query_passed = query
        return answer

    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=validate_question,
    ):
        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
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
            response = client.post(
                "/v1/query",
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


@pytest.mark.attachment()
def test_post_question_with_large_attachment(_setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with large attachment."""
    answer = constants.SUBJECT_ALLOWED

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
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.query_helper.load_llm",
                new=mock_llm_loader(ml()),
            ),
        ):
            conversation_id = suid.get_suid()

            response = client.post(
                "/v1/query",
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
            # error should be returned because of very large input
            assert response.status_code == requests.codes.request_entity_too_large


def test_post_too_long_query(_setup):
    """Check the REST API /v1/query with POST HTTP method for query that is too long."""
    query = "test query" * 1000
    conversation_id = suid.get_suid()
    response = client.post(
        "/v1/query",
        json={"conversation_id": conversation_id, "query": query},
    )

    # error should be returned
    assert response.status_code == requests.codes.request_entity_too_large
    error_response = response.json()["detail"]
    assert error_response["response"] == "Prompt is too long"
    assert "exceeds" in error_response["cause"]
