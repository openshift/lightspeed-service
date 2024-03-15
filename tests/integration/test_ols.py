"""Integration tests for basic OLS REST API endpoints."""

import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient
from langchain.schema import AIMessage, HumanMessage

from ols import constants
from ols.app.models.config import ProviderConfig, QueryFilter, ReferenceContent
from ols.utils import config, suid
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@pytest.fixture(scope="module")
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    from ols.app.main import app

    client = TestClient(app)


def test_post_question_on_unexpected_payload(setup):
    """Check the REST API /v1/query with POST HTTP method when unexpected payload is posted."""
    response = client.post("/v1/query", json="this is really not proper payload")
    assert response.status_code == requests.codes.unprocessable
    assert response.json() == {
        "detail": [
            {
                "input": "this is really not proper payload",
                "loc": ["body"],
                "msg": "Input should be a valid dictionary or object to extract fields from",
                "type": "model_attributes_type",
                "url": "https://errors.pydantic.dev/2.6/v/model_attributes_type",
            }
        ],
    }


def test_post_question_without_payload(setup):
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


def test_post_question_on_invalid_question(setup):
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    # let's pretend the question is invalid without even asking LLM
    answer = constants.SUBJECT_INVALID
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
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


def test_post_question_on_generic_response_type_summarize_error(setup):
    """Check the REST API /v1/query with POST HTTP method when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = constants.SUBJECT_VALID
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
        assert response.status_code == requests.codes.internal_server_error
        expected_json = {"detail": "Error while obtaining answer for user question"}
        assert response.json() == expected_json


def test_post_question_that_is_not_validated(setup):
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
        expected_details = {"detail": "Error while validating question"}
        assert response.json() == expected_details


def test_post_question_with_provider_but_not_model(setup):
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


def test_post_question_with_model_but_not_provider(setup):
    """Check how missing provider is detected in request."""
    conversation_id = suid.get_suid()
    response = client.post(
        "/v1/query",
        json={
            "conversation_id": conversation_id,
            "query": "test query",
            "model": constants.GRANITE_13B_CHAT_V1,
        },
    )
    assert response.status_code == requests.codes.unprocessable
    assert len(response.json()["detail"]) == 1
    assert response.json()["detail"][0]["type"] == "value_error"
    assert (
        response.json()["detail"][0]["msg"]
        == "Value error, LLM provider must be specified when the model is specified."
    )


def test_unknown_provider_in_post(setup):
    """Check the REST API /v1/query with POST method when unknown provider is requested."""
    # empty config - no providers
    with patch("ols.utils.config.llm_config.providers", new={}):
        response = client.post(
            "/v1/query",
            json={
                "query": "hello?",
                "provider": "some-provider",
                "model": "some-model",
            },
        )

        assert response.status_code == requests.codes.unprocessable
        assert response.json() == {
            "detail": {
                "response": "Unable to process this request because "
                "'Provider 'some-provider' is not a valid provider. "
                "Valid providers are: []'"
            }
        }


def test_unsupported_model_in_post(setup):
    """Check the REST API /v1/query with POST method when unsupported model is requested."""
    test_provider = "test-provider"
    provider_config = ProviderConfig()
    provider_config.models = {}  # no models configured

    with patch(
        "ols.utils.config.llm_config.providers",
        new={test_provider: provider_config},
    ):
        response = client.post(
            "/v1/query",
            json={
                "query": "hello?",
                "provider": test_provider,
                "model": "some-model",
            },
        )

        assert response.status_code == requests.codes.unprocessable
        assert response.json() == {
            "detail": {
                "response": "Unable to process this request because "
                "'Model 'some-model' is not a valid model for provider "
                "'test-provider'. Valid models are: []'"
            }
        }


def test_post_question_on_noyaml_response_type(setup) -> None:
    """Check the REST API /v1/query with POST HTTP method when call is success."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    config.dev_config.disable_auth = True
    answer = constants.SUBJECT_VALID
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        from tests.mock_classes.langchain_interface import mock_langchain_interface

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
            patch(
                "ols.utils.config.ols_config.reference_content.product_docs_index_path",
                "./invalid_dir",
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
            assert constants.NO_RAG_CONTENT_RESP in response.json()["response"]


@patch("ols.app.endpoints.ols.QuestionValidator.validate_question")
def test_post_question_with_keyword(mock_llm_validation, setup) -> None:
    """Check the REST API /v1/query with keyword validation."""
    config.init_empty_config()
    config.ols_config.query_validation_method = constants.QueryValidationMethod.KEYWORD
    config.dev_config.disable_auth = True
    query = "What is Openshift ?"

    from tests.mock_classes.langchain_interface import mock_langchain_interface

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


def test_post_query_with_query_filters_response_type(setup) -> None:
    """Check the REST API /v1/query with POST HTTP method with query filters."""
    config.init_empty_config()
    config.dev_config.disable_auth = True
    answer = constants.SUBJECT_VALID

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
    config.init_query_filter()
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        from tests.mock_classes.langchain_interface import mock_langchain_interface

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
            assert constants.NO_RAG_CONTENT_RESP in response.json()["response"]
            assert (
                "test query with redacted_ip will be replaced with redacted_ip"
                in response.json()["response"]
            )


def test_post_query_for_conversation_history(setup) -> None:
    """Check the REST API /v1/query with same conversation_id for conversation history."""
    config.init_empty_config()
    config.dev_config.disable_auth = True
    answer = constants.SUBJECT_VALID
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        from tests.mock_classes.langchain_interface import mock_langchain_interface

        ml = mock_langchain_interface("test response")
        with (
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain",
                new=mock_llm_chain(None),
            ),
            patch(
                "ols.src.query_helpers.docs_summarizer.LLMChain.invoke",
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
                    "context": "",
                    "query": "Query1",
                    "chat_history": [],
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
            chat_history_expected = [
                HumanMessage(content="Query1"),
                AIMessage(content=response.json()["response"]),
            ]
            invoke.assert_called_once_with(
                input={
                    "context": "",
                    "query": "Query2",
                    "chat_history": chat_history_expected,
                },
                config={"callbacks": [token_counter.return_value]},
            )
