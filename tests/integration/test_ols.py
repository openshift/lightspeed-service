"""Integration tests for basic OLS REST API endpoints."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols import constants
from ols.app.models.config import ProviderConfig, ReferenceContent
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.utils import config, suid
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    from ols.app.main import app

    client = TestClient(app)


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query():
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    conversation_id = suid.get_suid()
    response = client.post(
        "/v1/debug/query",
        json={"conversation_id": conversation_id, "query": "test query"},
    )

    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "conversation_id": conversation_id,
        "response": "test response",
    }


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query_no_conversation_id():
    """Check the REST API /v1/debug/query with POST HTTP method conversation ID is not provided."""
    response = client.post(
        "/v1/debug/query",
        json={"query": "test query"},
    )

    assert response.status_code == requests.codes.ok
    json = response.json()

    # check that conversation ID is being created
    assert len(json["conversation_id"]) > 0
    assert json["response"] == "test response"


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query_no_query():
    """Check the REST API /v1/debug/query with POST HTTP method when query is not specified."""
    conversation_id = suid.get_suid()
    response = client.post(
        "/v1/debug/query",
        json={"conversation_id": conversation_id},
    )

    # request can't be processed correctly
    assert response.status_code == requests.codes.unprocessable


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query_no_payload():
    """Check the REST API /v1/debug/query with POST HTTP method when payload is empty."""
    response = client.post("/v1/debug/query")

    # request can't be processed correctly
    assert response.status_code == requests.codes.unprocessable


def test_post_question_on_unexpected_payload():
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


def test_post_question_without_payload():
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


def test_post_question_on_invalid_question():
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
        expected_details = (
            "I can only answer questions about OpenShift and Kubernetes. "
            "Please rephrase your question"
        )
        expected_json = {
            "conversation_id": conversation_id,
            "response": expected_details,
        }
        assert response.json() == expected_json


def test_post_question_on_generic_response_type_summarize_error():
    """Check the REST API /v1/query with POST HTTP method when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = constants.SUBJECT_VALID
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        with patch(
            "ols.app.endpoints.ols.DocsSummarizer.summarize",
            side_effect=Exception("summarizer error"),
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
                json={"conversation_id": conversation_id, "query": "test query"},
            )
            assert response.status_code == requests.codes.internal_server_error
            expected_json = {"detail": "Error while obtaining answer for user question"}
            assert response.json() == expected_json


def test_post_question_on_generic_response_llm_configuration_error():
    """Check the REST API /v1/query with POST HTTP method when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = constants.SUBJECT_VALID
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        with patch(
            "ols.app.endpoints.ols.DocsSummarizer.summarize",
            side_effect=LLMConfigurationError("LLM configuration error"),
        ):
            conversation_id = suid.get_suid()
            response = client.post(
                "/v1/query",
                json={"conversation_id": conversation_id, "query": "test query"},
            )
            assert response.status_code == requests.codes.unprocessable
            expected_json = {
                "detail": {
                    "response": "Unable to process this request because 'LLM configuration error'"
                }
            }
            assert response.json() == expected_json


def test_post_question_that_is_not_validated():
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


def test_post_question_with_provider_but_not_model():
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
        == "Value error, LLM model must be specified when the provider is specified!"
    )


def test_post_question_with_model_but_not_provider():
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
        == "Value error, LLM provider must be specified when the model is specified!"
    )


class TestQuery:
    """Test the /v1/query endpoint."""

    def test_unknown_provider_in_post(self):
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
                    "'No configuration for LLM provider some-provider'"
                }
            }

    def test_unsupported_model_in_post(self):
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
                    "'No configuration provided for model some-model under "
                    "LLM provider test-provider'"
                }
            }


def test_post_question_on_generic_response_type() -> None:
    """Check the REST API /v1/query with POST HTTP method."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    answer = constants.SUBJECT_VALID
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        from tests.mock_classes.langchain_interface import mock_langchain_interface

        ml = mock_langchain_interface("test response")
        with patch(
            "ols.src.query_helpers.docs_summarizer.LLMLoader", new=mock_llm_loader(ml())
        ):
            with patch(
                "ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults"
            ):
                with patch(
                    "ols.utils.config.ols_config.reference_content.product_docs_index_path",
                    "./invalid_dir",
                ):
                    with patch(
                        "ols.src.query_helpers.docs_summarizer.LLMChain",
                        new=mock_llm_chain(ml),
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
                        assert (
                            "The following response was generated without access "
                            "to reference content:" in response.json()["response"]
                        )


def test_post_question_on_generic_history() -> None:
    """Check the REST API /v1/query with POST HTTP method to verify conversation history."""
    config.init_empty_config()
    config.ols_config.reference_content = ReferenceContent(None)
    config.ols_config.reference_content.product_docs_index_path = "./invalid_dir"
    config.ols_config.reference_content.product_docs_index_id = "product"
    answer = constants.SUBJECT_VALID
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        from tests.mock_classes.langchain_interface import mock_langchain_interface

        ml = mock_langchain_interface("test response")
        with patch(
            "ols.src.query_helpers.docs_summarizer.LLMLoader", new=mock_llm_loader(ml())
        ):
            with patch(
                "ols.src.query_helpers.docs_summarizer.ServiceContext.from_defaults"
            ):
                with patch(
                    "ols.utils.config.ols_config.reference_content.product_docs_index_path",
                    "./invalid_dir",
                ):
                    with patch(
                        "ols.src.query_helpers.docs_summarizer.LLMChain",
                        new=mock_llm_chain(ml),
                    ):
                        conversation_id = suid.get_suid()
                        response = client.post(
                            "/v1/query",
                            json={
                                "conversation_id": conversation_id,
                                "query": "First query",
                            },
                        )
                        response = client.post(
                            "/v1/query",
                            json={
                                "conversation_id": conversation_id,
                                "query": "Second query",
                            },
                        )
                        print(response)
                        assert response.status_code == requests.codes.ok
                        assert "First query" in response.json()["response"]
                        assert "Second query" in response.json()["response"]
