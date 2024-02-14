"""Integration tests for OLS REST API debug endpoint to retriveve LLM response."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols.utils import suid
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
        "referenced_documents": [],
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
    assert (
        "conversation_id" in json
    ), "Conversation ID is not part of response as should be"
    assert suid.check_suid(json["conversation_id"])
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
