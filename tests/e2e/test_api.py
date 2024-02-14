"""Integration tests for basic OLS REST API endpoints."""

import requests
from httpx import Client

client = Client(base_url="http://localhost:8080")

conversation_id = "12345678-abcd-0000-0123-456789abcdef"


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": {"status": "healthy"}}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": {"status": "healthy"}}


def test_raw_prompt():
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    r = client.post(
        "/v1/debug/query",
        json={"conversation_id": conversation_id, "query": "say hello"},
        timeout=20,
    )
    print(vars(r))
    response = r.json()

    assert r.status_code == requests.codes.ok
    assert response["conversation_id"] == conversation_id
    assert "hello" in response["response"].lower()


def test_invalid_question():
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    response = client.post(
        "/v1/query",
        json={"conversation_id": conversation_id, "query": "test query"},
        timeout=20,
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    expected_details = (
        "I can only answer questions about OpenShift and Kubernetes. "
        "Please rephrase your question"
    )
    expected_json = {
        "conversation_id": conversation_id,
        "response": expected_details,
        "referenced_documents": [],
    }
    assert response.json() == expected_json


def test_valid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for valid question and no yaml."""
    response = client.post(
        "/v1/query",
        json={"conversation_id": conversation_id, "query": "what is kubernetes?"},
        timeout=90,
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    json_response = response.json()
    json_response["conversation_id"] == conversation_id
    # assuming the response will be consistent
    assert (
        "Kubernetes is an open source container orchestration tool"
        in json_response["response"]
    )
    assert (
        "The following response was generated without access to reference content:"
        not in json_response["response"]
    )


def test_rag_question() -> None:
    """Ensure responses include rag references."""
    response = client.post(
        "/v1/query",
        json={"query": "what is the first step to install an openshift cluster?"},
        timeout=90,
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    json_response = response.json()
    assert len(json_response["referenced_documents"]) > 0
    assert "install" in json_response["referenced_documents"][0]
    assert "https://" in json_response["referenced_documents"][0]

    assert (
        "The following response was generated without access to reference content:"
        not in json_response["response"]
    )
