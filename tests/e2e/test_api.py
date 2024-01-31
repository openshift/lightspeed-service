"""Integration tests for basic OLS REST API endpoints."""

import requests
from httpx import Client

client = Client(base_url="http://localhost:8080")

conversation_id = "12345678-abcd-0000-0123-456789abcdef"


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


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
    assert response["query"] == "say hello"
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
    expected_details = str(
        {
            "detail": {
                "response": "I can only answer questions about \
            OpenShift and Kubernetes. Please rephrase your question"
            }
        }
    )
    expected_json = {
        "conversation_id": conversation_id,
        "model": None,
        "provider": None,
        "query": "test query",
        "response": expected_details,
    }
    assert response.json() == expected_json
