"""Integration tests for basic OLS REST API endpoints."""

import requests
from httpx import Client

client = Client(base_url="http://localhost:8080")


def test_readiness() -> None:
    """Test handler for /readiness REST API endpoint."""
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_liveness() -> None:
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_raw_prompt() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    r = client.post(
        "/v1/debug/query",
        json={"conversation_id": "1234", "query": "say hello"},
        timeout=20,
    )
    print(vars(r))
    response = r.json()

    assert r.status_code == requests.codes.ok
    assert response["conversation_id"] == "1234"
    assert response["query"] == "say hello"
    assert "hello" in response["response"].lower()


def test_invalid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    response = client.post(
        "/v1/query", json={"conversation_id": "1234", "query": "test query"}, timeout=20
    )
    print(vars(response))
    assert response.status_code == requests.codes.ok
    response.json().get("response") == str(
        {
            "detail": {
                "response": "I can only answer questions about \
            OpenShift and Kubernetes. Please rephrase your question"
            }
        }
    )
