import requests
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_feedback() -> None:
    # TODO: should we validate that the correct log messages are written?
    response = client.post(
        "/feedback", json={"conversation_id": 1234, "feedback_object": "blah"}
    )
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "feedback received"}


def test_feedback_wrong_request() -> None:
    response = client.post("/feedback", json={})
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_wrong_not_filled_in_request() -> None:
    response = client.post(
        "/feedback", json={"conversation_id": 0, "feedback_object": None}
    )
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity
