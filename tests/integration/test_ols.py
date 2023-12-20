from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "1"}


def test_readyz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "1"}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "This is the default endpoint for OLS",
        "status": "running",
    }


def test_status():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "This is the default endpoint for OLS",
        "status": "running",
    }


def test_feedback():
    # TODO: should we validate that the correct log messages are written?
    response = client.post(
        "/feedback", json={"conversation_id": 1234, "feedback_object": "blah"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "feedback received"}


def test_raw_prompt(monkeypatch):
    # the raw prompt should just return stuff from LangChainInterface, so mock that base method
    # model_context is what imports LangChainInterface, so we have to mock that particular usage/"instance"
    # of it in our tests

    import utils.model_context
    from tests.mock_classes.langchain_interface import mock_langchain_interface

    ml = mock_langchain_interface("test response")

    monkeypatch.setattr(utils.model_context, "LangChainInterface", ml)

    response = client.post(
        "/ols/raw_prompt", json={"conversation_id": "1234", "query": "test query"}
    )
    print(response)
    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": "1234",
        "query": "test query",
        "response": "test response",
    }
