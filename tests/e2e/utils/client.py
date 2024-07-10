"""General utilities for end-to-end tests."""

from typing import Optional

from httpx import Client

from tests.e2e.utils.constants import LLM_REST_API_TIMEOUT
from tests.e2e.utils.response import check_content_type


def get_http_client(url: str, user_token: Optional[str] = None) -> Client:
    """Get HTTP client."""
    client = Client(base_url=url, verify=False)  # noqa: S501, S113
    if user_token:
        client.headers.update({"Authorization": f"Bearer {user_token}"})
    return client


def perform_query(client, conversation_id, query):
    """Call service REST API using /query endpoint."""
    endpoint = "/v1/query"

    response = client.post(
        endpoint,
        json={"conversation_id": conversation_id, "query": query},
        timeout=LLM_REST_API_TIMEOUT,
    )
    check_content_type(response, "application/json")
    print(vars(response))
