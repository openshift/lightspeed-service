"""General utilities for end-to-end tests."""

from typing import Optional

from httpx import Client


def get_http_client(url: str, user_token: Optional[str] = None) -> Client:
    """Get HTTP client."""
    client = Client(base_url=url, verify=False)  # noqa: S501
    if user_token:
        client.headers.update({"Authorization": f"Bearer {user_token}"})
    return client
