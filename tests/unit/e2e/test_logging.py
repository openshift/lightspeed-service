"""Tests for concise E2E test logging."""

from typing import ClassVar

from tests.e2e.utils.client import perform_query


class _Response:
    headers: ClassVar[dict[str, str]] = {"content-type": "application/json"}


class _Client:
    def post(self, *args, **kwargs):
        return _Response()


def test_perform_query_does_not_dump_http_response(capsys):
    """HTTP helper does not print the complete response object."""
    perform_query(_Client(), "conversation-id", "question")

    assert capsys.readouterr().out == ""
