"""Unit tests for OLS endpoint."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from ols import config, constants
from ols.constants import QueryMode

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints import ols  # noqa:E402
from ols.app.models.config import UserDataCollection  # noqa:E402
from ols.app.models.models import (  # noqa:E402
    Attachment,
    CacheEntry,
    LLMRequest,
    RagChunk,
    SummarizerResponse,
    TokenCounter,
)
from ols.utils import suid  # noqa:E402
from ols.utils.errors_parsing import DEFAULT_ERROR_MESSAGE  # noqa:E402
from ols.utils.redactor import Redactor, RegexFilter  # noqa:E402


@pytest.fixture(scope="function")
def _load_config():
    """Load config before unit tests."""
    config.reload_from_yaml_file("tests/config/test_app_endpoints.yaml")


@pytest.fixture
def auth():
    """Tuple containing user ID and user name, mocking auth. output."""
    # we can use any UUID, so let's use randomly generated one
    return ("2a3dfd17-1f42-4831-aaa6-e28e7cb8e26b", "name", False, "fake-token")


@pytest.mark.usefixtures("_load_config")
def test_retrieve_conversation_new_id():
    """Check the function to retrieve conversation ID."""
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=None)
    new_id = ols.retrieve_conversation_id(llm_request)
    assert suid.check_suid(new_id), "Improper conversation ID generated"


@pytest.mark.usefixtures("_load_config")
def test_retrieve_conversation_id_existing_id():
    """Check the function to retrieve conversation ID when one already exists."""
    old_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=old_id)
    new_id = ols.retrieve_conversation_id(llm_request)
    assert new_id == old_id, "Old (existing) ID should be retrieved."


@pytest.mark.usefixtures("_load_config")
@pytest.mark.usefixtures("_load_config")
def test_retrieve_attachments_on_no_input():
    """Check the function to retrieve attachments from payload when attachments are not send."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes", conversation_id=conversation_id
    )
    attachments = ols.retrieve_attachments(llm_request)
    # empty list should be returned
    assert attachments is not None
    assert len(attachments) == 0


@pytest.mark.usefixtures("_load_config")
def test_retrieve_attachments_on_empty_list():
    """Check the function to retrieve attachments from payload when list of attachments is empty."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes",
        conversation_id=conversation_id,
        attachments=[],
    )
    attachments = ols.retrieve_attachments(llm_request)
    # empty list should be returned
    assert attachments is not None
    assert len(attachments) == 0


@pytest.mark.usefixtures("_load_config")
def test_retrieve_attachments_on_proper_input():
    """Check the function to retrieve attachments from payload."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes",
        conversation_id=conversation_id,
        attachments=[
            Attachment(
                attachment_type="log",
                content_type="text/plain",
                content="this is attachment",
            ),
        ],
    )
    attachments = ols.retrieve_attachments(llm_request)
    # empty list should be returned
    assert attachments is not None
    assert len(attachments) == 1

    expected = Attachment(
        attachment_type="log", content_type="text/plain", content="this is attachment"
    )
    assert attachments[0] == expected


@pytest.mark.usefixtures("_load_config")
def test_retrieve_attachments_on_improper_attachment_type():
    """Check the function to retrieve attachments from payload."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes",
        conversation_id=conversation_id,
        attachments=[
            Attachment(
                attachment_type="not-correct-one",
                content_type="text/plain",
                content="this is attachment",
            ),
        ],
    )
    with pytest.raises(
        HTTPException, match="Attachment with improper type not-correct-one detected"
    ):
        ols.retrieve_attachments(llm_request)


@pytest.mark.usefixtures("_load_config")
def test_retrieve_attachments_on_improper_content_type():
    """Check the function to retrieve attachments from payload."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes",
        conversation_id=conversation_id,
        attachments=[
            Attachment(
                attachment_type="log",
                content_type="not/known",
                content="this is attachment",
            ),
        ],
    )
    with pytest.raises(
        HTTPException, match="Attachment with improper content type not/known detected"
    ):
        ols.retrieve_attachments(llm_request)


@pytest.mark.usefixtures("_load_config")
def test_store_conversation_history():
    """Test if operation to store conversation history to cache is called."""
    conversation_id = suid.get_suid()
    skip_user_id_check = False
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query)
    response = ""

    with patch("ols.config.conversation_cache.insert_or_append") as insert_or_append:
        ols.store_conversation_history(
            constants.DEFAULT_USER_UID,
            conversation_id,
            llm_request,
            response,
            [],
            {},
        )

        expected_history = CacheEntry(query=HumanMessage(query))
        insert_or_append.assert_called_with(
            constants.DEFAULT_USER_UID,
            conversation_id,
            expected_history,
            skip_user_id_check,
        )


@pytest.mark.usefixtures("_load_config")
def test_store_conversation_history_some_response():
    """Test if operation to store conversation history to cache is called."""
    user_id = "1234"
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query)
    response = "*response*"
    skip_user_id_check = False

    with patch("ols.config.conversation_cache.insert_or_append") as insert_or_append:
        ols.store_conversation_history(
            user_id, conversation_id, llm_request, response, [], skip_user_id_check
        )

    expected_history = CacheEntry(
        query=HumanMessage(query), response=AIMessage(response)
    )
    insert_or_append.assert_called_with(
        user_id, conversation_id, expected_history, skip_user_id_check
    )


@pytest.mark.usefixtures("_load_config")
def test_store_conversation_history_empty_user_id():
    """Test if basic input verification is done during history store operation."""
    user_id = ""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.store_conversation_history(
            user_id, conversation_id, llm_request, "", [], {}
        )
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.store_conversation_history(
            user_id, conversation_id, llm_request, None, [], {}
        )


@pytest.mark.usefixtures("_load_config")
def test_store_conversation_history_improper_user_id():
    """Test if basic input verification is done during history store operation."""
    user_id = "::::"
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.store_conversation_history(
            user_id, conversation_id, llm_request, "", [], {}
        )


@pytest.mark.usefixtures("_load_config")
def test_store_conversation_history_improper_conversation_id():
    """Test if basic input verification is done during history store operation."""
    conversation_id = "::::"
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    with pytest.raises(HTTPException, match="Invalid conversation ID"):
        ols.store_conversation_history(
            constants.DEFAULT_USER_UID, conversation_id, llm_request, "", [], {}
        )


@pytest.mark.usefixtures("_load_config")
def test_store_conversation_history_with_tool_data():
    """Test if tool_calls and tool_results are stored in cache entry."""
    conversation_id = suid.get_suid()
    skip_user_id_check = False
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query)
    response = "Here is information about Kubernetes"
    tool_calls = [{"name": "get_info", "args": {"topic": "k8s"}, "id": "tc1"}]
    tool_results = [{"id": "tc1", "status": "success", "content": "K8s info"}]

    with patch("ols.config.conversation_cache.insert_or_append") as insert_or_append:
        ols.store_conversation_history(
            constants.DEFAULT_USER_UID,
            conversation_id,
            llm_request,
            response,
            [],
            {},
            skip_user_id_check,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

        expected_history = CacheEntry(
            query=HumanMessage(query),
            response=AIMessage(response),
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        insert_or_append.assert_called_with(
            constants.DEFAULT_USER_UID,
            conversation_id,
            expected_history,
            skip_user_id_check,
        )


@pytest.mark.usefixtures("_load_config")
def test_store_conversation_history_without_tool_data():
    """Test that tool_calls and tool_results default to empty lists."""
    conversation_id = suid.get_suid()
    skip_user_id_check = False
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query)
    response = "Here is information"

    with patch("ols.config.conversation_cache.insert_or_append") as insert_or_append:
        ols.store_conversation_history(
            constants.DEFAULT_USER_UID,
            conversation_id,
            llm_request,
            response,
            [],
            {},
            skip_user_id_check,
        )

        # Verify the cache entry has empty tool_calls and tool_results
        call_args = insert_or_append.call_args
        cache_entry = call_args[0][2]  # Third positional argument is the cache entry
        assert cache_entry.tool_calls == []
        assert cache_entry.tool_results == []


@pytest.mark.usefixtures("_load_config")
def test_query_filter_no_redact_filters():
    """Test the function to redact query when no filters are setup."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    result = ols.redact_query(conversation_id, llm_request)
    assert result is not None
    assert result.query == query


@pytest.mark.usefixtures("_load_config")
def test_query_filter_with_one_redact_filter():
    """Test the function to redact query when filter is setup."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)

    # use one custom filter
    assert config.ols_config.query_filters is not None
    q = Redactor(config.ols_config.query_filters)
    q.regex_filters = [
        RegexFilter(
            pattern=re.compile(r"Kubernetes"),
            name="kubernetes-filter",
            replace_with="FooBar",
        )
    ]
    config._query_filters = q

    result = ols.redact_query(conversation_id, llm_request)
    assert result is not None
    assert result.query == "Tell me about FooBar"


@pytest.mark.usefixtures("_load_config")
def test_query_filter_with_two_redact_filters():
    """Test the function to redact query when multiple filters are setup."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)

    # use two custom filters
    assert config.ols_config.query_filters is not None
    q = Redactor(config.ols_config.query_filters)
    q.regex_filters = [
        RegexFilter(
            pattern=re.compile(r"Kubernetes"),
            name="kubernetes-filter",
            replace_with="FooBar",
        ),
        RegexFilter(
            pattern=re.compile(r"FooBar"),
            name="FooBar-filter",
            replace_with="Baz",
        ),
    ]
    config._query_filters = q

    result = ols.redact_query(conversation_id, llm_request)
    assert result is not None
    assert result.query == "Tell me about Baz"


@pytest.mark.usefixtures("_load_config")
def test_query_filter_on_redact_error():
    """Test the function to redact query when redactor raises an error."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    with pytest.raises(HTTPException, match="Error while redacting query"):
        with patch("ols.utils.redactor.Redactor.redact", side_effect=Exception):
            ols.redact_query(conversation_id, llm_request)


@pytest.mark.usefixtures("_load_config")
def test_attachments_redact_on_no_filters_defined():
    """Test the function to redact attachments when no filters are setup."""
    conversation_id = suid.get_suid()
    attachments = [
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by Kubernetes",
        ),
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by OpenShift",
        ),
    ]
    # try to redact all attachments
    redacted = ols.redact_attachments(conversation_id, attachments)
    assert redacted is not None

    # no filters are set up, so the redacted attachments must
    # be the same as original ones
    assert redacted == attachments


@pytest.mark.usefixtures("_load_config")
def test_attachments_redact_with_one_filter_defined():
    """Test the function to redact attachments when one filter is setup."""
    conversation_id = suid.get_suid()
    attachments = [
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by Kubernetes",
        ),
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by OpenShift",
        ),
    ]

    # use two custom filters
    assert config.ols_config.query_filters is not None
    q = Redactor(config.ols_config.query_filters)
    q.regex_filters = [
        RegexFilter(
            pattern=re.compile(r"Kubernetes"),
            name="kubernetes-filter",
            replace_with="FooBar",
        ),
    ]
    config._query_filters = q

    # try to redact all attachments
    redacted = ols.redact_attachments(conversation_id, attachments)
    assert redacted is not None

    # one filter must be applied
    assert redacted[0] == Attachment(
        attachment_type="log",
        content_type="text/plain",
        content="Log created by FooBar",
    )
    # no filters should be applied
    assert redacted[1] == Attachment(
        attachment_type="log",
        content_type="text/plain",
        content="Log created by OpenShift",
    )


@pytest.mark.usefixtures("_load_config")
def test_attachments_redact_with_two_filters_defined():
    """Test the function to redact attachments when two filters are setup."""
    conversation_id = suid.get_suid()
    attachments = [
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by Kubernetes",
        ),
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by OpenShift",
        ),
    ]

    # use two custom filters
    assert config.ols_config.query_filters is not None
    q = Redactor(config.ols_config.query_filters)
    q.regex_filters = [
        RegexFilter(
            pattern=re.compile(r"Kubernetes"),
            name="kubernetes-filter",
            replace_with="FooBar",
        ),
        RegexFilter(
            pattern=re.compile(r"FooBar"),
            name="FooBar-filter",
            replace_with="Baz",
        ),
    ]
    config._query_filters = q

    # try to redact all attachments
    redacted = ols.redact_attachments(conversation_id, attachments)
    assert redacted is not None

    # both filters must be applied
    assert redacted[0] == Attachment(
        attachment_type="log",
        content_type="text/plain",
        content="Log created by Baz",
    )
    # no filters should be applied
    assert redacted[1] == Attachment(
        attachment_type="log",
        content_type="text/plain",
        content="Log created by OpenShift",
    )


@pytest.mark.usefixtures("_load_config")
def test_attachments_redact_on_redact_error():
    """Test the function to redact attachments when redactor raises an error."""
    conversation_id = suid.get_suid()
    attachments = [
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by Kubernetes",
        ),
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="Log created by OpenShift",
        ),
    ]

    # try to redact all attachments
    with pytest.raises(HTTPException, match="Error while redacting attachment"):
        with patch("ols.utils.redactor.Redactor.redact", side_effect=Exception):
            ols.redact_attachments(conversation_id, attachments)


@pytest.mark.usefixtures("_load_config")
def test_conversation_request(auth):
    """Test conversation request API endpoint."""
    with (
        patch("ols.app.endpoints.ols.DocsSummarizer") as mock_docs_summarizer,
        patch("ols.config.conversation_cache.get"),
    ):
        mock_response = (
            "Kubernetes is an open-source container-orchestration system..."  # summary
        )
        mock_docs_summarizer.return_value.create_response.return_value = (
            SummarizerResponse(
                response=mock_response,
                rag_chunks=[],
                history_truncated=False,
                token_counter=None,
            )
        )
        llm_request = LLMRequest(query="Tell me about Kubernetes")
        response = ols.conversation_request(llm_request, auth)
        assert (
            response.response
            == "Kubernetes is an open-source container-orchestration system..."
        )
        assert suid.check_suid(
            response.conversation_id
        ), "Improper conversation ID returned"


@pytest.mark.usefixtures("_load_config")
def test_conversation_request_dedup_ref_docs(auth):
    """Test deduplication of referenced docs."""
    with (
        patch("ols.app.endpoints.ols.DocsSummarizer") as mock_docs_summarizer,
        patch("ols.config.conversation_cache.get"),
    ):
        mock_rag_chunk = [
            RagChunk(text="text1", doc_url="url-b", doc_title="title-b"),
            RagChunk(
                text="text2", doc_url="url-b", doc_title="title-b"
            ),  # duplicate doc
            RagChunk(text="text3", doc_url="url-a", doc_title="title-a"),
        ]
        mock_docs_summarizer.return_value.create_response.return_value = (
            SummarizerResponse(
                response="some response",
                rag_chunks=mock_rag_chunk,
                history_truncated=False,
                token_counter=None,
            )
        )
        llm_request = LLMRequest(query="some query")
        response = ols.conversation_request(llm_request, auth)

        assert len(response.referenced_documents) == 2
        assert response.referenced_documents[0].doc_url == "url-b"
        assert response.referenced_documents[0].doc_title == "title-b"
        assert response.referenced_documents[1].doc_url == "url-a"
        assert response.referenced_documents[1].doc_title == "title-a"


@pytest.mark.usefixtures("_load_config")
def test_generate_response_valid_subject():
    """Test how generate_response function checks validation results."""
    # mock the DocsSummarizer
    mock_response = (
        "Kubernetes is an open-source container-orchestration system..."  # summary
    )
    with patch("ols.app.endpoints.ols.DocsSummarizer") as mock_docs_summarizer:
        mock_docs_summarizer.return_value.create_response.return_value = (
            SummarizerResponse(
                mock_response,
                [],
                False,
                token_counter=None,
            )
        )

        # prepare arguments for DocsSummarizer
        conversation_id = suid.get_suid()
        llm_request = LLMRequest(query="Tell me about Kubernetes")
        previous_input = []

        # try to get response
        summarizer_response = ols.generate_response(
            conversation_id, llm_request, previous_input
        )

        # check the response
        assert "Kubernetes" in summarizer_response.response
        assert summarizer_response.rag_chunks == []
        assert summarizer_response.history_truncated is False


@pytest.mark.usefixtures("_load_config")
def test_generate_response_passes_mode_to_summarizer():
    """Test that generate_response passes mode from LLMRequest to DocsSummarizer."""
    mock_response = "some response"
    with (
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer.__init__",
            return_value=None,
        ) as mock_init,
        patch(
            "ols.src.query_helpers.docs_summarizer.DocsSummarizer.create_response",
            return_value=SummarizerResponse(
                mock_response, [], False, token_counter=None
            ),
        ),
    ):
        conversation_id = suid.get_suid()
        llm_request = LLMRequest(
            query="My pod is crashlooping", mode=QueryMode.TROUBLESHOOTING
        )
        previous_input: list = []

        ols.generate_response(conversation_id, llm_request, previous_input)

        _, kwargs = mock_init.call_args
        assert kwargs.get("mode") == QueryMode.TROUBLESHOOTING


@pytest.mark.usefixtures("_load_config")
def test_generate_response_on_summarizer_error():
    """Test how generate_response function checks validation results."""
    with patch(
        "ols.src.query_helpers.docs_summarizer.DocsSummarizer.create_response"
    ) as mock_summarize:
        # mock the DocsSummarizer
        mock_summarize.side_effect = Exception  # any exception might occur

        # prepare arguments for DocsSummarizer
        conversation_id = suid.get_suid()

        llm_request = LLMRequest(query="Tell me about Kubernetes")
        previous_input = None

        # try to get response
        with pytest.raises(HTTPException, match=DEFAULT_ERROR_MESSAGE):
            ols.generate_response(conversation_id, llm_request, previous_input)


@pytest.fixture
def transcripts_location(tmpdir):
    """Fixture sets feedback location to tmpdir and return the path."""
    config.ols_config.user_data_collection = UserDataCollection(
        transcripts_disabled=False, transcripts_storage=tmpdir.strpath
    )
    return tmpdir.strpath


def test_transcripts_are_not_stored_when_disabled(transcripts_location, auth):
    """Test nothing is stored when the transcript collection is disabled."""
    with (
        patch(
            "ols.app.endpoints.ols.config.ols_config.user_data_collection.transcripts_disabled",
            True,
        ),
        patch(
            "ols.app.endpoints.ols.generate_response",
            return_value=SummarizerResponse("something", [], False, None),
        ),
        patch(
            "ols.app.endpoints.ols.store_conversation_history",
            return_value=None,
        ),
    ):
        llm_request = LLMRequest(query="Tell me about Kubernetes")
        response = ols.conversation_request(llm_request, auth)
        assert response
        assert response.response == "something"

        transcript_dir = Path(transcripts_location)
        assert list(transcript_dir.glob("*/*/*.json")) == []


def test_construct_transcripts_path(transcripts_location):
    """Test for the helper function construct_transcripts_path."""
    user_id = "00000000-0000-0000-0000-000000000000"
    conversation_id = "11111111-1111-1111-1111-111111111111"
    path = ols.construct_transcripts_path(user_id, conversation_id)

    assert str(path.resolve()).endswith(f"{user_id}/{conversation_id}")


def test_store_transcript(transcripts_location):
    """Test transcript is successfully stored."""
    user_id = suid.get_suid()
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    response = "Kubernetes is ..."
    rag_chunks = [
        RagChunk(text="text1", doc_url="url1", doc_title="title1"),
        RagChunk(text="text2", doc_url="url2", doc_title="title2"),
    ]
    truncated = True
    tool_calls = [
        {"name": "tool1", "args": {}, "id": "1", "type": "tool_call"},
    ]
    tool_results = [
        {
            "id": "1",
            "status": "success",
            "content": "bla",
            "type": "tool_result",
            "round": 1,
        }
    ]
    attachments = [
        Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="this is attachment",
        )
    ]

    ols.store_transcript(
        user_id,
        conversation_id,
        query,
        llm_request,
        response,
        rag_chunks,
        truncated,
        tool_calls,
        tool_results,
        attachments,
    )

    transcript_dir = Path(transcripts_location) / user_id / conversation_id

    # check file exists in the expected path
    assert transcript_dir.exists()
    transcripts = list(transcript_dir.glob("*.json"))
    assert len(transcripts) == 1

    # check the transcript json content
    with open(transcripts[0]) as f:
        transcript = json.loads(f.read())
    # we don't really care about the timestamp, so let's just set it to
    # a fixed value
    transcript["metadata"]["timestamp"] = "fake-timestamp"
    assert transcript == {
        "metadata": {
            "provider": None,
            "model": None,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "mode": "ask",
            "timestamp": "fake-timestamp",
        },
        "redacted_query": query,
        "llm_response": response,
        "rag_chunks": [
            {"text": "text1", "doc_url": "url1", "doc_title": "title1"},
            {"text": "text2", "doc_url": "url2", "doc_title": "title2"},
        ],
        "truncated": truncated,
        "tool_calls": [
            {
                "name": "tool1",
                "args": {},
                "id": "1",
                "type": "tool_result",
                "status": "success",
                "content": "bla",
                "round": 1,
            }
        ],
        "attachments": [
            {
                "attachment_type": "log",
                "content_type": "text/plain",
                "content": "this is attachment",
            }
        ],
    }


def test_calc_tokens_no_token_counter():
    """Test calc_tokens returns 0 when token counter is None."""
    assert ols.calc_tokens(None, "input_tokens") == 0
    assert ols.calc_tokens(None, "output_tokens") == 0
    assert ols.calc_tokens(None, "reasoning_tokens") == 0


def test_calc_tokens_for_token_counter():
    """Test calc_tokens extracts the correct attribute from a token counter."""
    token_counter = TokenCounter(
        input_tokens=100, output_tokens=200, reasoning_tokens=50
    )
    assert ols.calc_tokens(token_counter, "input_tokens") == 100
    assert ols.calc_tokens(token_counter, "output_tokens") == 200
    assert ols.calc_tokens(token_counter, "reasoning_tokens") == 50


def test_consume_tokens_no_quota_limiters():
    """Test the function consume_tokens if no quota limiters are configured."""
    quota_limiters = None
    ols.consume_tokens(quota_limiters, None, "user_id", 1, 1, "", "")


def test_consume_tokens_empty_quota_limiters():
    """Test the function consume_tokens if empty list of quota limiters are configured."""
    quota_limiters = []
    ols.consume_tokens(quota_limiters, None, "user_id", 1, 1, "", "")


def test_consume_tokens_with_existing_quota_limiter():
    """Test the function consume_tokens for configured quota limiter."""

    class MockQuotaLimiter:
        """Mocked quota limiter."""

        def consume_tokens(self, input_tokens=0, output_tokens=0, subject_id=""):
            self._input_tokens = input_tokens
            self._output_tokens = output_tokens
            self._subject_id = subject_id

    mock_quota_limiter = MockQuotaLimiter()

    quota_limiters = [mock_quota_limiter]
    token_usage_history = None
    input_tokens = 1
    output_tokens = 2
    user_id = "user_id"
    provider = "provider"
    model = "model"
    ols.consume_tokens(
        quota_limiters,
        token_usage_history,
        user_id,
        input_tokens,
        output_tokens,
        provider,
        model,
    )
    assert mock_quota_limiter._input_tokens == input_tokens
    assert mock_quota_limiter._output_tokens == output_tokens
    assert mock_quota_limiter._subject_id == user_id


def test_consume_tokens_multiple_quota_limiters():
    """Test the function consume_tokens for more configured quota limiter."""

    class MockQuotaLimiter:
        def consume_tokens(self, input_tokens=0, output_tokens=0, subject_id=""):
            self._input_tokens = input_tokens
            self._output_tokens = output_tokens
            self._subject_id = subject_id

    mock_quota_limiter1 = MockQuotaLimiter()
    mock_quota_limiter2 = MockQuotaLimiter()

    quota_limiters = [mock_quota_limiter1, mock_quota_limiter2]
    token_usage_history = None
    input_tokens = 1
    output_tokens = 2
    user_id = "user_id"
    provider = "provider"
    model = "model"
    ols.consume_tokens(
        quota_limiters,
        token_usage_history,
        user_id,
        input_tokens,
        output_tokens,
        provider,
        model,
    )

    for mock_quota_limiter in [mock_quota_limiter1, mock_quota_limiter2]:
        assert mock_quota_limiter._input_tokens == input_tokens
        assert mock_quota_limiter._output_tokens == output_tokens
        assert mock_quota_limiter._subject_id == user_id


def test_consume_tokens_token_usage_history():
    """Test the function consume_tokens when token_usage_history is setup."""

    class MockTokenUsageHistory:
        def consume_tokens(
            self, user_id="", provider="", model="", input_tokens=0, output_tokens=0
        ):
            self._user_id = user_id
            self._provider = provider
            self._model = model
            self._input_tokens = input_tokens
            self._output_tokens = output_tokens

    quota_limiters = []
    token_usage_history = MockTokenUsageHistory()
    input_tokens = 1
    output_tokens = 2
    user_id = "user_id"
    provider = "provider"
    model = "model"
    ols.consume_tokens(
        quota_limiters,
        token_usage_history,
        user_id,
        input_tokens,
        output_tokens,
        provider,
        model,
    )

    assert token_usage_history._user_id == user_id
    assert token_usage_history._provider == provider
    assert token_usage_history._model == model
    assert token_usage_history._input_tokens == input_tokens
    assert token_usage_history._output_tokens == output_tokens


def test_consume_tokens_token_limiters_and_token_usage_history():
    """Test the function consume_tokens when token_usage_history is setup."""

    class MockQuotaLimiter:
        def consume_tokens(self, input_tokens=0, output_tokens=0, subject_id=""):
            self._input_tokens = input_tokens
            self._output_tokens = output_tokens
            self._subject_id = subject_id

    class MockTokenUsageHistory:
        def consume_tokens(
            self, user_id="", provider="", model="", input_tokens=0, output_tokens=0
        ):
            self._user_id = user_id
            self._provider = provider
            self._model = model
            self._input_tokens = input_tokens
            self._output_tokens = output_tokens

    mock_quota_limiter1 = MockQuotaLimiter()
    mock_quota_limiter2 = MockQuotaLimiter()

    quota_limiters = [mock_quota_limiter1, mock_quota_limiter2]
    token_usage_history = MockTokenUsageHistory()
    input_tokens = 1
    output_tokens = 2
    user_id = "user_id"
    provider = "provider"
    model = "model"
    ols.consume_tokens(
        quota_limiters,
        token_usage_history,
        user_id,
        input_tokens,
        output_tokens,
        provider,
        model,
    )

    assert token_usage_history._user_id == user_id
    assert token_usage_history._provider == provider
    assert token_usage_history._model == model
    assert token_usage_history._input_tokens == input_tokens
    assert token_usage_history._output_tokens == output_tokens

    for mock_quota_limiter in [mock_quota_limiter1, mock_quota_limiter2]:
        assert mock_quota_limiter._input_tokens == input_tokens
        assert mock_quota_limiter._output_tokens == output_tokens
        assert mock_quota_limiter._subject_id == user_id


def test_check_token_available_no_quota_limiters():
    """Test the function check_tokens_available if no quota limiters are configured."""
    quota_limiters = None
    ols.check_tokens_available(quota_limiters, "user_id")


def test_check_token_available_empty_limiters():
    """Test the function check_tokens_available if emply list of quota limiters are configured."""
    quota_limiters = []
    ols.check_tokens_available(quota_limiters, "user_id")


def test_check_token_available_configured_quota_limiter():
    """Test the function check_tokens_available if one quota limiter is configured."""

    class MockQuotaLimiter:
        def ensure_available_quota(self, subject_id=""):
            self._subject_id = subject_id

    mock_quota_limiter = MockQuotaLimiter()
    quota_limiters = [mock_quota_limiter]
    user_id = "user_id"

    ols.check_tokens_available(quota_limiters, user_id)
    assert mock_quota_limiter._subject_id == user_id


def test_check_token_available_on_exceed_error():
    """Test the function check_tokens_available if quota exceed error is thrown."""

    class MockQuotaLimiter:
        def ensure_available_quota(self, subject_id=""):
            raise Exception("Raised exception")

    mock_quota_limiter = MockQuotaLimiter()
    quota_limiters = [mock_quota_limiter]
    user_id = "user_id"

    expected = (
        "500: {'response': 'The quota has been exceeded', 'cause': 'Raised exception'}"
    )
    with pytest.raises(HTTPException, match=expected):
        ols.check_tokens_available(quota_limiters, user_id)


def test_get_available_quotas_no_quota_limiters():
    """Test the function get_available_quotas when no quota limiters are setup."""
    quotas = ols.get_available_quotas(None, "user_id")
    assert quotas == {}


def test_get_available_quotas_one_quota_limiter():
    """Test the function get_available_quotas when one quota limiter is setup."""

    class MockQuotaLimiter:
        def __init__(self, available_quota=0):
            self._available_quota = available_quota

        def available_quota(self, subject_id=""):
            return self._available_quota

    mock_quota_limiter = MockQuotaLimiter(10)
    quotas = ols.get_available_quotas([mock_quota_limiter], "user_id")
    assert quotas == {
        "MockQuotaLimiter": 10,
    }


def test_get_available_quotas_two_quota_limiters():
    """Test the function get_available_quotas when two quota limiters are setup."""

    class MockQuotaLimiter1:
        def __init__(self, available_quota=0):
            self._available_quota = available_quota

        def available_quota(self, subject_id=""):
            return self._available_quota

    class MockQuotaLimiter2:
        def __init__(self, available_quota=0):
            self._available_quota = available_quota

        def available_quota(self, subject_id=""):
            return self._available_quota

    mock_quota_limiter_1 = MockQuotaLimiter1(10)
    mock_quota_limiter_2 = MockQuotaLimiter2(20)
    quotas = ols.get_available_quotas(
        [mock_quota_limiter_1, mock_quota_limiter_2], "user_id"
    )
    assert quotas == {
        "MockQuotaLimiter1": 10,
        "MockQuotaLimiter2": 20,
    }


def test_merge_tools_info():
    """Test the function merge_tools_info."""
    tool_calls = [
        {"name": "tool1", "args": {}, "id": "1", "type": "tool_call"},
        {"name": "tool2", "args": {}, "id": "2", "type": "tool_call"},
    ]
    tool_results = [
        {
            "id": "1",
            "status": "success",
            "content": "bla",
            "type": "tool_result",
            "round": 1,
        },
        {
            "id": "2",
            "status": "error",
            "content": "bla",
            "type": "tool_result",
            "round": 1,
        },
    ]
    merged_tools = ols.merge_tools_info(tool_calls, tool_results)
    assert len(merged_tools) == 2
    assert merged_tools[0] == {
        "name": "tool1",
        "args": {},
        "id": "1",
        "type": "tool_result",
        "status": "success",
        "content": "bla",
        "round": 1,
    }
    assert merged_tools[1] == {
        "name": "tool2",
        "args": {},
        "id": "2",
        "type": "tool_result",
        "status": "error",
        "content": "bla",
        "round": 1,
    }


def test_merge_tools_info_failures(caplog):
    """Test the function merge_tools_info when no tool calls are provided."""
    tool_calls = []
    tool_results = [{"id": "1"}]
    merged_tools = ols.merge_tools_info(tool_calls, tool_results)
    assert merged_tools == []
    assert (
        "tool_calls and tool_results must have the same number of items" in caplog.text
    )

    tool_calls = [{"id": "1"}, {"id": "1"}]
    tool_results = [{"id": "1"}, {"id": "2"}]
    merged_tools = ols.merge_tools_info(tool_calls, tool_results)
    assert merged_tools == []
    assert "tool_calls must have unique ids" in caplog.text

    tool_calls = [{"id": "1"}, {"id": "2"}]
    tool_results = [{"id": "1"}, {"id": "1"}]
    merged_tools = ols.merge_tools_info(tool_calls, tool_results)
    assert merged_tools == []
    assert "tool_results must have unique ids" in caplog.text

    tool_calls = [{"id": "1"}, {"id": "2"}]
    tool_results = [{"id": "3"}, {"id": "1"}]
    merged_tools = ols.merge_tools_info(tool_calls, tool_results)
    assert merged_tools == []
    assert "tool_calls and tool_results must have the same number of ids" in caplog.text
