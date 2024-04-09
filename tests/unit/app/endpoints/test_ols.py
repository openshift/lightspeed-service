"""Unit tests for OLS endpoint."""

import json
import re
from http import HTTPStatus
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from langchain.schema import AIMessage, HumanMessage

from ols import constants
from ols.app.endpoints import ols
from ols.app.models.config import UserDataCollection
from ols.app.models.models import LLMRequest
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.utils import config, suid
from ols.utils.query_filter import QueryFilter, RegexFilter
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture(scope="module")
def _load_config():
    """Load config before unit tests."""
    config.init_config("tests/config/test_app_endpoints.yaml")


@pytest.fixture
def auth():
    """Tuple containing user ID and user name, mocking auth. output."""
    # we can use any UUID, so let's use randomly generated one
    return ("2a3dfd17-1f42-4831-aaa6-e28e7cb8e26b", "name")


def test_retrieve_conversation_new_id(_load_config):
    """Check the function to retrieve conversation ID."""
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=None)
    new_id = ols.retrieve_conversation_id(llm_request)
    assert suid.check_suid(new_id), "Improper conversation ID generated"


def test_retrieve_conversation_id_existing_id(_load_config):
    """Check the function to retrieve conversation ID when one already exists."""
    old_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=old_id)
    new_id = ols.retrieve_conversation_id(llm_request)
    assert new_id == old_id, "Old (existing) ID should be retrieved." ""


def test_retrieve_previous_input_no_previous_history(_load_config):
    """Check how function to retrieve previous input handle empty history."""
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=None)
    input = ols.retrieve_previous_input(constants.DEFAULT_USER_UID, llm_request)
    assert input == []


def test_retrieve_previous_input_empty_user_id(_load_config):
    """Check how function to retrieve previous input handle empty user ID."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes", conversation_id=conversation_id
    )
    # cache must check if user ID is correct
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.retrieve_previous_input("", llm_request)
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.retrieve_previous_input(None, llm_request)


def test_retrieve_previous_input_improper_user_id(_load_config):
    """Check how function to retrieve previous input handle improper user ID."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes", conversation_id=conversation_id
    )
    # cache must check if user ID is correct
    with pytest.raises(HTTPException, match="Invalid user ID improper_user_id"):
        ols.retrieve_previous_input("improper_user_id", llm_request)


@patch("ols.utils.config.conversation_cache.get")
def test_retrieve_previous_input_for_previous_history(get, _load_config):
    """Check how function to retrieve previous input handle existing history."""
    conversation_id = suid.get_suid()
    get.return_value = "input"
    llm_request = LLMRequest(
        query="Tell me about Kubernetes", conversation_id=conversation_id
    )
    previous_input = ols.retrieve_previous_input(
        constants.DEFAULT_USER_UID, llm_request
    )
    assert previous_input == "input"


@patch("ols.utils.config.conversation_cache.insert_or_append")
def test_store_conversation_history(insert_or_append, _load_config):
    """Test if operation to store conversation history to cache is called."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query)
    response = ""
    ols.store_conversation_history(
        constants.DEFAULT_USER_UID, conversation_id, llm_request, response
    )
    expected_history = [
        HumanMessage(content="Tell me about Kubernetes"),
        AIMessage(content=""),
    ]
    insert_or_append.assert_called_with(
        constants.DEFAULT_USER_UID, conversation_id, expected_history
    )


@patch("ols.utils.config.conversation_cache.insert_or_append")
def test_store_conversation_history_some_response(insert_or_append, _load_config):
    """Test if operation to store conversation history to cache is called."""
    user_id = "1234"
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query)
    response = "*response*"
    ols.store_conversation_history(user_id, conversation_id, llm_request, response)
    expected_history = [
        HumanMessage(content="Tell me about Kubernetes"),
        AIMessage(content="*response*"),
    ]
    insert_or_append.assert_called_with(user_id, conversation_id, expected_history)


def test_store_conversation_history_empty_user_id(_load_config):
    """Test if basic input verification is done during history store operation."""
    user_id = ""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.store_conversation_history(user_id, conversation_id, llm_request, "")
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.store_conversation_history(user_id, conversation_id, llm_request, None)


def test_store_conversation_history_improper_user_id(_load_config):
    """Test if basic input verification is done during history store operation."""
    user_id = "::::"
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    with pytest.raises(HTTPException, match="Invalid user ID"):
        ols.store_conversation_history(user_id, conversation_id, llm_request, "")


def test_store_conversation_history_improper_conversation_id(_load_config):
    """Test if basic input verification is done during history store operation."""
    conversation_id = "::::"
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    with pytest.raises(HTTPException, match="Invalid conversation ID"):
        ols.store_conversation_history(
            constants.DEFAULT_USER_UID, conversation_id, llm_request, ""
        )


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.KEYWORD,
)
@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
def test_validate_question_valid_kw(llm_validate_question_mock, _load_config):
    """Check the behaviour of validate_question function using valid keyword."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes?"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    resp = ols.validate_question(conversation_id, llm_request)

    assert resp
    assert llm_validate_question_mock.call_count == 0


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.KEYWORD,
)
def test_validate_question_invalid_kw(_load_config):
    """Check the behaviour of validate_question function using invalid keyword."""
    conversation_id = suid.get_suid()
    query = "What does 42 signify ?"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    resp = ols.validate_question(conversation_id, llm_request)
    assert not resp


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
def test_validate_question(validate_question_mock, _load_config):
    """Check the behaviour of validate_question function."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    ols.validate_question(conversation_id, llm_request)
    validate_question_mock.assert_called_with(conversation_id, query)


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
def test_validate_question_on_configuration_error(validate_question_mock, _load_config):
    """Check the behaviour of validate_question function when wrong configuration is detected."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    validate_question_mock.side_effect = LLMConfigurationError

    # HTTP exception should be raises
    with pytest.raises(HTTPException, match="Unable to process this request"):
        ols.validate_question(conversation_id, llm_request)


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
def test_validate_question_on_validation_error(validate_question_mock, _load_config):
    """Check the behaviour of validate_question function when query is not validated properly."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    validate_question_mock.side_effect = (
        ValueError  # any exception except HTTPException can be used there
    )

    # HTTP exception should be raises
    with pytest.raises(HTTPException, match="Error while validating question"):
        ols.validate_question(conversation_id, llm_request)


@patch(
    "ols.app.endpoints.ols.config.ols_config.query_validation_method",
    constants.QueryValidationMethod.DISABLED,
)
@patch("ols.app.endpoints.ols._validate_question_keyword")
@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
def test_validate_question_disabled(
    validate_question_llm_mock, validate_question_kw_mock
):
    """Check the behaviour of validate_question function when it is disabled."""
    conversation_id = suid.get_suid()
    query = "What does 42 signify ?"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    resp = ols.validate_question(conversation_id, llm_request)

    assert validate_question_llm_mock.call_count == 0
    assert validate_question_kw_mock.call_count == 0
    assert resp


def test_query_filter_no_redact_filters(_load_config):
    """Test the function to redact query when no filters are setup."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    result = ols.redact_query(conversation_id, llm_request)
    assert result is not None
    assert result.query == query


def test_query_filter_with_one_redact_filter(_load_config):
    """Test the function to redact query when filter is setup."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)

    # use one custom filter
    q = QueryFilter()
    q.regex_filters = [
        RegexFilter(
            pattern=re.compile(r"Kubernetes"),
            name="kubernetes-filter",
            replace_with="FooBar",
        )
    ]
    config.query_redactor = q

    result = ols.redact_query(conversation_id, llm_request)
    assert result is not None
    assert result.query == "Tell me about FooBar"


def test_query_filter_with_two_redact_filters(_load_config):
    """Test the function to redact query when multiple filters are setup."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)

    # use two custom filters
    q = QueryFilter()
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
    config.query_redactor = q

    result = ols.redact_query(conversation_id, llm_request)
    assert result is not None
    assert result.query == "Tell me about Baz"


@patch("ols.utils.config.query_redactor")
def test_query_filter_on_redact_error(mock_redact_query, _load_config):
    """Test the function to redact query when redactor raises an error."""
    conversation_id = suid.get_suid()
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    mock_redact_query.redact_query.side_effect = Exception
    with pytest.raises(HTTPException, match="Error while redacting query"):
        ols.redact_query(conversation_id, llm_request)


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
@patch("ols.src.query_helpers.docs_summarizer.DocsSummarizer.summarize")
@patch("ols.utils.config.conversation_cache.get")
def test_conversation_request(
    mock_conversation_cache_get,
    mock_summarize,
    mock_validate_question,
    _load_config,
    auth,
):
    """Test conversation request API endpoint."""
    # valid question
    mock_validate_question.return_value = True
    mock_response = (
        "Kubernetes is an open-source container-orchestration system..."  # summary
    )
    mock_summarize.return_value = {
        "response": mock_response,
        "referenced_documents": [],
        "history_truncated": False,
    }
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    response = ols.conversation_request(llm_request, auth)
    assert (
        response.response
        == "Kubernetes is an open-source container-orchestration system..."
    )
    assert suid.check_suid(
        response.conversation_id
    ), "Improper conversation ID returned"

    # invalid question
    mock_validate_question.return_value = False
    llm_request = LLMRequest(query="Generate a yaml")
    response = ols.conversation_request(llm_request, auth)
    assert response.response == constants.INVALID_QUERY_RESP
    assert suid.check_suid(
        response.conversation_id
    ), "Improper conversation ID returned"

    # validation failure
    mock_validate_question.side_effect = HTTPException
    with pytest.raises(HTTPException) as excinfo:
        llm_request = LLMRequest(query="Generate a yaml")
        response = ols.conversation_request(llm_request, auth)
        assert excinfo.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert len(response.conversation_id) == 0


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
@patch("ols.utils.config.conversation_cache.get")
def test_conversation_request_on_wrong_configuration(
    mock_conversation_cache_get,
    mock_validate_question,
    _load_config,
    auth,
):
    """Test conversation request API endpoint."""
    # mock invalid configuration
    message = "wrong model is configured"
    mock_validate_question.side_effect = Mock(
        side_effect=LLMConfigurationError(message)
    )
    llm_request = LLMRequest(query="Tell me about Kubernetes")

    # call must fail because we mocked invalid configuration state
    with pytest.raises(HTTPException, match="Unable to process this request"):
        ols.conversation_request(llm_request, auth)


@patch("ols.app.endpoints.ols.retrieve_previous_input", new=Mock(return_value=None))
@patch(
    "ols.app.endpoints.ols.validate_question",
    new=Mock(return_value=False),
)
def test_question_validation_in_conversation_start(_load_config, auth):
    """Test if question validation is skipped in follow-up conversation."""
    # note the `validate_question` is patched to always return as `SUBJECT_REJECTED`
    # this should resolve in rejection in summarization
    conversation_id = suid.get_suid()
    query = "some elaborate question"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)

    response = ols.conversation_request(llm_request, auth)

    assert response.response.startswith(constants.INVALID_QUERY_RESP)


@patch(
    "ols.app.endpoints.ols.retrieve_previous_input", new=Mock(return_value="something")
)
@patch(
    "ols.app.endpoints.ols.validate_question",
    new=Mock(return_value=constants.SUBJECT_REJECTED),
)
@patch("ols.src.query_helpers.docs_summarizer.DocsSummarizer.summarize")
def test_no_question_validation_in_follow_up_conversation(
    mock_summarize, _load_config, auth
):
    """Test if question validation is skipped in follow-up conversation."""
    # note the `validate_question` is patched to always return as `SUBJECT_REJECTED`
    # but as it is not the first question, it should proceed to summarization
    mock_summarize.return_value = {
        "response": "some elaborate answer",
        "referenced_documents": [],
        "history_truncated": False,
    }
    conversation_id = suid.get_suid()
    query = "some elaborate question"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)

    response = ols.conversation_request(llm_request, auth)

    assert response.response == "some elaborate answer"


@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "llm response"}))
@patch("ols.app.endpoints.ols.load_llm", new=mock_llm_loader(None))
def test_conversation_request_debug_api_no_conversation_id(_load_config):
    """Test conversation request debug API endpoint."""
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    response = ols.conversation_request_debug_api(llm_request)
    assert response.response == "llm response"


@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "llm response"}))
@patch("ols.app.endpoints.ols.load_llm", new=mock_llm_loader(None))
def test_conversation_request_debug_api_with_conversation_id(_load_config):
    """Test conversation request debug API endpoint."""
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(
        query="Tell me about Kubernetes", conversation_id=conversation_id
    )
    response = ols.conversation_request_debug_api(llm_request)
    assert response.response == "llm response"


@patch("ols.app.endpoints.ols.validate_question")
def test_conversation_request_invalid_subject(mock_validate, _load_config, auth):
    """Test how generate_response function checks validation results."""
    # prepare arguments for DocsSummarizer
    llm_request = LLMRequest(query="Tell me about Kubernetes")

    mock_validate.return_value = False
    response = ols.conversation_request(llm_request, auth)
    assert response.response == constants.INVALID_QUERY_RESP
    assert len(response.referenced_documents) == 0
    assert not response.truncated


@patch("ols.src.query_helpers.docs_summarizer.DocsSummarizer.summarize")
def test_generate_response_valid_subject(mock_summarize, _load_config):
    """Test how generate_response function checks validation results."""
    # mock the DocsSummarizer
    mock_response = (
        "Kubernetes is an open-source container-orchestration system..."  # summary
    )
    mock_summarize.return_value = {
        "response": mock_response,
        "referenced_documents": [],
        "history_truncated": False,
    }

    # prepare arguments for DocsSummarizer
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    previous_input = None

    # try to get response
    response, documents, truncated = ols.generate_response(
        conversation_id, llm_request, previous_input
    )

    # check the response
    assert "Kubernetes" in response
    assert len(documents) == 0
    assert not truncated


@patch("ols.src.query_helpers.docs_summarizer.DocsSummarizer.summarize")
def test_generate_response_on_summarizer_error(mock_summarize, _load_config):
    """Test how generate_response function checks validation results."""
    # mock the DocsSummarizer
    mock_response = Mock()
    mock_response.response = (
        "Kubernetes is an open-source container-orchestration system..."  # summary
    )
    mock_summarize.side_effect = Exception  # any exception might occur

    # prepare arguments for DocsSummarizer
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    previous_input = None

    # try to get response
    with pytest.raises(
        HTTPException, match="Error while obtaining answer for user question"
    ):
        ols.generate_response(conversation_id, llm_request, previous_input)


@patch(
    "ols.src.query_helpers.question_validator.QuestionValidator.validate_question",
    side_effect=Exception("mocked exception"),
)
def test_generate_response_unknown_validation_result(_load_config):
    """Test how generate_response function checks validation results."""
    # prepare arguments for DocsSummarizer
    conversation_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    previous_input = None

    # try to get response
    with pytest.raises(
        HTTPException, match="Error while obtaining answer for user question"
    ):
        ols.generate_response(conversation_id, llm_request, previous_input)


@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "llm response"}))
@patch("ols.app.endpoints.ols.load_llm", new=mock_llm_loader(None))
def test_generate_bare_response(_load_config):
    """Test the generation of bare response for debug endpoint."""
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=None)
    response = ols.generate_bare_response(None, llm_request)
    assert response == "llm response"


@pytest.fixture
def transcripts_location(tmpdir):
    """Fixture sets feedback location to tmpdir and return the path."""
    config.init_empty_config()
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
            "ols.app.endpoints.ols.validate_question",
            return_value=True,
        ),
        patch(
            "ols.app.endpoints.ols.generate_response",
            return_value=("something", [], False),
        ),
    ):
        llm_request = LLMRequest(query="Tell me about Kubernetes")
        response = ols.conversation_request(llm_request, auth)
        assert response
        assert response.response == "something"

        transcript_dir = Path(transcripts_location)
        assert list(transcript_dir.glob("*/*/*.json")) == []


def test_store_transcript(transcripts_location):
    """Test transcript is successfully stored."""
    user_id = suid.get_suid()
    conversation_id = suid.get_suid()
    query_is_valid = True
    query = "Tell me about Kubernetes"
    llm_request = LLMRequest(query=query, conversation_id=conversation_id)
    response = "Kubernetes is ..."
    ref_docs = ["d"]
    truncated = True

    ols.store_transcript(
        user_id,
        conversation_id,
        query_is_valid,
        llm_request,
        response,
        ref_docs,
        truncated,
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
            "timestamp": "fake-timestamp",
        },
        "redacted_query": query,
        "query_is_valid": query_is_valid,
        "llm_response": response,
        "referenced_documents": ref_docs,
        "truncated": truncated,
    }
