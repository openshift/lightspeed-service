"""Unit tests for the API models."""

import pytest
from pydantic import ValidationError

from ols.app.models.models import (
    Attachment,
    CacheEntry,
    FeedbackRequest,
    FeedbackResponse,
    LivenessResponse,
    LLMRequest,
    LLMResponse,
    RagChunk,
    ReadinessResponse,
    ReferencedDocument,
    StatusResponse,
)
from ols.utils import suid


class TestLLM:
    """Unit tests for the LLMRequest/LLMResponse models."""

    @staticmethod
    def test_llm_request_required_inputs():
        """Test required inputs of the LLMRequest model."""
        query = "Tell me about Kubernetes"

        llm_request = LLMRequest(query=query)

        assert llm_request.query == query
        assert llm_request.conversation_id is None
        assert llm_request.provider is None
        assert llm_request.model is None

    @staticmethod
    def test_llm_request_optional_inputs():
        """Test optional inputs of the LLMRequest model."""
        query = "Tell me about Kubernetes"
        conversation_id = "id"
        provider = "openai"
        model = "gpt-4o-mini"
        llm_request = LLMRequest(
            query=query,
            conversation_id=conversation_id,
            provider=provider,
            model=model,
        )

        assert llm_request.query == query
        assert llm_request.conversation_id == conversation_id
        assert llm_request.provider == provider
        assert llm_request.model == model

    @staticmethod
    def test_llm_request_provider_and_model():
        """Test the LLMRequest model with provider and model."""
        # model set and provider not
        with pytest.raises(
            ValidationError,
            match="LLM provider must be specified when the model is specified.",
        ):
            LLMRequest(query="bla", provider=None, model="davinci")

        # provider set and model not
        with pytest.raises(
            ValidationError,
            match="LLM model must be specified when the provider is specified.",
        ):
            LLMRequest(query="bla", provider="openai", model=None)

    @staticmethod
    def test_llm_response():
        """Test the LLMResponse model."""
        conversation_id = "id"
        response = "response"
        referenced_documents = [
            ReferencedDocument(
                docs_url="https://foo.bar.com/index.html", title="Foo Bar"
            )
        ]

        llm_response = LLMResponse(
            conversation_id=conversation_id,
            response=response,
            referenced_documents=referenced_documents,
            truncated=False,
        )

        assert llm_response.conversation_id == conversation_id
        assert llm_response.response == response
        assert llm_response.referenced_documents == referenced_documents
        assert not llm_response.truncated


class TestStatusResponse:
    """Unit tests for the StatusResponse model."""

    @staticmethod
    def test_status_response():
        """Test the StatusResponse model."""
        functionality = "feedback"
        status = {"enabled": True}

        status_response = StatusResponse(functionality=functionality, status=status)

        assert status_response.functionality == functionality
        assert status_response.status == status


class TestFeedback:
    """Unit tests for the FeedbackRequest/FeedbackResponse models."""

    @staticmethod
    def test_feedback_request():
        """Test the FeedbackRequest model."""
        conversation_id = suid.get_suid()
        user_question = "user question"
        llm_response = "llm response"
        sentiment = 1
        user_feedback = "user feedback"

        feedback_request = FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment=sentiment,
            user_feedback=user_feedback,
        )

        assert feedback_request.conversation_id == conversation_id
        assert feedback_request.user_question == user_question
        assert feedback_request.llm_response == llm_response
        assert feedback_request.sentiment == sentiment
        assert feedback_request.user_feedback == user_feedback

    @staticmethod
    def test_feedback_request_optional_fields():
        """Test either sentiment or user_feedback needs to be set."""
        conversation_id = suid.get_suid()
        user_question = "user question"
        llm_response = "llm response"
        sentiment = 1
        user_feedback = "user feedback"

        # no sentiment or user_feedback raises validation error
        with pytest.raises(
            ValidationError, match="Either 'sentiment' or 'user_feedback' must be set"
        ):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
            )

        # just of those set doesn't raise - sentiment set
        FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment=sentiment,
        )

        # just of those set doesn't raise - user_feedback set
        FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            user_feedback=user_feedback,
        )

    @staticmethod
    def test_feedback_request_improper_conversation_id():
        """Test if conversation ID format is checked."""
        conversation_id = "this-is-bad"
        user_question = "user question"
        llm_response = "llm response"
        sentiment = 1
        user_feedback = "user feedback"

        # ValueError should be raised
        with pytest.raises(ValueError, match="Improper conversation ID this-is-bad"):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
                sentiment=sentiment,
                user_feedback=user_feedback,
            )

    @staticmethod
    def test_feedback_sentiment():
        """Test the sentiment field of the FeedbackRequest model."""
        conversation_id = suid.get_suid()
        user_question = "user question"
        llm_response = "llm response"

        feedback_request = FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment=1,
        )
        assert feedback_request.sentiment == 1

        feedback_request = FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment=-1,
        )
        assert feedback_request.sentiment == -1

        feedback_request = FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment=None,
            user_feedback="user feedback",
        )
        assert feedback_request.sentiment is None

        # can convert strings
        feedback_request = FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment="1",
        )
        assert feedback_request.sentiment == 1

        # check some invalid values
        with pytest.raises(
            ValidationError, match="Improper value 2, needs to be -1 or 1"
        ):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
                sentiment=2,
            )

        with pytest.raises(
            ValidationError, match="Improper value 0, needs to be -1 or 1"
        ):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
                sentiment=0,
            )

        with pytest.raises(
            ValidationError, match="Improper value 2, needs to be -1 or 1"
        ):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
                sentiment="2",
            )

        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
                sentiment="",
            )

        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
                sentiment="foo",
            )

    @staticmethod
    def test_feedback_response():
        """Test the FeedbackResponse model."""
        feedback_response = "feedback received"

        feedback_request = FeedbackResponse(response=feedback_response)

        assert feedback_request.response == feedback_response


class TestLiveness:
    """Unit test for the LivenessResponse model."""

    @staticmethod
    def test_liveness_response():
        """Test the LivenessResponse model."""
        liveness_response = True

        liveness_request = LivenessResponse(alive=liveness_response)

        assert liveness_request.alive == liveness_response


class TestReadiness:
    """Unit test for the ReadinessResponse model."""

    @staticmethod
    def test_readiness_response():
        """Test the ReadinessResponse model."""
        ready = True
        reason = "service is ready"

        readiness_request = ReadinessResponse(ready=ready, reason=reason)

        assert readiness_request.ready == ready
        assert readiness_request.reason == reason


class TestCacheEntry:
    """Unit test for the CacheEntry model."""

    @staticmethod
    def test_basic_interface():
        """Test the CacheEntry model."""
        cache_entry = CacheEntry(query="query")
        assert cache_entry.query == "query"
        assert cache_entry.response == ""

        cache_entry = CacheEntry(query="query", response=None)
        assert cache_entry.query == "query"
        assert cache_entry.response == ""

        cache_entry = CacheEntry(query="query", response="response")
        assert cache_entry.query == "query"
        assert cache_entry.response == "response"

    @staticmethod
    def test_to_dict():
        """Test the to_dict method of the CacheEntry model."""
        cache_entry = CacheEntry(query="query", response="response")
        assert cache_entry.to_dict() == {
            "human_query": "query",
            "ai_response": "response",
            "attachments": [],
        }

    @staticmethod
    def test_from_dict():
        """Test the from_dict method of the CacheEntry model."""
        attachment = Attachment(
            attachment_type="log",
            content_type="text/plain",
            content="this is attachment",
        )
        cache_entry = CacheEntry.from_dict(
            {
                "human_query": "query",
                "ai_response": "response",
                "attachments": [attachment.model_dump()],
            }
        )
        assert cache_entry.query == "query"
        assert cache_entry.response == "response"
        assert cache_entry.attachments == [attachment]

    @staticmethod
    def test_cache_entries_to_history():
        """Test the cache_entries_to_history method of the CacheEntry model."""
        cache_entries = [
            CacheEntry(query="query1", response="response1"),
            CacheEntry(query="query2", response="response2"),
        ]
        history = CacheEntry.cache_entries_to_history(cache_entries)
        assert history == [
            "human: query1",
            "ai: response1",
            "human: query2",
            "ai: response2",
        ]

    @staticmethod
    def test_cache_entries_to_history_no_whitespace():
        """Test content is stripped."""
        cache_entries = [
            CacheEntry(query="\ngood\nmorning\n", response="\ngood\nnight\n"),
        ]
        history = CacheEntry.cache_entries_to_history(cache_entries)
        assert history == [
            "human: good\nmorning",
            "ai: good\nnight",
        ]

    @staticmethod
    def test_cache_entries_to_history_no_content():
        """Test empty AI response is handled."""
        cache_entries = [
            CacheEntry(query="what?", response=""),
        ]
        history = CacheEntry.cache_entries_to_history(cache_entries)
        assert history == [
            "human: what?",
            "ai: ",
        ]

    @staticmethod
    def test_cache_entries_to_history_no_response():
        """Test no AI response is handled."""
        cache_entries = [
            CacheEntry(query="what?", response=None),
        ]
        history = CacheEntry.cache_entries_to_history(cache_entries)
        assert history == [
            "human: what?",
            "ai: ",
        ]


def test_ref_docs_from_rag_chunks():
    """Test the ReferencedDocument model method `from_rag_chunks`."""
    # urls are unsorted to ensure there is not a hidden sorting
    rag_chunk_1 = RagChunk("bla2", "url2", "title2")
    rag_chunk_2 = RagChunk("bla1", "url1", "title1")
    rag_chunk_3 = RagChunk("bla3", "url3", "title3")
    rag_chunk_4 = RagChunk("bla2", "url2", "title2")  # duplicated doc

    ref_docs = ReferencedDocument.from_rag_chunks(
        [rag_chunk_1, rag_chunk_2, rag_chunk_3, rag_chunk_4]
    )
    expected = [
        ReferencedDocument(docs_url="url2", title="title2"),
        ReferencedDocument(docs_url="url1", title="title1"),
        ReferencedDocument(docs_url="url3", title="title3"),
    ]

    assert ref_docs == expected
