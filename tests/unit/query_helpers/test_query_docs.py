"""Unit test for query doc retriever."""

from unittest import mock

import pytest

from ols.src.query_helpers.query_docs import QueryDocs, RetrieveDocsExceptionError
from tests.mock_classes.mock_retrievers import (
    MockVectorStore,
    mock_null_value_retriever,
    mock_retriever,
)


def test_query_docs_success():
    """For retrieving the docs."""
    with mock.patch(
        target="tests.mock_classes.mock_retrievers.MockVectorStore.as_retriever",
        return_value=mock_retriever(),
    ):
        docs = QueryDocs().get_relevant_docs(
            vectordb=MockVectorStore(),
            query="foo",
            search_kwargs={"k": 1},
        )

        assert len(docs) == 1
        assert docs[0].page_content == "foo"
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["source"] == "adhoc"


def test_query_docs_failure():
    """Exception is raised when there is an issue with fetching the docs."""
    with (
        mock.patch(
            target="tests.mock_classes.mock_retrievers.MockVectorStore.as_retriever",
            return_value=mock_null_value_retriever(),
        ),
        pytest.raises(
            RetrieveDocsExceptionError,
            match="error in getting the docs from vectorstore",
        ),
    ):
        QueryDocs().get_relevant_docs(
            vectordb=MockVectorStore(),
            query="foo",
            search_kwargs={"k": 1},
        )


def test_query_docs_correct_search_type():
    """For retrieving the docs with correct search type."""
    with mock.patch(
        target="tests.mock_classes.mock_retrievers.MockVectorStore.as_retriever",
        return_value=mock_retriever(),
    ):
        docs = QueryDocs().get_relevant_docs(
            vectordb=MockVectorStore(),
            query="foo",
            search_type="similarity",
            search_kwargs={"k": 1},
        )

        assert len(docs) == 1
        assert docs[0].page_content == "foo"
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["source"] == "adhoc"


def test_query_docs_incorrect_search_type():
    """Exception is raised when incorrect search type is provided."""
    with pytest.raises(
        RetrieveDocsExceptionError,
        match="search type is invalid: unknown",
    ):
        QueryDocs().get_relevant_docs(
            vectordb=MockVectorStore(),
            query="foo",
            search_type="unknown",
            search_kwargs={"k": 1},
        )
