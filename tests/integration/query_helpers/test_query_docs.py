"""Integration tests using light weight FAISS index."""

import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents.base import Document

from ols.src.query_helpers.query_docs import QueryDocs, RetrieveDocsExceptionError


@pytest.fixture
def setup_faiss():
    """Documents stored as  up FAISS index."""
    list_of_documents = [
        Document(page_content="foo", metadata={"page": 1, "source": "adhoc"}),
        Document(page_content="bar", metadata={"page": 1, "source": "adhoc"}),
        Document(page_content="foo", metadata={"page": 2, "source": "adhoc"}),
        Document(page_content="barbar", metadata={"page": 2, "source": "adhoc"}),
        Document(page_content="foo", metadata={"page": 3, "source": "adhoc"}),
        Document(page_content="bar burr", metadata={"page": 3, "source": "adhoc"}),
        Document(page_content="foo", metadata={"page": 4, "source": "adhoc"}),
        Document(page_content="bar bruh", metadata={"page": 4, "source": "adhoc"}),
    ]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_documents(list_of_documents, embeddings)
    return db


def test_retrieve_top_k_similarity_search(setup_faiss):
    """Fetch top k similarity search."""
    docs = QueryDocs().get_relevant_docs(
        vectordb=setup_faiss, query="foo", search_kwargs={"k": 1}
    )

    assert len(docs) == 1
    assert docs[0].page_content == "foo"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "adhoc"


def test_retrieve_mmr(setup_faiss):
    """Fetch more documents for the MMR algorithm to consider. But only return the top 1."""
    docs = QueryDocs().get_relevant_docs(
        vectordb=setup_faiss,
        search_type="mmr",
        query="foo",
        search_kwargs={"k": 1, "fetch_k": 4},
    )

    assert len(docs) == 1
    assert docs[0].page_content == "foo"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "adhoc"


def test_similarity_score(setup_faiss):
    """Fetch only the docs that has scores above similarity score threshold."""
    docs = QueryDocs().get_relevant_docs(
        vectordb=setup_faiss,
        search_type="similarity_score_threshold",
        query="foo",
        search_kwargs={"score_threshold": 0.3},
    )

    assert len(docs) == 4
    assert docs[0].page_content == "foo"
    assert docs[0].metadata["page"] == 1
    assert docs[1].page_content == "foo"
    assert docs[1].metadata["page"] == 2
    assert docs[2].page_content == "foo"
    assert docs[2].metadata["page"] == 3
    assert docs[3].page_content == "foo"
    assert docs[3].metadata["page"] == 4
    assert docs[0].metadata["source"] == "adhoc"


def test_for_filtering(setup_faiss):
    """Fetch only the filtered docs."""
    docs = QueryDocs().get_relevant_docs(
        vectordb=setup_faiss,
        search_type="similarity_score_threshold",
        query="foo",
        search_kwargs={"score_threshold": 0.3, "filter": {"page": 1}},
    )

    assert len(docs) == 1
    assert docs[0].page_content == "foo"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "adhoc"


def test_invalid_search_type(setup_faiss):
    """Test for invalid search type."""
    with pytest.raises(
        RetrieveDocsExceptionError, match="search type is invalid: stuff"
    ):
        QueryDocs().get_relevant_docs(
            vectordb=setup_faiss,
            query="foo",
            search_kwargs={"k": 1},
            search_type="stuff",
        )
