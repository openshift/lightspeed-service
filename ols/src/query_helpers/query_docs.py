"""This module has the components to retrieve docs from vector store."""

import logging
from typing import Any

from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

logger = logging.getLogger(__name__)


class RetrieveDocsExceptionError(Exception):
    """Issue with retrieving the docs."""


class QueryDocs:
    """Generic interface for retrieving docs from any vector store."""

    def get_relevant_docs(
        self,
        query: str,
        vectordb: VectorStore,
        search_type: str = "similarity",
        **search_kwargs: Any,
    ) -> list[Document]:
        """Return list of documents from the vectorstore.

        Args:
            query: string to find relevant documents for
            vectordb: Vector db from which you need to retrieve the docs.
            search_type: Defines the type of search that
                the retriever should perform.
                Can be "similarity" (default), "mmr", or
                "similarity_score_threshold".
            search_kwargs: Keyword arguments to pass to the
                search function. Can include things like:
                    k: Amount of documents to return (Default: 4)
                    score_threshold: Minimum relevance threshold
                        for similarity_score_threshold
                    fetch_k: Amount of documents to pass to MMR algorithm (
                    Default: 20)
                    lambda_mult: Diversity of results returned by MMR;
                        1 for minimum diversity and 0 for maximum. (Default:
                        0.5)
                    filter: Filter by document metadata

        Returns:
            A list of docs.

        Example:
           ```python
            # Retrieve more documents with higher diversity
            # Useful if your dataset has many similar documents
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 6, 'lambda_mult': 0.25}
            )

            # Fetch more documents for the MMR algorithm to consider
            # But only return the top 5
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 50}
            )

            # Only retrieve documents that have a relevance score
            # Above a certain threshold
            docsearch.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.8}
            )

            # Only get the single most similar document from the dataset
            docsearch.as_retriever(search_kwargs={'k': 1})

            # Use a filter to only retrieve documents from a specific paper
            docsearch.as_retriever(
                search_kwargs={'filter': {'paper_title':'GPT-4 Technical
                Report'}}
            )
            ```
        """
        logger.info(
            """Retrieving docs for
                query: %s,
                vectordb: %s,
                search_type: %s,
                search_kwargs: %s
        """,
            query,
            vectordb,
            search_type,
            search_kwargs,
        )

        if search_type not in {"mmr", "similarity", "similarity_score_threshold"}:
            logger.exception("incorrect search type '%s'", search_type)
            raise RetrieveDocsExceptionError(f"search type is invalid: {search_type}")

        db_retriever: VectorStoreRetriever = vectordb.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs["search_kwargs"]
        )

        try:
            docs = db_retriever.get_relevant_documents(query=query)
        except Exception as e:
            logger.error("exception raised while getting the docs for query: %s", query)
            raise RetrieveDocsExceptionError(
                "error in getting the docs from vectorstore"
            ) from e

        return docs
