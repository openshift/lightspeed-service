# type: ignore
"""Module for loading index."""

import logging
from typing import Any, Optional

from ols.app.models.config import ReferenceContent
from ols.constants import RAG_CONTENT_LIMIT

logger = logging.getLogger(__name__)

# delay import of llama_index dependencies
BaseIndex = Any
BaseRetriever = Any


# NOTE: Loading/importing something from llama_index bumps memory
# consumption up to ~400MiB. To avoid loading llama_index in all cases,
# we load it only when it is required.
# As these dependencies are lazily loaded, we can't use them in type hints.
# So this module is excluded from mypy checks as a whole.
def load_llama_index_deps() -> None:
    """Load llama_index dependencies."""
    # pylint: disable=global-statement disable=C0415
    global Settings
    global StorageContext
    global load_index_from_storage
    global EmbedType
    global BaseIndex
    global BaseRetriever
    global resolve_llm
    global FaissVectorStore
    global QueryFusionRetriever
    from llama_index.core import (
        Settings,
        StorageContext,
        load_index_from_storage,
    )
    from llama_index.core.embeddings.utils import EmbedType
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.llms.utils import resolve_llm
    from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
    from llama_index.vector_stores.faiss import FaissVectorStore


def calculate_retrievers_weights(number_of_retrievers: int) -> list[float]:
    """Calculate weights for retrievers.

    Weights always prioritizes the first retriever.
    """
    if number_of_retrievers == 0:
        logger.warning("Number of retrievers is 0. Returning empty weights list.")
        return []
    if number_of_retrievers == 1:
        return [1.0]

    # assign a higher weight to the first retriever
    first_weight = 0.6  # adjust this value as needed
    remaining_weight = 1.0 - first_weight
    other_weight = remaining_weight / (number_of_retrievers - 1)

    # generate the weights list
    retriever_weights = [first_weight] + [other_weight] * (number_of_retrievers - 1)

    return retriever_weights


class IndexLoader:
    """Load index from local file storage."""

    def __init__(self, index_config: Optional[ReferenceContent]) -> None:
        """Initialize loader."""
        load_llama_index_deps()
        self._indexes = None
        self._retriever = None

        self._index_config = index_config
        logger.debug("Config used for index load: %s", str(self._index_config))

        if self._index_config is None:
            logger.warning("Config for reference content is not set.")
        elif self._index_config.indexes is None or len(self._index_config.indexes) == 0:
            logger.warning("Indexes are not set in the config for reference content.")
        else:

            self._embed_model_path = self._index_config.embeddings_model_path
            self._embed_model = self._get_embed_model()
            self._load_index()

    def _get_embed_model(self) -> Any:
        """Get embed model according to configuration."""
        if self._embed_model_path is not None:
            # pylint: disable=C0415
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.debug(
                "Loading embedding model info from path %s", self._embed_model_path
            )
            return HuggingFaceEmbedding(model_name=self._embed_model_path)

        logger.warning("Embedding model path is not set.")
        logger.warning("Embedding model is set to default")
        return "local:sentence-transformers/all-mpnet-base-v2"

    def _load_index(self) -> None:
        """Load vector index."""
        logger.debug("Using %s as embedding model for index", str(self._embed_model))
        logger.info("Setting up settings for index load...")
        Settings.embed_model = self._embed_model
        Settings.llm = resolve_llm(None)

        indexes = []
        for i, index_config in enumerate(self._index_config.indexes):
            if index_config.product_docs_index_path is None:
                logger.warning("Index path is not set for index #%d, skip loading.", i)
                continue
            try:
                # pylint: disable=W0201
                logger.info("Setting up storage context for index #%d...", i)
                storage_context = StorageContext.from_defaults(
                    vector_store=FaissVectorStore.from_persist_dir(
                        index_config.product_docs_index_path
                    ),
                    persist_dir=index_config.product_docs_index_path,
                )
                logger.info("Loading vector index #%d...", i)
                index = load_index_from_storage(
                    storage_context=storage_context,
                    index_id=index_config.product_docs_index_id,
                )
                indexes.append(index)
                logger.info("Vector index #%d is loaded.", i)
            except Exception as err:
                logger.exception(
                    "Error loading vector index #%d:\n%s, skipped.", i, err
                )
        if len(indexes) == 0:
            logger.warning("No indexes are loaded.")
            return
        if len(indexes) < len(self._index_config.indexes):
            logger.warning(
                "Some indexes are not loaded. "
                "Check the logs for details about the errors."
            )
        else:
            logger.info("All indexes are loaded.")
        self._indexes = indexes

    @property
    def vector_indexes(self) -> Optional[list[BaseIndex]]:
        """Get index."""
        if self._indexes is None:
            logger.warning(
                "Proceeding without RAG content. "
                "Either there is an error or required parameters are not set."
            )
        return self._indexes

    def get_retriever(
        self, similarity_top_k=RAG_CONTENT_LIMIT
    ) -> Optional[BaseRetriever]:
        """Get QueryFusionRetriever from indexes."""
        if self._indexes is None:
            logger.error("Cannot get retriever. Indexes are not loaded or empty.")
            return None
        if (
            self._retriever is not None
            and self._retriever.similarity_top_k == similarity_top_k
        ):
            return self._retriever

        retrievers = [
            index.as_retriever(similarity_top_k=similarity_top_k)
            for index in self._indexes
        ]

        if len(retrievers) == 1:
            retriever = QueryFusionRetriever(
                retrievers=retrievers,
                similarity_top_k=similarity_top_k,
                num_queries=1,  # set this to 1 to disable query generation
                use_async=False,
                verbose=False,
            )
        else:
            # set weights and mode in case of multiple retrievers/indices
            retriever = QueryFusionRetriever(
                retrievers=retrievers,
                retriever_weights=calculate_retrievers_weights(len(retrievers)),
                mode="relative_score",  # relative score fusion mode to apply weights
                similarity_top_k=similarity_top_k,
                num_queries=1,  # set this to 1 to disable query generation
                use_async=False,
                verbose=False,
            )
        self._retriever = retriever
        return self._retriever
