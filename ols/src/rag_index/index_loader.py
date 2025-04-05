# type: ignore
"""Module for loading index."""

import logging
from typing import Any, Optional

from ols.app.models.config import ReferenceContent

logger = logging.getLogger(__name__)


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
    global resolve_llm
    global FaissVectorStore
    global VectorStoreIndex
    global SummaryIndex
    global IndexNode
    from llama_index.core import (
        Settings,
        StorageContext,
        SummaryIndex,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.core.embeddings.utils import EmbedType
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.llms.utils import resolve_llm
    from llama_index.core.schema import IndexNode
    from llama_index.vector_stores.faiss import FaissVectorStore


class IndexLoader:
    """Load index from local file storage."""

    def __init__(self, index_config: Optional[ReferenceContent]) -> None:
        """Initialize loader."""
        load_llama_index_deps()
        self._index = None

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

        index_nodes = []
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
                index_nodes.append(
                    IndexNode(
                        index_id=index_config.product_docs_index_id,
                        obj=index.as_retriever(similarity_top_k=5),
                        text=index_config.product_docs_index_id,
                    )
                )
                logger.info("Vector index #%d is loaded.", i)
            except Exception as err:
                logger.exception("Error loading vector index #%d:\n%s", i, err)

        logger.info("All indexes are loaded, merging them into a single graph.")
        if len(index_nodes) == 0:
            logger.warning("No index is eligible for merging. Index is not ready.")
            return
        self._index = SummaryIndex(objects=index_nodes)
        logger.info("Index is ready.")

    @property
    def vector_index(self) -> Optional[ReferenceContent]:
        """Get index."""
        if self._index is None:
            logger.warning(
                "Proceeding without RAG content. "
                "Either there is an error or required parameters are not set."
            )
        return self._index
