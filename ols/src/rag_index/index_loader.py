"""Module for loading index."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.indices.base import BaseIndex
from llama_index.vector_stores.faiss import FaissVectorStore

from ols.app.models.config import ReferenceContent

# This is to avoid importing HuggingFaceEmbeddings in all cases, because in
# runtime it is used only under some conditions. OTOH we need to make Python
# interpreter happy in all circumstances, hence the definiton of Any symbol.
if TYPE_CHECKING:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # TCH004
else:
    HuggingFaceEmbeddings = Any

logger = logging.getLogger(__name__)


class IndexLoader:
    """Load index from local file storage."""

    def __init__(self, index_config: Optional[ReferenceContent]) -> None:
        """Initialize loader."""
        self._index: Optional[BaseIndex] = None

        self._index_config = index_config
        logger.debug(f"Config used for index load: {self._index_config}")

        if self._index_config is None:
            logger.warning("Config for reference content is not set.")
        else:
            self._index_path = self._index_config.product_docs_index_path
            self._index_id = self._index_config.product_docs_index_id

            self._embed_model_path = self._index_config.embeddings_model_path
            self._embed_model = self._get_embed_model()
            self._load_index()

    def _get_embed_model(self) -> Optional[str | HuggingFaceEmbeddings]:
        """Get embed model according to configuration."""
        if self._embed_model_path is not None:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            logger.debug(
                f"Loading embedding model info from path {self._embed_model_path}"
            )
            return HuggingFaceEmbeddings(model_name=self._embed_model_path)

        logger.warning("Embedding model path is not set.")
        logger.warning("Embedding model is set to default")
        return "local:sentence-transformers/all-mpnet-base-v2"

    def _set_context(self) -> None:
        """Set storage/service context required for index load."""
        logger.debug(f"Using {self._embed_model!s} as embedding model for index.")
        logger.info("Setting up service context for index load...")
        self._service_context = ServiceContext.from_defaults(
            embed_model=self._embed_model, llm=None
        )
        logger.info("Setting up storage context for index load...")
        self._storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_dir(self._index_path),
            persist_dir=self._index_path,
        )

    def _load_index(self) -> None:
        """Load vector index."""
        if self._index_path is None:
            logger.warning("Index path is not set.")
        else:
            try:
                self._set_context()
                logger.info("Loading vector index...")
                self._index = load_index_from_storage(
                    service_context=self._service_context,
                    storage_context=self._storage_context,
                    index_id=self._index_id,
                )
                logger.info("Vector index is loaded.")
            except Exception as err:
                logger.exception(f"Error loading vector index:\n{err}")

    @property
    def vector_index(self) -> Optional[BaseIndex]:
        """Get index."""
        if self._index is None:
            logger.warning(
                "Proceeding without RAG content. "
                "Either there is an error or required parameters are not set."
            )
        return self._index
