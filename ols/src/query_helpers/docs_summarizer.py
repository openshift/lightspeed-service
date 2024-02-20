"""A class for summarizing documentation context."""

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import llama_index
from langchain.chains import LLMChain
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.indices.vector_store.base import VectorStoreIndex

from ols import constants
from ols.src.prompts.prompts import CHAT_PROMPT
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils import config
from ols.utils.token_handler import (
    # TODO: Use constants from config
    CONTEXT_WINDOW_LIMIT,
    RESPONSE_WINDOW_LIMIT,
    TokenHandler,
)

# this is to avoid importing HuggingFaceBgeEmbeddings in all cases, because in
# runtime it is used only under some conditions. OTOH we need to make Python
# interpreter happy in all circumstances, hence the definiton of Any symbol.
if TYPE_CHECKING:
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # TCH004
else:
    HuggingFaceBgeEmbeddings = Any

logger = logging.getLogger(__name__)


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    @staticmethod
    def _file_path_to_doc_url(file_path: str) -> str:
        """Convert file_path metadata to the corresponding URL on the OCP docs website.

        Embedding node metadata 'file_path' in the form
        file_path: /workspace/source/ocp-product-docs-plaintext/hardware_enablement/
                    psap-node-feature-discovery-operator.txt
        is mapped into a doc URL such as
        https://docs.openshift.com/container-platform/4.14/hardware_enablement/
        psap-node-feature-discovery-operator.html.
        """
        return (
            constants.OCP_DOCS_ROOT_URL
            + constants.OCP_DOCS_VERSION
            + file_path.removeprefix(constants.EMBEDDINGS_ROOT_DIR)
        ).removesuffix("txt") + "html"

    def _format_rag_data(self, rag_data: list[dict]) -> tuple[str, list[str]]:
        """Format rag text & metadata.

        Join multiple rag text with new line.
        Create list of metadata from rag data dictionary.
        """
        rag_text = []
        file_path = []
        for data in rag_data:
            rag_text.append(data["text"])
            file_path.append(self._file_path_to_doc_url(data["file_path"]))

        return "\n\n".join(rag_text), file_path

    @staticmethod
    def _get_rag_data(rag_index: VectorStoreIndex, query: str) -> list[dict]:
        """Get rag index data.

        Get relevant rag content based on query.
        Calculate available tokens.
        Returns rag content text & metadata as dictionary.
        """
        # TODO: Implement a different module for retrieval
        # with different options, currently it is top_k.
        # We do have a module with langchain, but not compatible
        # with llamaindex object.
        retriever = rag_index.as_retriever(similarity_top_k=1)
        retrieved_nodes = retriever.retrieve(query)

        # Truncate rag context, if required.
        token_handler_obj = TokenHandler()
        interim_prompt = CHAT_PROMPT.format(context="", query=query)
        prompt_token_count = len(token_handler_obj.text_to_tokens(interim_prompt))
        available_tokens = (
            CONTEXT_WINDOW_LIMIT - RESPONSE_WINDOW_LIMIT - prompt_token_count
        )

        return token_handler_obj.truncate_rag_context(retrieved_nodes, available_tokens)

    def summarize(
        self,
        conversation_id: str,
        query: str,
        history: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Summarize the given query based on the provided conversation context.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            A dictionary containing below property
            - the summary as a string
            - referenced documents as a list of strings
            - flag indicating that conversation history has been truncated
              to fit within context window.
        """
        verbose = kwargs.get("verbose", "").lower() == "true"

        # Set up llama index to show prompting if verbose is True
        # TODO: remove this, we can't be setting global handlers, it will
        # affect other calls
        if verbose:
            llama_index.set_global_handler("simple")

        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {verbose}"
        )
        logger.info(f"{conversation_id} call settings: {settings_string}")

        logger.info(f"{conversation_id} Getting service context")

        embed_model = DocsSummarizer.get_embed_model()

        bare_llm = self.llm_loader(
            self.provider, self.model, llm_params=self.llm_params
        ).llm  # type: ignore
        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=embed_model, **kwargs
        )
        logger.info(
            f"{conversation_id} using embed model: {service_context.embed_model!s}"
        )

        rag_context_data: list[dict] = []
        referenced_documents: list[str] = []

        truncated = (
            False  # TODO tisnik: need to be implemented based on provided inputs
        )

        # TODO get this from global config
        if (
            config.ols_config.reference_content is not None
            and config.ols_config.reference_content.product_docs_index_path is not None
        ):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=config.ols_config.reference_content.product_docs_index_path
                )
                logger.info(f"{conversation_id} Setting up index")
                index = load_index_from_storage(
                    storage_context=storage_context,
                    index_id=config.ols_config.reference_content.product_docs_index_id,
                    service_context=service_context,
                    verbose=verbose,
                )

                logger.info(f"{conversation_id}: Getting index data.")
                rag_context_data = self._get_rag_data(index, query)

            except Exception as err:
                logger.error(f"Error loading vector index: {err}")
        else:
            logger.info("Reference content is not configured")

        rag_context, referenced_documents = self._format_rag_data(rag_context_data)

        chat_engine = LLMChain(
            llm=bare_llm,
            prompt=CHAT_PROMPT,
            verbose=verbose,
        )
        summary = chat_engine.invoke({"context": rag_context, "query": query})
        response = summary["text"]

        if len(rag_context) == 0:
            logger.info("Using llm to answer the query without reference content")
            response = (
                "The following response was generated without access to reference content:"
                "\n\n"
                f"{response}"
            )

        logger.info(f"{conversation_id} Summary response: {response}")
        logger.info(f"{conversation_id} Referenced documents: {referenced_documents}")

        return {
            "response": response,
            "referenced_documents": referenced_documents,
            "history_truncated": truncated,
        }

    @staticmethod
    def get_embed_model() -> Optional[str | HuggingFaceBgeEmbeddings]:
        """Get embed model according to configuration."""
        if (
            config.ols_config.reference_content is not None
            and config.ols_config.reference_content.embeddings_model_path is not None
        ):
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            # TODO syedriko consolidate these env vars into a central location as per OLS-345.
            os.environ["TRANSFORMERS_CACHE"] = (
                config.ols_config.reference_content.embeddings_model_path
            )
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            return HuggingFaceBgeEmbeddings(
                model_name=config.ols_config.reference_content.embeddings_model_path
            )
        else:
            return "local:BAAI/bge-base-en"
