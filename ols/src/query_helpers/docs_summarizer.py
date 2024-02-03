"""A class for summarizing documentation context."""

import logging
from typing import Any, Optional

import llama_index
from langchain.chains import LLMChain
from langchain.schema import ChatMessage
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.postprocessor import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.prompts import PromptTemplate as LlamaTemplate
from llama_index.response.schema import Response
from llama_index.schema import NodeWithScore, QueryBundle

from ols import constants
from ols.src.llms.llm_loader import LLMLoader
from ols.src.query_helpers import QueryHelper
from ols.src.query_helpers.chat_history import (
    get_langchain_chat_history,
    get_llama_index_chat_history,
)
from ols.src.query_helpers.constants import summary_prompt_for_langchain
from ols.utils import config

logger = logging.getLogger(__name__)


class _NoLLMMetadataPostprocessor(BaseNodePostprocessor):
    """A LlamaIndex PostProcessor that prevents node metadata from reaching the LLM."""

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> list[NodeWithScore]:
        for n in nodes:
            n.node.excluded_llm_metadata_keys = list(n.node.metadata.keys())
        return nodes


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    @staticmethod
    def _file_path_to_doc_url(file_path: str) -> str:
        """Convert file_path metadata to the corresponding URL on the OCP docs website.

        Embedding node metadata 'file_path' in the form
        file_path: /workspace/source/ocp-product-docs-plaintext/hardware_enablement/
                    psap-node-feature-discovery-operator.txt
        is mapped into a doc URL such as
        https://docs.openshift.com/container-platform/4.14/hardware_enablement/psap-node-feature-discovery-operator.html.
        """
        return (
            constants.OCP_DOCS_ROOT_URL
            + constants.OCP_DOCS_VERSION
            + file_path.removeprefix(constants.EMBEDDINGS_ROOT_DIR)
        ).removesuffix("txt") + "html"

    def summarize(
        self,
        conversation_id: str,
        query: str,
        history: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[Response | str, str]:
        """Summarize the given query based on the provided conversation context.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            A tuple containing  Response or String and referenced documents as a string.
        """
        rag_sucess = True
        if (
            config.ols_config.reference_content is not None
            and config.ols_config.reference_content.product_docs_index_path is not None
        ):
            try:
                summary, referenced_documents = self.get_docs_with_rag(
                    conversation_id, query, history
                )
                logger.info(f"{conversation_id} Summary response: {summary.response}")
                logger.info(
                    f"{conversation_id} Referenced documents: {referenced_documents}"
                )
            except Exception as err:
                logger.info(f"Error getting answer from reference index: {err}")
                rag_sucess = False

            if rag_sucess:
                return summary, referenced_documents
        summary_no_rag, referenced_documents = self.get_docs_no_rag(
            conversation_id, query, history
        )
        logger.info(f"{conversation_id} Summary response: {summary_no_rag}")
        logger.info(f"{conversation_id} Referenced documents: {referenced_documents}")
        return summary_no_rag, referenced_documents

    def get_docs_no_rag(
        self,
        conversation_id: str,
        query: str,
        history: Optional[str] = None,
    ) -> tuple[str, str]:
        """Summarize the given query, with no RAG, based on the provided conversation context.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            A tuple containing the summary as string and referenced documents as a string.
        """
        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {query}, "
            f"history: {history}, "
        )
        logger.debug(f"{conversation_id} call settings: {settings_string}")
        langchian_chat_engine, chat_history_no_reference = (
            self.get_chat_engine_langchain(conversation_id, query, history)
        )
        response = langchian_chat_engine.invoke(
            {"query": query, "chat_history": chat_history_no_reference}
        )
        response_text = response.get("text")
        response_val = (
            "The following response was generated without access to reference content:"
            "\n\n"
            # NOTE: The LLM returns AIMessage, but typing sees it as a plain str
            f"{response_text}"  # type: ignore
        )
        return response_val, ""

    def get_docs_with_rag(
        self,
        conversation_id: str,
        query: str,
        history: Optional[str] = None,
    ) -> tuple[Response, str]:
        """Summarize the given query using RAG based on the provided conversation context.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            A tuple containing the summary as Response Object and referenced documents as a string.
        """
        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {query}, "
            f"history: {history}, "
        )
        logger.debug(f"{conversation_id} call settings: {settings_string}")
        try:
            llama_chat_engine = self.get_chat_engine_llama(
                conversation_id, query, history
            )
            response = llama_chat_engine.chat(query)
            referenced_documents = "\n".join(
                [
                    source_node.node.metadata["file_name"]
                    for source_node in response.source_nodes
                ]
            )

        except Exception as err:
            logger.info(f"Error getting answer from reference index: {err}")
            raise
        return response, referenced_documents

    def get_chat_engine_langchain(
        self, conversation_id: str, query: str, history: Optional[str] = None, **kwargs
    ) -> tuple[LLMChain, list[ChatMessage]]:
        """Get LLMChain to summerize when there is no RAG.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            LLMChain object to send user query,
            List of ChatMessages containing history messages.
        """
        logger.debug("Creating llmchain to answer the query without reference content")
        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
        )
        logger.debug(f"{conversation_id} call settings: {settings_string}")

        bare_llm = LLMLoader(self.provider, self.model).llm
        logger.debug(f"History:  {history}")
        chat_history_no_reference = get_langchain_chat_history(history)
        logger.debug(f"Chat history for chat_llm_chain:  {chat_history_no_reference}")
        chat_llm_chain = LLMChain(
            llm=bare_llm,
            prompt=summary_prompt_for_langchain,
        )
        return chat_llm_chain, chat_history_no_reference

    def get_chat_engine_llama(
        self, conversation_id: str, query: str, history: Optional[str] = None, **kwargs
    ) -> BaseChatEngine:
        """Get Chat Engine to query RAG Index.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            BaseChatEngine object to send user.
        """
        logger.debug("Creating chat engine to answer the query with reference content")
        bare_llm = LLMLoader(self.provider, self.model).llm

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

        # TODO: use history there
        logger.debug(f"History: {history}")
        chat_history = get_llama_index_chat_history(history)
        logger.debug(f"Chat history for chat_engine: {chat_history}")
        summarization_template = LlamaTemplate(constants.SUMMARIZATION_TEMPLATE)

        logger.info(f"{conversation_id} Getting service context")

        embed_model = "local:BAAI/bge-base-en"

        service_context = ServiceContext.from_defaults(
            chunk_size=1024, llm=bare_llm, embed_model=embed_model, **kwargs
        )
        logger.info(
            f"{conversation_id} using embed model: {service_context.embed_model!s}"
        )
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=config.ols_config.reference_content.product_docs_index_path
            )
            logger.info(f"{conversation_id} Setting up index")
            index = load_index_from_storage(
                storage_context=storage_context,
                service_context=service_context,
                index_id=config.ols_config.reference_content.product_docs_index_id,
                verbose=verbose,
            )
            logger.info(f"{conversation_id} Setting up query engine")

            logger.info(f"{conversation_id} Submitting summarization query")

            chat_engine = index.as_chat_engine(
                service_context=service_context,
                chat_history=chat_history,
                chat_mode="context",
                system_prompt=summarization_template,
            )
            return chat_engine
        except Exception as err:
            logger.error(f"Error loading vector index: {err}")
            raise
