"""A class for summarizing documentation context."""

import logging
from typing import Any, Optional

import llama_index
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.postprocessor import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.response.schema import Response
from llama_index.schema import NodeWithScore, QueryBundle

from ols import constants
from ols.src.llms.llm_loader import LLMLoader
from ols.src.query_helpers import QueryHelper
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
    ) -> tuple[Response, str]:
        """Summarize the given query based on the provided conversation context.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            A tuple containing the summary as a string and referenced documents as a string.
        """
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
        summarization_template = PromptTemplate(constants.SUMMARIZATION_TEMPLATE)

        logger.info(f"{conversation_id} Getting service context")

        embed_model = "local:BAAI/bge-base-en"

        service_context = ServiceContext.from_defaults(
            chunk_size=1024, llm=bare_llm, embed_model=embed_model, **kwargs
        )

        logger.info(
            f"{conversation_id} using embed model: {service_context.embed_model!s}"
        )

        # TODO get this from global config
        if (
            config.ols_config.reference_content is not None
            and config.ols_config.reference_content.product_docs_index_path is not None
        ):
            use_llm_without_reference_content = False
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
                logger.info(f"{conversation_id} Setting up query engine")
                query_engine = index.as_query_engine(
                    text_qa_template=summarization_template,
                    verbose=verbose,
                    streaming=False,
                    similarity_top_k=1,
                    node_postprocessors=[_NoLLMMetadataPostprocessor()],
                )

                logger.info(f"{conversation_id} Submitting summarization query")
                summary = query_engine.query(query)

                referenced_documents = "\n".join(
                    {
                        self._file_path_to_doc_url(
                            source_node.node.metadata["file_path"]
                        )
                        for source_node in summary.source_nodes
                    }
                )
            except Exception as err:
                logger.error(f"Error loading vector index: {err}")
                use_llm_without_reference_content = True
        else:
            logger.info("Reference content is not configured")
            use_llm_without_reference_content = True

        if use_llm_without_reference_content:
            logger.info("Using llm to answer the query without reference content")
            response = bare_llm.invoke(query)
            # TODO: we use "ChatOpenAI" for openai providers which is derived from "BaseChatModel",
            # the invocation of which returns a "BaseMessage" object, and the actual llm response
            # text is contained within the "content" attribute of BaseMessage.
            # But for watsonx/bam providers we use implementations which are derived from the
            # "LLM" class, the invocation of which returns a raw string containing the llm response
            # text.
            # We need to find a better way to abstract that difference so that every caller
            # doesn't need to deal with this, but for now this hack makes OLS work with
            # both kinds of providers.
            if hasattr(response, "content"):
                response=response.content
            summary = Response(
                "The following response was generated without access to reference content:"
                "\n\n"
                # NOTE: The LLM returns AIMessage, but typing sees it as a plain str
                f"{response}"  # type: ignore
            )
            referenced_documents = ""

        logger.info(f"{conversation_id} Summary response: {summary!s}")
        logger.info(f"{conversation_id} Referenced documents: {referenced_documents}")

        return summary, referenced_documents
