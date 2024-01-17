"""A class for summarizing documentation context."""

import os

import llama_index
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.embeddings import TextEmbeddingsInference
from llama_index.prompts import PromptTemplate

from ols import constants
from ols.src.llms.llm_loader import LLMLoader
from ols.utils import config
from ols.utils.logger import Logger


class DocsSummarizer:
    """A class for summarizing documentation context."""

    def __init__(self):
        """Initialize the DocsSummarizer."""
        self.logger = Logger("docs_summarizer").logger

    def summarize(self, conversation: str, query: str, **kwargs) -> tuple[str, str]:
        """Summarize the given query based on the provided conversation context.

        Args:
            conversation: The unique identifier for the conversation.
            query: The query to be summarized.
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            A tuple containing the summary as a string and referenced documents as a string.
        """
        provider = config.ols_config.summarizer_provider
        model = config.ols_config.summarizer_model
        bare_llm = LLMLoader(provider, model).llm

        verbose = kwargs.get("verbose", "").lower() == "true"

        # Set up llama index to show prompting if verbose is True
        # TODO: remove this, we can't be setting global handlers, it will
        # affect other calls
        if verbose:
            llama_index.set_global_handler("simple")

        settings_string = f"conversation: {conversation}, query: {query}, provider: {provider}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        summarization_template = PromptTemplate(constants.SUMMARIZATION_TEMPLATE)

        self.logger.info(f"{conversation} Getting service context")
        self.logger.info(f"{conversation} using model: {model}")

        embed_model = "local:BAAI/bge-base-en"
        # TODO get this from global config instead of env
        # Not a priority because embedding model probably won't be configurable in the final product
        tei_embedding_url = os.getenv("TEI_SERVER_URL", None)
        if tei_embedding_url:
            self.logger.info(f"{conversation} using TEI embedding server")

            embed_model = TextEmbeddingsInference(
                model_name=constants.TEI_EMBEDDING_MODEL,
                base_url=tei_embedding_url,
            )

        service_context = ServiceContext.from_defaults(
            chunk_size=1024, llm=bare_llm, embed_model=embed_model, **kwargs
        )

        self.logger.info(
            f"{conversation} using embed model: {service_context.embed_model!s}"
        )

        # TODO get this from global config
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=constants.PRODUCT_DOCS_PERSIST_DIR
            )
            self.logger.info(f"{conversation} Setting up index")
            index = load_index_from_storage(
                storage_context=storage_context,
                index_id=constants.PRODUCT_INDEX,
                service_context=service_context,
                verbose=verbose,
            )
            self.logger.info(f"{conversation} Setting up query engine")
            query_engine = index.as_query_engine(
                text_qa_template=summarization_template,
                verbose=verbose,
                streaming=False,
                similarity_top_k=1,
            )

            self.logger.info(f"{conversation} Submitting summarization query")
            summary = query_engine.query(query)

            referenced_documents = "\n".join(
                [
                    source_node.node.metadata["file_name"]
                    for source_node in summary.source_nodes
                ]
            )
        except FileNotFoundError as err:
            self.logger.error(f"FileNotFoundError: {err.strerror}, file= {err.filename}")
            self.logger.info("Using llm to answer the query without RAG content")
           
            response = bare_llm.invoke(query)
            summary = f""" The following response was generated without access to RAG content:

                        {response.content}
                      """
            referenced_documents = ""
        
        self.logger.info(f"{conversation} Summary response: {summary!s}")
        self.logger.info(f"{conversation} Referenced documents: {referenced_documents}")
        return str(summary), referenced_documents
