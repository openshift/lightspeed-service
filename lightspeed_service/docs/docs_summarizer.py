import os

import llama_index
from dotenv import load_dotenv
from llama_index import load_index_from_storage
from llama_index import StorageContext
from llama_index.prompts import PromptTemplate

from lightspeed_service import constants
from lightspeed_service.utils.logger import Logger
from lightspeed_service.utils.model_context import get_watsonx_context

load_dotenv()


class DocsSummarizer:
    """
    A class for summarizing documentation context.
    """

    def __init__(self):
        """
        Initialize the DocsSummarizer.
        """
        self.logger = Logger("docs_summarizer").logger

    def summarize(self, conversation, query, **kwargs):
        """
        Summarize the given query based on the provided conversation
        context.

        Args:
        - conversation: The unique identifier for the conversation.
        - query: The query to be summarized.
        - kwargs: Additional keyword arguments for customization (model,
            verbose, etc.).

        Returns:
        - Tuple[str, str]: A tuple containing the summary as a string
            and referenced documents as a string.
        """
        model = kwargs.get(
            "model",
            os.getenv("DOC_SUMMARIZER_MODEL", "ibm/granite-13b-chat-v1"),
        )
        verbose = kwargs.get("verbose", "").lower() == "true"

        # Set up llama index to show prompting if verbose is True
        if verbose:
            llama_index.set_global_handler("simple")

        settings_string = (
            f"conversation: {conversation}, query: {query}, model: {model}, "
            f"verbose: {verbose}"
        )
        self.logger.info(f"{conversation} call settings: {settings_string}")

        summarization_template = PromptTemplate(
            constants.SUMMARIZATION_TEMPLATE
        )

        self.logger.info(f"{conversation} Getting service context")
        self.logger.info(f"{conversation} using model: {model}")

        tei_embedding_url = os.getenv("TEI_SERVER_URL", None)
        if tei_embedding_url:
            self.logger.info(f"{conversation} using TEI embedding server")
            service_context = get_watsonx_context(
                model=model,
                tei_embedding_model=constants.TEI_EMBEDDING_MODEL,
                url=tei_embedding_url,
            )
        else:
            service_context = get_watsonx_context(model=model)

        self.logger.info(
            f"{conversation} using embed model: "
            f"{str(service_context.embed_model)}"
        )

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

        self.logger.info(f"{conversation} Summary response: {str(summary)}")
        self.logger.info(
            f"{conversation} Referenced documents: {referenced_documents}"
        )

        return str(summary), referenced_documents
