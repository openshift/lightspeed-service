# base python things
import os
from dotenv import load_dotenv

# external deps
import llama_index
from llama_index import StorageContext, load_index_from_storage
from llama_index.prompts import Prompt, PromptTemplate

# internal modules
from modules.model_context import get_watsonx_context

# internal tools
from tools.ols_logger import OLSLogger

load_dotenv()
DEFAULT_MODEL = os.getenv("DOC_SUMMARIZER", "ibm/granite-13b-chat-v1")


class DocsSummarizer:
    def __init__(self):
        self.logger = OLSLogger("docs_summarizer").logger

    def summarize(self, conversation, query, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = DEFAULT_MODEL

        if "verbose" in kwargs:
            if kwargs["verbose"] == 'True' or kwargs["verbose"] == 'true':
                verbose = True
            else:
                verbose = False
        else:
            verbose = False

        # make llama index show the prompting
        if verbose == True:
            llama_index.set_global_handler("simple")

        settings_string = f"conversation: {conversation}, query: {query},model: {model}, verbose: {verbose}"
        self.logger.info(
            conversation
            + " call settings: "
            + settings_string
        )

        summarization_template_str = """
The following context contains several pieces of documentation. Please summarize the context for the user.
Documentation context:
{context_str}

Summary:

"""
        summarization_template = PromptTemplate(
            summarization_template_str
        )

        self.logger.info(conversation + " Getting sevice context")
        self.logger.info(conversation + " using model: " + model)
        service_context = get_watsonx_context(model=model)

        storage_context = StorageContext.from_defaults(persist_dir="vector-db/ocp-product-docs")
        self.logger.info(conversation + " Setting up index")
        index = load_index_from_storage(
            storage_context=storage_context,
            index_id="product",
            service_context=service_context,
        )

        self.logger.info(conversation + " Setting up query engine")
        query_engine = index.as_query_engine(
            text_qa_template=summarization_template,
            verbose=verbose,
            streaming=False, similarity_top_k=1
        )

        # TODO: figure out how to log the full query sent to the query engine in a better way

        self.logger.info(conversation + " Submitting summarization query")
        summary = query_engine.query(query)

        referenced_documents = ""
        for source_node in summary.source_nodes:
            # print(source_node.node.metadata['file_name'])
            referenced_documents += source_node.node.metadata["file_name"] + "\n"

        self.logger.info(conversation + " Summary response: " + str(summary))
        for line in referenced_documents.splitlines():
            self.logger.info(conversation + " Referenced documents: " + line)

        return str(summary), referenced_documents


if __name__ == "__main__":
    """to execute, from the repo root, use python -m modules.docs_fetcher"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Search the RAG and find relevant product documentation, and summarize it"
    )
    parser.add_argument(
        "-c",
        "--conversation-id",
        default="1234",
        type=str,
        help="A short identifier for the conversation",
    )
    parser.add_argument(
        "-q",
        "--query",
        default="What is the weather like today?",
        type=str,
        help="The user query to use",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, type=str, help="The model to use"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        help="Set Verbose status of langchains [True/False]",
    )

    args = parser.parse_args()

    docs_summarizer = DocsSummarizer()
    docs_summarizer.summarize(
        args.conversation_id, args.query, model=args.model, verbose=args.verbose
    )
