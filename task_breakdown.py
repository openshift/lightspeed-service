import llama_index
from llama_index import StorageContext, load_index_from_storage
from model_context import get_watsonx_context
from llama_index.prompts import Prompt, PromptTemplate
import logging
import sys


class TaskBreakdown:
    def __init__(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("task_breakdown")

    def breakdown_tasks(self, conversation, model, query):
        # make llama index show the prompting
        llama_index.set_global_handler("simple")
        summary_task_breakdown_template_str = (
            "{context_str}\n"
            "Given the previous summary documentation, what steps would you take to answer the following question: {query_str}\n"
        )
        summary_task_breakdown_template = PromptTemplate(
            summary_task_breakdown_template_str
        )

        self.logger.info(conversation + " Getting sevice context")
        self.logger.info(conversation + " using model: " + model)
        service_context = get_watsonx_context(model=model)

        storage_context = StorageContext.from_defaults(persist_dir="../vector-db")
        self.logger.info(conversation + " Setting up index")
        index = load_index_from_storage(
            storage_context=storage_context,
            index_id="summary",
            service_context=service_context,
        )

        self.logger.info(conversation + " Setting up query engine")
        query_engine = index.as_query_engine(
            text_qa_template=summary_task_breakdown_template,
            verbose=True,
            streaming=False,
        )

        self.logger.info(conversation + " Submitting task breakdown query")
        task_breakdown_response = query_engine.query(query)

        referenced_documents = ""
        for source_node in task_breakdown_response.source_nodes:
            # print(source_node.node.metadata['file_name'])
            referenced_documents += source_node.node.metadata["file_name"] + "\n"

        for line in str(task_breakdown_response).splitlines():
            self.logger.info(conversation + " Task breakdown response: " + line)
        for line in referenced_documents.splitlines():
            self.logger.info(conversation + " Referenced documents: " + line)

        return str(task_breakdown_response).splitlines(), referenced_documents


if __name__ == "__main__":
    task_breakdown = TaskBreakdown()
    # arg 1 is the conversation id
    # arg 2 is the desired model
    # arg 3 is a quoted string to pass as the query
    task_breakdown.breakdown_tasks(sys.argv[1], sys.argv[2], sys.argv[3])
