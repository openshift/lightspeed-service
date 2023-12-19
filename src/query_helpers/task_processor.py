import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src import constants
from utils.model_context import get_watsonx_predictor
from query_helpers.yes_no_classifier import YesNoClassifier
from src.query_helpers.task_performer import TaskPerformer
from src.query_helpers.task_rephraser import TaskRephraser
from utils.logger import Logger

load_dotenv()


class TaskProcessor:
    """
    Class responsible for processing a list of tasks based on the provided input.
    """

    def __init__(self):
        """
        Initializes the TaskProcessor instance.
        """
        self.logger = Logger("task_processor").logger

    def process_tasks(self, conversation, tasklist, original_query, **kwargs):
        """
        Processes a list of tasks and returns the results.

        Args:
        - conversation (str): The identifier for the conversation or task context.
        - tasklist (list): A list of tasks to be processed.
        - original_query (str): The original query or information related to the tasks.
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - list: A list containing the response status and outputs.
        """
        model = kwargs.get(
            "model", os.getenv("TASK_PROCESSOR_MODEL", constants.GRANITE_13B_CHAT_V1)
        )
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = f"conversation: {conversation}, tasklist: {tasklist}, query: {original_query}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.TASK_PERFORMER_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} Beginning task processing")
        outputs = []

        self.logger.info(f"{conversation} using model: {model}")
        bare_llm = get_watsonx_predictor(model=model, min_new_tokens=5)
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        for task in tasklist:
            self.logger.info(f"{conversation} task: {task}")

            task_query = prompt_instructions.format(task=task, query=original_query)
            self.logger.info(f"{conversation} task query: {task_query}")

            response = llm_chain(inputs={"task": task, "query": original_query})
            self.logger.info(f"{conversation} task response: {str(response)}")

            clean_response = response["text"].split("")[0]

            yes_no_classifier = YesNoClassifier()
            response_status = int(
                yes_no_classifier.classify(conversation, clean_response)
            )
            self.logger.info(f"{conversation} response status: {str(response_status)}")

            if response_status == 0:
                self.logger.info(
                    f"{conversation} Aborting task processing for no response - need details"
                )
                resolution_request = f"In trying to answer your question, we were unable to determine how to proceed. The step we failed on was the following:\n {task}\n The failure message was:\n {clean_response}\n Please try rephrasing your request to include information that might help complete the task."
                self.logger.info(
                    f"{conversation} resolution request: {resolution_request}"
                )
                return [response_status, resolution_request]
            elif response_status == 1:
                task_rephraser = TaskRephraser()
                rephrased_task = task_rephraser.rephrase_task(
                    conversation, clean_response, original_query
                )

                task_performer = TaskPerformer()
                outputs.append(
                    task_performer.perform_task(
                        conversation, rephrased_task, original_query
                    )
                )
                self.logger.info(f"{conversation} current outputs: {str(outputs)}")
            else:
                self.logger.info(f"{conversation} Unknown response status")
                return [response_status, "Unknown error occurred"]

        return [1, outputs]
