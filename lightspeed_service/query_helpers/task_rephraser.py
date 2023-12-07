import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from lightspeed_service import constants
from lightspeed_service.utils.logger import Logger
from lightspeed_service.utils.model_context import get_watsonx_predictor


load_dotenv()


class TaskRephraser:
    """
    This class is responsible for rephrasing a given task and query into
    a single, new task.
    """

    def __init__(self):
        """
        Initializes the TaskRephraser instance.
        """
        self.logger = Logger("task_rephraser").logger

    def rephrase_task(self, conversation, task, original_query, **kwargs):
        """
        Rephrases a given task and query into a single, new task.

        Args:
        - conversation (str): The identifier for the conversation or
            task context.
        - task (str): The original task.
        - original_query (str): The original query or information
            related to the task.
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - str: The rephrased task.
        """
        model = kwargs.get(
            "model", os.getenv("REPHRASE_MODEL", constants.GRANITE_13B_CHAT_V1)
        )
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = (
            f"conversation: {conversation}, query: {original_query}, model: "
            f"{model}, verbose: {verbose}"
        )
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.TASK_REPHRASER_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} Rephrasing task and query")
        self.logger.info(f"{conversation} using model: {model}")

        bare_llm = get_watsonx_predictor(model=model, min_new_tokens=5)
        llm_chain = LLMChain(
            llm=bare_llm, prompt=prompt_instructions, verbose=verbose
        )

        task_query = prompt_instructions.format(
            task=task, query=original_query
        )

        self.logger.info(f"{conversation} task query: {task_query}")

        response = llm_chain(inputs={"task": task, "query": original_query})

        self.logger.info(f"{conversation} response: {str(response)}")
        return response["text"]
