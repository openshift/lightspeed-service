import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import src.constants as constants
from utils.model_context import get_watsonx_predictor
from utils.logger import Logger

load_dotenv()


class YamlGenerator:
    """
    This class is responsible for generating YAML responses to user requests.
    """

    def __init__(self):
        """
        Initializes the YamlGenerator instance.
        """
        self.logger = Logger("yaml_generator").logger

    def generate_yaml(self, conversation_id, query, history=None, **kwargs):
        """
        Generates YAML response to a user request.

        Args:
        - conversation_id (str): The identifier for the conversation or task context.
        - query (str): The user request.
        - history (str): The history of the conversation (if available).
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - str: The generated YAML response.
        """
        model = kwargs.get(
            "model", os.getenv("YAML_MODEL", constants.GRANITE_20B_CODE_INSTRUCT_V1)
        )
        verbose = kwargs.get("verbose", "").lower() == "true"
        settings_string = f"conversation: {conversation_id}, query: {query}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation_id} call settings: {settings_string}")
        self.logger.info(f"{conversation_id} using model: {model}")
        bare_llm = get_watsonx_predictor(model=model)

        if history:
            prompt_instructions = PromptTemplate.from_template(
                constants.YAML_GENERATOR_WITH_HISTORY_PROMPT_TEMPLATE
            )
            task_query = prompt_instructions.format(query=query, history=history)
        else:
            prompt_instructions = PromptTemplate.from_template(
                constants.YAML_GENERATOR_PROMPT_TEMPLATE
            )
            task_query = prompt_instructions.format(query=query)

        self.logger.info(f"{conversation_id} task query: {task_query}")
        llm_chain = LLMChain(llm=bare_llm, verbose=verbose, prompt=prompt_instructions)
        response = llm_chain(inputs={"query": query, "history": history})
        self.logger.info(f"{conversation_id} response:\n{response['text']}")
        return response["text"]
