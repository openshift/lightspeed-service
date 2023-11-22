import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import src.constants as constants
from utils.model_context import get_watsonx_predictor
from utils.logger import Logger

load_dotenv()


class QuestionValidator:
    """
    This class is responsible for validating questions and providing one-word responses.
    """

    def __init__(self):
        """
        Initializes the QuestionValidator instance.
        """
        self.logger = Logger("question_validator").logger

    def validate_question(self, conversation, query, **kwargs):
        """
        Validates a question and provides a one-word response.

        Args:
        - conversation (str): The identifier for the conversation or task context.
        - query (str): The question to be validated.
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - list: A list of one-word responses.
        """
        model = kwargs.get(
            "model",
            os.getenv(
                "QUESTION_VALIDATOR_MODEL", constants.GRANITE_20B_CODE_INSTRUCT_V1
            ),
        )
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = f"conversation: {conversation}, query: {query}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.QUESTION_VALIDATOR_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} Validating query")
        self.logger.info(f"{conversation} using model: {model}")

        bare_llm = get_watsonx_predictor(
            model=model, min_new_tokens=1, max_new_tokens=4
        )
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        task_query = prompt_instructions.format(query=query)

        self.logger.info(f"{conversation} task query: {task_query}")

        response = llm_chain(inputs={"query": query})
        clean_response = str(response["text"]).strip()

        self.logger.info(f"{conversation} response: {clean_response}")

        # will return an array:
        # [INVALID,NOYAML]
        # [VALID,NOYAML]
        # [VALID,YAML]
        return clean_response.split(",")
