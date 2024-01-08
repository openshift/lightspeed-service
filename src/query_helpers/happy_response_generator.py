from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from src import constants
from src.llms.llm_loader import LLMLoader
from utils import config
from utils.logger import Logger


class HappyResponseGenerator:
    """This class is responsible for generating a pleasant response to a user question."""

    def __init__(self) -> None:
        """Initializes the HappyResponseGenerator instance."""
        self.logger = Logger("happy_response_generator").logger

    def generate(self, conversation, user_question, **kwargs):
        """Generates a pleasant response to a user question.

        Args:
        - conversation (str): The identifier for the conversation or task context.
        - user_question (str): The question posed by the user.
        - **kwargs: Additional keyword arguments for customization.

        Returns:
        - str: The generated happy response.
        """
        model = config.ols_config.validator_model
        provider = config.ols_config.validator_provider

        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = f"conversation: {conversation}, query: {user_question}, provider: {provider}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.HAPPY_RESPONSE_GENERATOR_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} using provider: {provider}")
        self.logger.info(f"{conversation} using model: {model}")
        self.logger.info(f"{conversation} user query: {user_question}")
        query = prompt_instructions.format(question=user_question)

        self.logger.info(f"{conversation} full prompt: {query}")

        bare_llm = LLMLoader(provider, model).llm
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        response = llm_chain(inputs={"question": user_question})

        self.logger.info(f"{conversation} happy response: {str(response['text'])}")

        return str(response["text"])
