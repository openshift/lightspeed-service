"""Class responsible for classifying a statement as yes, no, or undetermined."""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols.src import constants
from ols.src.llms.llm_loader import LLMLoader
from ols.utils import config
from ols.utils.logger import Logger


class YesNoClassifier:
    """This class is responsible for classifying a statement as yes, no, or undetermined."""

    def __init__(self) -> None:
        """Initialize the `YesNoClassifier` instance."""
        self.logger = Logger("yes_no_classifier").logger

    def classify(self, conversation: str, statement: str, **kwargs) -> int:
        """Classifies a statement as yes, no, or undetermined.

        Args:
          conversation: The identifier for the conversation or task context.
          statement: The statement to be classified.
          **kwargs: Additional keyword arguments for customization.

        Returns:
            The classification result (1 for yes, 0 for no, 9 for undetermined).
        """
        model = config.ols_config.validator_model
        provider = config.ols_config.validator_provider
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = f"conversation: {conversation}, query: {statement}, provider: {provider}, model: {model}, verbose: {verbose}"
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.YES_OR_NO_CLASSIFIER_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} using model: {model}")
        self.logger.info(f"{conversation} determining yes/no: {statement}")
        query = prompt_instructions.format(statement=statement)

        self.logger.info(f"{conversation} yes/no query: {query}")
        self.logger.info(f"{conversation} using model: {model}")

        bare_llm = LLMLoader(provider, model).llm
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        response = llm_chain(inputs={"statement": statement})

        self.logger.info(f"{conversation} bare response: {response}")
        self.logger.info(f"{conversation} yes/no response: {response['text']}")

        if response["text"] not in ["0", "1", "9"]:
            raise ValueError("Returned response not 0, 1, or 9")

        return int(response["text"])
