"""Class responsible for classifying a statement as yes, no, or undetermined."""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import constants
from ols.src.llms.llm_loader import LLMLoader
from ols.src.query_helpers import QueryHelper


class YesNoClassifier(QueryHelper):
    """This class is responsible for classifying a statement as yes, no, or undetermined."""

    def classify(self, conversation: str, statement: str, **kwargs) -> int:
        """Classifies a statement as yes, no, or undetermined.

        Args:
          conversation: The identifier for the conversation or task context.
          statement: The statement to be classified.
          **kwargs: Additional keyword arguments for customization.

        Returns:
            The classification result (1 for yes, 0 for no, 9 for undetermined).
        """
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = (
            f"conversation: {conversation}, "
            f"query: {statement}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, verbose: {verbose}"
        )
        self.logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.YES_OR_NO_CLASSIFIER_PROMPT_TEMPLATE
        )

        self.logger.info(f"{conversation} using model: {self.model}")
        self.logger.info(f"{conversation} determining yes/no: {statement}")
        query = prompt_instructions.format(statement=statement)

        self.logger.info(f"{conversation} yes/no query: {query}")
        self.logger.info(f"{conversation} using model: {self.model}")

        bare_llm = LLMLoader(self.provider, self.model).llm
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        response = llm_chain(inputs={"statement": statement})

        self.logger.info(f"{conversation} bare response: {response}")
        self.logger.info(f"{conversation} yes/no response: {response['text']}")

        if response["text"] not in ["0", "1", "9"]:
            msg = f"Returned response not 0, 1, or 9: {response['text']}"
            self.logger.error(msg)
            raise ValueError(msg)

        return int(response["text"])
