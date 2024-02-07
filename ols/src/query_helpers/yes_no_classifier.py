"""Class responsible for classifying a statement as yes, no, or undetermined."""

import logging

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import constants
from ols.src.llms.llm_loader import LLMLoader
from ols.src.query_helpers import QueryHelper

logger = logging.getLogger(__name__)


class YesNoClassifier(QueryHelper):
    """This class is responsible for classifying a statement as yes, no, or undetermined."""

    def classify(self, conversation_id: str, statement: str, **kwargs) -> int:
        """Classifies a statement as yes, no, or undetermined.

        Args:
          conversation_id: The identifier for the conversation or task context.
          statement: The statement to be classified.
          **kwargs: Additional keyword arguments for customization.

        Returns:
            The classification result (1 for yes, 0 for no, 9 for undetermined).
        """
        verbose = kwargs.get("verbose", "").lower() == "true"

        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {statement}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, verbose: {verbose}"
        )
        logger.info(f"{conversation_id} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.YES_OR_NO_CLASSIFIER_PROMPT_TEMPLATE
        )

        logger.info(f"{conversation_id} determining yes/no: {statement}")
        query = prompt_instructions.format(statement=statement)

        logger.info(f"{conversation_id} yes/no query: {query}")

        bare_llm = LLMLoader(self.provider, self.model).llm
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        response = llm_chain(inputs={"statement": statement})

        logger.info(f"{conversation_id} bare response: {response}")
        logger.info(f"{conversation_id} yes/no response: {response['text']}")

        if response["text"] not in ["0", "1", "9"]:
            msg = f"Returned response not 0, 1, or 9: {response['text']}"
            logger.error(msg)
            raise ValueError(msg)

        return int(response["text"])
