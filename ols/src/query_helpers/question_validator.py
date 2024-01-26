"""Class responsible for validating questions and providing one-word responses."""

import logging

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import constants
from ols.src.llms.llm_loader import LLMLoader
from ols.src.query_helpers import QueryHelper

logger = logging.getLogger(__name__)


class QuestionValidator(QueryHelper):
    """This class is responsible for validating questions and providing one-word responses."""

    def validate_question(
        self, conversation: str, query: str, verbose: bool = False
    ) -> list[str]:
        """Validate a question and provides a one-word response.

        Args:
          conversation: The identifier for the conversation or task context.
          query: The question to be validated.
          verbose: If `LLMChain` should be verbose. Defaults to `False`.

        Returns:
            A list of one-word responses.
        """
        settings_string = (
            f"conversation: {conversation}, "
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {verbose}"
        )
        logger.info(f"{conversation} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.QUESTION_VALIDATOR_PROMPT_TEMPLATE
        )

        logger.info(f"{conversation} Validating query")
        logger.info(f"{conversation} using model: {self.model}")

        bare_llm = LLMLoader(
            self.provider, self.model, params={"min_new_tokens": 1, "max_new_tokens": 4}
        ).llm

        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        task_query = prompt_instructions.format(query=query)

        logger.info(f"{conversation} task query: {task_query}")

        response = llm_chain(inputs={"query": query})
        clean_response = str(response["text"]).strip()

        logger.info(f"{conversation} response: {clean_response}")

        if response["text"] not in ["INVALID,NOYAML", "VALID,NOYAML", "VALID,YAML"]:
            msg = f"Returned response \"{response['text']}\" did not match the expected format"
            logger.error(msg)
            raise ValueError(msg)

        # will return an array:
        # [INVALID,NOYAML]
        # [VALID,NOYAML]
        # [VALID,YAML]
        return clean_response.split(",")
