"""Class responsible for validating questions and providing one-word responses."""

import logging
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import constants
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils import config

logger = logging.getLogger(__name__)


class QuestionValidator(QueryHelper):
    """This class is responsible for validating questions and providing one-word responses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the QuestionValidator."""
        llm_params = {
            "min_new_tokens": 1,
            "max_new_tokens": 4,
        }
        super().__init__(*args, **kwargs, llm_params=llm_params)

    def validate_question(
        self, conversation_id: str, query: str, verbose: bool = False
    ) -> str:
        """Validate a question and provides a one-word response.

        Args:
          conversation_id: The identifier for the conversation or task context.
          query: The question to be validated.
          verbose: If `LLMChain` should be verbose. Defaults to `False`.

        Returns:
            One-word response.
        """
        if config.dev_config.disable_question_validation:
            logger.debug(
                f"{conversation_id} Question validation is disabled. "
                f"Treating question as [valid,generic]."
            )
            return constants.SUBJECT_VALID

        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {verbose}"
        )
        logger.info(f"{conversation_id} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            constants.QUESTION_VALIDATOR_PROMPT_TEMPLATE
        )

        logger.info(f"{conversation_id} Validating query")

        bare_llm = self.llm_loader(
            self.provider, self.model, llm_params=self.llm_params
        ).llm
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        task_query = prompt_instructions.format(query=query)

        logger.info(f"{conversation_id} task query: {task_query}")

        response = llm_chain(inputs={"query": query})
        clean_response = str(response["text"]).strip()

        logger.info(f"{conversation_id} response: {clean_response}")

        # Will return list with one of the following:
        # SUBJECT_VALID
        # SUBJECT_INVALID
        return clean_response
