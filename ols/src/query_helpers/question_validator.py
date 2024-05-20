"""Class responsible for validating questions and providing one-word responses."""

import logging
from typing import Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import config
from ols.app.metrics import TokenMetricUpdater
from ols.constants import SUBJECT_REJECTED, GenericLLMParameters
from ols.src.prompts.prompts import QUESTION_VALIDATOR_PROMPT_TEMPLATE
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


class QuestionValidator(QueryHelper):
    """This class is responsible for validating questions and providing one-word responses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the QuestionValidator."""
        generic_llm_params = {GenericLLMParameters.MAX_NEW_TOKENS: 4}
        super().__init__(*args, **dict(kwargs, generic_llm_params=generic_llm_params))

    def validate_question(
        self, conversation_id: str, query: str, verbose: bool = False
    ) -> bool:
        """Validate a question and provides a one-word response.

        Args:
          conversation_id: The identifier for the conversation or task context.
          query: The question to be validated.
          verbose: If `LLMChain` should be verbose. Defaults to `False`.

        Returns:
            bool: true/false indicating if the question was deemed valid
        """
        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {verbose}"
        )
        logger.info(f"{conversation_id} call settings: {settings_string}")

        prompt_instructions = PromptTemplate.from_template(
            QUESTION_VALIDATOR_PROMPT_TEMPLATE
        )

        bare_llm = self.llm_loader(self.provider, self.model, self.generic_llm_params)

        # we just need to compute prompt length (in tokens) and check
        # if it's in context window limit
        provider_config = config.llm_config.providers.get(self.provider)
        model_config = provider_config.models.get(self.model)
        TokenHandler().calculate_and_check_available_tokens(query, model_config)

        llm_chain = LLMChain(
            llm=bare_llm,
            prompt=prompt_instructions,
            verbose=verbose,
        )

        logger.debug(f"{conversation_id} validating user query: {query}")

        with TokenMetricUpdater(
            llm=bare_llm,
            provider=self.provider,
            model=self.model,
        ) as token_counter:
            response = llm_chain.invoke(
                input={"query": query}, config={"callbacks": [token_counter]}
            )
        clean_response = str(response["text"]).strip()

        logger.debug(f"{conversation_id} query validation response: {clean_response}")

        # Default to be permissive(allow the question) if we don't get a clean
        # rejection from the LLM.
        return SUBJECT_REJECTED not in clean_response
