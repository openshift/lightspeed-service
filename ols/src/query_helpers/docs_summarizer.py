"""A class for summarizing documentation context."""

import logging
from typing import Any, AsyncGenerator, Optional

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import VectorStoreIndex

from ols import config
from ols.app.metrics import TokenMetricUpdater
from ols.app.models.models import RagChunk, SummarizerResponse
from ols.constants import RAG_CONTENT_LIMIT, GenericLLMParameters
from ols.src.prompts.prompt_generator import GeneratePrompt
from ols.src.prompts.prompts import QUERY_SYSTEM_INSTRUCTION
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the QuestionValidator."""
        super().__init__(*args, **kwargs)
        self._prepare_llm()
        self._get_system_prompt()
        self.verbose = config.ols_config.logging_config.app_log_level == logging.DEBUG

    def _prepare_llm(self) -> None:
        """Prepare the LLM configuration."""
        self.provider_config = config.llm_config.providers.get(self.provider)
        self.model_config = self.provider_config.models.get(self.model)
        self.generic_llm_params = {
            GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: self.model_config.parameters.max_tokens_for_response  # noqa: E501
        }
        self.bare_llm = self.llm_loader(
            self.provider, self.model, self.generic_llm_params
        )

    def _get_system_prompt(self) -> None:
        """Retrieve the system prompt."""
        # use system prompt from config if available otherwise use
        # default system prompt fine-tuned for the service
        if config.ols_config.system_prompt is not None:
            self.system_prompt = config.ols_config.system_prompt
        else:
            self.system_prompt = QUERY_SYSTEM_INSTRUCTION
        logger.debug("System prompt: %s", self.system_prompt)

    def _prepare_prompt(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[str]] = None,
    ) -> tuple[ChatPromptTemplate, dict[str, str], list[RagChunk], bool]:
        """Summarize the given query based on the provided conversation context.

        Args:
            query: The query to be summarized.
            vector_index: Vector index to get RAG data/context.
            history: The history of the conversation (if available).

        Returns:
            A tuple containing the final prompt, input values, RAG chunks,
            and a flag for truncated history.
        """
        settings_string = (
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {self.verbose}"
        )
        logger.debug("call settings: %s", settings_string)

        token_handler = TokenHandler()

        # Use sample text for context/history to get complete prompt
        # instruction. This is used to calculate available tokens.
        temp_prompt, temp_prompt_input = GeneratePrompt(
            query, ["sample"], ["ai: sample"], self.system_prompt
        ).generate_prompt(self.model)

        available_tokens = token_handler.calculate_and_check_available_tokens(
            temp_prompt.format(**temp_prompt_input),
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
        )

        # Retrieve RAG content
        if vector_index:
            retriever = vector_index.as_retriever(similarity_top_k=RAG_CONTENT_LIMIT)
            rag_chunks, available_tokens = token_handler.truncate_rag_context(
                retriever.retrieve(query), self.model, available_tokens
            )
        else:
            logger.warning("Proceeding without RAG content. Check start up messages.")
            rag_chunks = []
        rag_context = [rag_chunk.text for rag_chunk in rag_chunks]
        if len(rag_context) == 0:
            logger.debug("Using llm to answer the query without reference content")

        # Truncate history
        history, truncated = token_handler.limit_conversation_history(
            history or [], self.model, available_tokens
        )

        final_prompt, llm_input_values = GeneratePrompt(
            query, rag_context, history, self.system_prompt
        ).generate_prompt(self.model)

        # Tokens-check: We trigger the computation of the token count
        # without care about the return value. This is to ensure that
        # the query is within the token limit.
        token_handler.calculate_and_check_available_tokens(
            final_prompt.format(**llm_input_values),
            self.model_config.context_window_size,
            self.model_config.parameters.max_tokens_for_response,
        )

        return final_prompt, llm_input_values, rag_chunks, truncated

    def create_response(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[str]] = None,
    ) -> SummarizerResponse:
        """Create a response for the given query based on the provided conversation context."""
        final_prompt, llm_input_values, rag_chunks, truncated = self._prepare_prompt(
            query, vector_index, history
        )

        chat_engine = LLMChain(
            llm=self.bare_llm,
            prompt=final_prompt,
            verbose=self.verbose,
        )

        with TokenMetricUpdater(
            llm=self.bare_llm,
            provider=self.provider_config.type,
            model=self.model,
        ) as generic_token_counter:
            summary = chat_engine.invoke(
                input=llm_input_values,
                config={"callbacks": [generic_token_counter]},
            )

        # retrieve text response returned from LLM, strip whitespace characters from beginning/end
        response = summary["text"].strip()
        # TODO: Better handling of stop token.
        # Recently watsonx/granite-13b started adding stop token to response.
        response = response.replace("<|endoftext|>", "")

        return SummarizerResponse(
            response, rag_chunks, truncated, generic_token_counter.token_counter
        )

    async def generate_response(
        self,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, SummarizerResponse]:
        """Generate a response for the given query based on the provided conversation context."""
        final_prompt, llm_input_values, rag_chunks, truncated = self._prepare_prompt(
            query, vector_index, history
        )

        with TokenMetricUpdater(
            llm=self.bare_llm,
            provider=self.provider_config.type,
            model=self.model,
        ) as generic_token_counter:
            async for chunk in self.bare_llm.astream(
                final_prompt.format_prompt(**llm_input_values).to_messages(),
                config={"callbacks": [generic_token_counter]},
            ):
                # TODO: it is bad to have provider specific code here
                # the reason we have provider classes is to hide specific
                # implementation details there. But it requires expanding
                # the current providers interface, eg. to stream messages

                # openai returns an `AIMessageChunk` while Watsonx plain string
                chunk_content = chunk.content if hasattr(chunk, "content") else chunk
                if "<|endoftext|>" in chunk_content:
                    chunk_content = chunk_content.replace("<|endoftext|>", "")
                yield chunk_content

        yield SummarizerResponse("", rag_chunks, truncated, generic_token_counter.token_counter)  # type: ignore[misc]
