"""A class for summarizing documentation context."""

import logging
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from llama_index.indices.vector_store.base import VectorStoreIndex

from ols import constants
from ols.app.metrics import TokenMetricUpdater
from ols.app.models.config import ProviderConfig
from ols.src.prompts.prompt_generator import generate_prompt
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils import config
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

    def _format_rag_data(self, rag_data: list[dict]) -> tuple[str, list[str]]:
        """Format rag text & metadata.

        Join multiple rag text with new line.
        Create list of metadata from rag data dictionary.
        """
        rag_text = []
        docs_url = []
        for data in rag_data:
            rag_text.append(data["text"])
            docs_url.append(data["docs_url"])

        return "\n\n".join(rag_text), docs_url

    def _get_rag_data(
        self,
        rag_index: VectorStoreIndex,
        query: str,
        provider_config: ProviderConfig,
        model_options: Optional[dict[str, Any]],
    ) -> list[dict]:
        """Get rag index data.

        Get relevant rag content based on query.
        Calculate available tokens.
        Returns rag content text & metadata as dictionary.
        """
        # TODO: Implement a different module for retrieval
        # with different options, currently it is top_k.
        # We do have a module with langchain, but not compatible
        # with llamaindex object.
        retriever = rag_index.as_retriever(similarity_top_k=1)
        retrieved_nodes = retriever.retrieve(query)

        token_config = config.llm_config.providers.get(self.provider)
        token_config = token_config.models.get(self.model)
        context_window_size = token_config.context_window_size
        response_token_limit = token_config.response_token_limit
        logger.info(
            f"context_window_size: {context_window_size}, "
            f"response_token_limit: {response_token_limit}"
        )
        # Truncate rag context, if required.
        token_handler_obj = TokenHandler()

        rag_context = " "
        prompt, _ = generate_prompt(
            self.provider,
            self.model,
            model_options,
            query,
            [],
            rag_context,
        )
        interim_prompt = prompt.format(
            context=rag_context, query=query, chat_history=[]
        )

        prompt_token_count = len(token_handler_obj.text_to_tokens(interim_prompt))
        available_tokens = (
            context_window_size - response_token_limit - prompt_token_count
        )
        # TODO: Now we have option to set context window & response limit set
        # from the config. With this we need to change default max token parameter
        # for the model dynamically. Also a check for
        # (response limit + prompt + any additional user context) < context window

        return token_handler_obj.truncate_rag_context(retrieved_nodes, available_tokens)

    def _get_model_options(
        self, provider_config: ProviderConfig
    ) -> Optional[dict[str, Any]]:
        if provider_config is not None:
            model_config = provider_config.models.get(self.model)
            return model_config.options
        return None

    def summarize(
        self,
        conversation_id: str,
        query: str,
        vector_index: Optional[VectorStoreIndex] = None,
        history: list[BaseMessage] = [],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Summarize the given query based on the provided conversation context.

        Args:
            conversation_id: The unique identifier for the conversation.
            query: The query to be summarized.
            vector_index: Vector index to get rag data/context.
            history: The history of the conversation (if available).
            kwargs: Additional keyword arguments for customization (model, verbose, etc.).

        Returns:
            A dictionary containing below property
            - the summary as a string
            - referenced documents as a list of strings
            - flag indicating that conversation history has been truncated
              to fit within context window.
        """
        verbose = kwargs.get("verbose", "").lower() == "true"
        settings_string = (
            f"conversation_id: {conversation_id}, "
            f"query: {query}, "
            f"provider: {self.provider}, "
            f"model: {self.model}, "
            f"verbose: {verbose}"
        )
        logger.info(f"{conversation_id} call settings: {settings_string}")

        bare_llm = self.llm_loader(self.provider, self.model, self.llm_params)

        rag_context_data: list[dict] = []
        referenced_documents: list[str] = []

        provider_config = config.llm_config.providers.get(self.provider)
        model_options = self._get_model_options(provider_config)

        if vector_index is not None:
            rag_context_data = self._get_rag_data(
                vector_index, query, provider_config, model_options
            )
        else:
            logger.warning("Proceeding without RAG content. Check start up messages.")

        token_handler = TokenHandler()

        rag_context, referenced_documents = self._format_rag_data(rag_context_data)

        dummy_history = [HumanMessage(content="dummy")]
        prompt, _ = generate_prompt(
            self.provider,
            self.model,
            model_options,
            query,
            dummy_history,
            rag_context,
        )
        interim_prompt = prompt.format(
            context=rag_context, query=query, chat_history=[]
        )

        prompt_token_count = len(token_handler.text_to_tokens(interim_prompt))

        if provider_config is not None:
            model_config = provider_config.models.get(self.model)
            response_token_limit = model_config.response_token_limit
            context_window_size = model_config.context_window_size
            available_tokens = (
                context_window_size - response_token_limit - prompt_token_count
            )

            history, truncated = TokenHandler.limit_conversation_history(
                history, available_tokens
            )
        else:
            truncated = False

        final_prompt, llm_input_values = generate_prompt(
            self.provider,
            self.model,
            model_options,
            query,
            history,
            rag_context,
        )

        chat_engine = LLMChain(
            llm=bare_llm,
            prompt=final_prompt,
            verbose=verbose,
        )
        with TokenMetricUpdater(
            llm=bare_llm,
            provider=self.provider,
            model=self.model,
        ) as token_counter:
            summary = chat_engine.invoke(
                input=llm_input_values,
                config={"callbacks": [token_counter]},
            )

        response = summary["text"]

        if len(rag_context) == 0:
            logger.info("Using llm to answer the query without reference content")
            response = constants.NO_RAG_CONTENT_RESP + str(response)

        logger.info(f"{conversation_id} Summary response: {response}")
        logger.info(f"{conversation_id} Referenced documents: {referenced_documents}")

        return {
            "response": response,
            "referenced_documents": referenced_documents,
            "history_truncated": truncated,
        }
