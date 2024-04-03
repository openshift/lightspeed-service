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
        available_tokens: int,
    ) -> tuple[list[dict], int]:
        """Get rag index data.

        Get relevant rag content based on query.
        Returns rag content text & metadata as dictionary, Available tokens.
        """
        # TODO: Implement a different module for retrieval
        # with different options, currently it is top_k.
        # We do have a module with langchain, but not compatible
        # with llamaindex object.
        retriever = rag_index.as_retriever(similarity_top_k=1)
        retrieved_nodes = retriever.retrieve(query)

        # Truncate rag context, if required.
        token_handler_obj = TokenHandler()

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
        model_config = provider_config.models.get(self.model)
        model_options = self._get_model_options(provider_config)

        # Use dummy text for context/history to get complete prompt instruction.
        temp_prompt, temp_prompt_input = generate_prompt(
            self.provider,
            self.model,
            model_options,
            query,
            [HumanMessage(content="dummy")],
            "dummy",
        )
        token_handler = TokenHandler()
        available_tokens = token_handler.get_available_tokens(
            temp_prompt.format(**temp_prompt_input), model_config
        )

        if vector_index is not None:
            rag_context_data, available_tokens = self._get_rag_data(
                vector_index, query, available_tokens
            )
        else:
            logger.warning("Proceeding without RAG content. Check start up messages.")

        rag_context, referenced_documents = self._format_rag_data(rag_context_data)

        history, truncated = TokenHandler.limit_conversation_history(
            history, available_tokens
        )

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
