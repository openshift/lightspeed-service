"""A class for summarizing documentation context."""

import logging
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from llama_index.indices.vector_store.base import VectorStoreIndex

from ols.app.metrics import TokenMetricUpdater
from ols.app.models.config import ProviderConfig
from ols.app.models.models import ReferencedDocument
from ols.constants import NO_RAG_CONTENT_RESP, RAG_CONTENT_LIMIT
from ols.src.prompts.prompt_generator import generate_prompt
from ols.src.query_helpers.query_helper import QueryHelper
from ols.utils import config
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


class DocsSummarizer(QueryHelper):
    """A class for summarizing documentation context."""

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
        logger.debug(f"{conversation_id} call settings: {settings_string}")

        token_handler = TokenHandler()
        bare_llm = self.llm_loader(self.provider, self.model, self.llm_params)

        provider_config = config.llm_config.providers.get(self.provider)
        model_config = provider_config.models.get(self.model)
        model_options = self._get_model_options(provider_config)

        # Use dummy text for context/history to get complete prompt instruction.
        # This is used to calculate available tokens.
        temp_prompt, temp_prompt_input = generate_prompt(
            self.provider,
            self.model,
            model_options,
            query,
            [HumanMessage(content="dummy")],
            "dummy",
        )
        available_tokens = token_handler.get_available_tokens(
            temp_prompt.format(**temp_prompt_input), model_config
        )

        # Get RAG context, truncate if applicable.
        rag_context_data: dict[str, list[str]] = {}

        if vector_index is not None:
            retriever = vector_index.as_retriever(similarity_top_k=RAG_CONTENT_LIMIT)
            rag_context_data, available_tokens = token_handler.truncate_rag_context(
                retriever.retrieve(query), available_tokens
            )
        else:
            logger.warning("Proceeding without RAG content. Check start up messages.")

        rag_context = "\n\n".join(rag_context_data.get("text", []))
        referenced_documents = [
            ReferencedDocument(docs_url=docs_url, title=title)
            for docs_url, title in zip(
                rag_context_data.get("docs_url", []), rag_context_data.get("title", [])
            )
        ]

        # Truncate history, if applicable
        history, truncated = token_handler.limit_conversation_history(
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
            logger.debug("Using llm to answer the query without reference content")
            response = NO_RAG_CONTENT_RESP + str(response)

        logger.debug(f"{conversation_id} Summary response: {response}")
        logger.debug(f"{conversation_id} Referenced documents: {referenced_documents}")

        return {
            "response": response,
            "referenced_documents": referenced_documents,
            "history_truncated": truncated,
        }
