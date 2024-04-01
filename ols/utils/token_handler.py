"""Utility to handle tokens."""

import logging

from langchain_core.messages.base import BaseMessage
from llama_index.schema import NodeWithScore
from tiktoken import get_encoding

from ols.app.models.config import ModelConfig

# TODO: distribute these constants if needed
DEFAULT_TOKENIZER_MODEL = "cl100k_base"

MINIMUM_CONTEXT_LIMIT = 1
SIMILARITY_SCORE_THRESHOLD = 0.5


logger = logging.getLogger(__name__)


class TokenHandler:
    """This class handles tokens.

    Convert text to tokens.
    Get rough estimation of token count.
    Truncate text based on token limit.
    """

    def __init__(self, encoding_name: str = DEFAULT_TOKENIZER_MODEL) -> None:
        """Initialize the class instance."""
        self._encoder = get_encoding(encoding_name)

    def text_to_tokens(self, text: str) -> list[int]:
        """Convert text to tokens.

        Args:
            text: context text, ex: "This is my doc"

        Returns:
            List of tokens, ex: [1, 2, 3, 4]
        """
        # Note: We need an approximate tokens count.
        # For different models, exact tokens count may vary.
        return self._encoder.encode(text)

    def tokens_to_text(self, tokens: list) -> str:
        """Convert tokens to text.

        Args:
            tokens: ex: [1, 2, 3, 4]

        Returns:
            text: ex "This is my doc"
        """
        return self._encoder.decode(tokens)

    def message_to_tokens(self, message: BaseMessage) -> list[int]:
        """Convert message (ie. HumanMessage etc.) to tokens.

        Args:
            message: instance of any class derived from BaseMessage

        Returns:
            List of tokens, ex: [1, 2, 3, 4]
        """
        content = message.content

        # content is either string or list of strings
        if isinstance(content, str):
            return self.text_to_tokens(content)
        content = " ".join(content)
        return self.text_to_tokens(content)

    def get_available_tokens(self, prompt: str, model_config: ModelConfig) -> int:
        """Get available tokens that can be used for prompt augmentation.

        Args:
            prompt: format prompt template to string before passing as arg
            model_config: model config to get other tokens spec.

        Returns:
            available_tokens: int, tokens that can be used for augmentation.
        """
        context_window_size = model_config.context_window_size
        response_token_limit = model_config.response_token_limit
        logger.debug(
            f"Context window size: {context_window_size}, "
            f"Response token limit: {response_token_limit}"
        )

        prompt_token_count = len(self.text_to_tokens(prompt))
        logger.debug(f"Prompt tokens: {prompt_token_count}")

        available_tokens = (
            context_window_size - response_token_limit - prompt_token_count
        )
        return available_tokens

    def truncate_rag_context(
        self, retrieved_nodes: list[NodeWithScore], max_tokens: int = 500
    ) -> tuple[list[dict], int]:
        """Process retrieved node text and truncate if required.

        Args:
            retrieved_nodes: retrieved nodes object from index
            max_tokens: maximum tokens allowed for rag context

        Returns:
            context: A list of dictionary containing text & metadata
            max_tokens: int, available tokens after context usage
            Context Example:
                [
                    {
                        "text": "This is my doc",
                        "file_name": "doc1.pdf",
                        "doc_link": link1,
                    },
                ]
        """
        context = []

        for node in retrieved_nodes:
            context_dict = {}

            tokens = self.text_to_tokens(node.get_text())
            tokens_count = len(tokens)
            logger.info(f"Tokens count: {tokens_count}.")
            available_tokens = min(tokens_count, max_tokens)

            if available_tokens < MINIMUM_CONTEXT_LIMIT:
                logger.warning(f"{available_tokens} tokens are less than threshold.")
                break

            context_dict["text"] = self.tokens_to_text(tokens[:available_tokens])
            # Add Metadata
            context_dict["docs_url"] = node.metadata.get("docs_url", None)

            context.append(context_dict)
            max_tokens -= available_tokens

        return context, max_tokens

    @staticmethod
    def limit_conversation_history(
        history: list[BaseMessage], limit: int = 0
    ) -> tuple[list[BaseMessage], bool]:
        """Limit conversation history to specified number of tokens."""
        total_length = 0
        index = 0

        token_handler_obj = TokenHandler()

        for message in reversed(history):
            message_length = len(token_handler_obj.message_to_tokens(message))
            total_length += message_length
            # if total length of already checked messages is higher than limit
            # then skip all remaining messages (we need to skip from top)
            if total_length > limit:
                return history[len(history) - index :], True
            index += 1
        return history, False
