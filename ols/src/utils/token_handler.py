"""Utility to handle tokens."""

import logging

import tiktoken

from ols.src.utils.constants import (
    MINIMUM_CONTEXT_LIMIT,
    TOKENIZER_MODEL,
)

logger = logging.getLogger(__name__)


class TokenHandler:
    """This class handles tokens.

    Convert text to tokens.
    Get rough estimation of token count.
    Truncate text based on token limit.
    """

    def __init__(self):
        """Initialize the class instance."""
        self._encoder = tiktoken.get_encoding(TOKENIZER_MODEL)

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

    def truncate_rag_context(
        self, retrieved_nodes, max_tokens: int = 500
    ) -> list[dict]:
        """Process retrieved node text and truncate if required.

        Args:
            retrieved_nodes: retrieved nodes object from index
            max_tokens: maximum tokens allowed for rag context

        Returns:
            context: A list of dictionary containing text & metadata
            Example:
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
            context_dict["file_name"] = node.metadata.get("file_name", None)
            # TODO: Below metadata yet to be added.
            context_dict["doc_link"] = node.metadata.get("doc_link", None)

            context.append(context_dict)
            max_tokens -= available_tokens

        return context
