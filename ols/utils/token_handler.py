"""Utility to handle tokens."""

import logging
from math import ceil

from llama_index.core.schema import NodeWithScore
from tiktoken import get_encoding

from ols.app.models.models import RagChunk
from ols.constants import (
    DEFAULT_TOKENIZER_MODEL,
    MINIMUM_CONTEXT_TOKEN_LIMIT,
    RAG_SIMILARITY_CUTOFF,
    TOKEN_BUFFER_WEIGHT,
)
from ols.src.prompts.prompt_generator import (
    restructure_history,
    restructure_rag_context_post,
    restructure_rag_context_pre,
)

logger = logging.getLogger(__name__)


class PromptTooLongError(Exception):
    """Prompt is too long."""


class TokenHandler:
    """This class handles tokens.

    Convert text to tokens.
    Get rough estimation of token count.
    Truncate text based on token limit.
    """

    def __init__(self, encoding_name: str = DEFAULT_TOKENIZER_MODEL) -> None:
        """Initialize the class instance."""
        # Note: We need an approximate tokens count.
        # For different models, exact tokens may vary due to different tokenizer.
        self._encoder = get_encoding(encoding_name)

    def text_to_tokens(self, text: str) -> list[int]:
        """Convert text to tokens.

        Args:
            text: context text, ex: "This is my doc"

        Returns:
            List of tokens, ex: [1, 2, 3, 4]
        """
        # return self._encoder.encode(text)
        # TODO: Better handling of stop token.
        return self._encoder.encode(text, allowed_special={"<|endoftext|>"})

    def tokens_to_text(self, tokens: list) -> str:
        """Convert tokens to text.

        Args:
            tokens: ex: [1, 2, 3, 4]

        Returns:
            text: ex "This is my doc"
        """
        return self._encoder.decode(tokens)

    @staticmethod
    def _get_token_count(tokens: list[int]) -> int:
        """Get approximate tokens count."""
        # Note: As we get approximate tokens count, we want to have enough
        # buffer so that there is less chance of under-estimation.
        # We increase by certain percentage to nearest integer (ceil).
        return ceil(len(tokens) * TOKEN_BUFFER_WEIGHT)

    def calculate_and_check_available_tokens(
        self, prompt: str, context_window_size: int, max_tokens_for_response: int
    ) -> int:
        """Get available tokens that can be used for prompt augmentation.

        Args:
            prompt: format prompt template to string before passing as arg
            context_window_size: context window size of LLM
            max_tokens_for_response: max tokens allowed for response (estimation)

        Returns:
            available_tokens: int, tokens that can be used for augmentation.
        """
        logger.debug(
            "Context window size: %d, Max generated tokens: %d",
            context_window_size,
            max_tokens_for_response,
        )

        prompt_token_count = TokenHandler._get_token_count(self.text_to_tokens(prompt))
        logger.debug("Prompt tokens: %d", prompt_token_count)

        # The context_window_size is the maximum number of tokens that
        # can be used for a "conversation" in the LLM model. This
        # includes prompt AND response. Hence we need to subtract the
        # prompt tokens and max tokens for response from the context
        # window size.
        available_tokens = (
            context_window_size - max_tokens_for_response - prompt_token_count
        )

        if available_tokens <= 0:
            limit = context_window_size - max_tokens_for_response
            raise PromptTooLongError(
                f"Prompt length {prompt_token_count} exceeds LLM "
                f"available context window limit {limit} tokens"
            )

        return available_tokens

    def truncate_rag_context(
        self, retrieved_nodes: list[NodeWithScore], model: str, max_tokens: int = 500
    ) -> tuple[list[RagChunk], int]:
        """Process retrieved node text and truncate if required.

        Args:
            retrieved_nodes: retrieved nodes object from index
            model: model name; required for adding proper tags
            max_tokens: maximum tokens allowed for rag context

        Returns:
            list of `RagChunk` objects, available tokens after context usage
        """
        rag_chunks = []

        for node in retrieved_nodes:

            score = float(node.get_score(raise_error=False))
            if score < RAG_SIMILARITY_CUTOFF:
                logger.debug(
                    "RAG content similarity score: %f is less than threshold %f.",
                    score,
                    RAG_SIMILARITY_CUTOFF,
                )
                break

            # Prepend all model specific tags, so that those will be considered for
            # token calculation. This requires formatting again after truncation,
            # whenever there are tags required at the end.
            # Alternative to this is to calculate tokens for special tags before
            # and add the number accordingly.
            # Example: Once token is calculated;
            # ```
            # if "granite" in model:
            #    tokens_count += 7
            # else:
            #    tokens_count += 3
            # ```
            node_text = node.get_text()
            node_text = restructure_rag_context_pre(node_text, model)
            tokens = self.text_to_tokens(node_text)
            tokens_count = TokenHandler._get_token_count(tokens)
            logger.debug("RAG content tokens count: %d.", tokens_count)

            available_tokens = min(tokens_count, max_tokens)
            logger.debug("Available tokens: %d.", tokens_count)

            if available_tokens < MINIMUM_CONTEXT_TOKEN_LIMIT:
                logger.debug("%d tokens are less than threshold.", available_tokens)
                break

            node_text = self.tokens_to_text(tokens[:available_tokens])
            node_text = restructure_rag_context_post(node_text, model)
            rag_chunks.append(
                RagChunk(
                    text=node_text,
                    doc_url=node.metadata.get("docs_url", ""),
                    doc_title=node.metadata.get("title", ""),
                )
            )

            max_tokens -= available_tokens

        return rag_chunks, max_tokens

    def limit_conversation_history(
        self, history: list[str], model: str, limit: int = 0
    ) -> tuple[list[str], bool]:
        """Limit conversation history to specified number of tokens."""
        total_length = 0
        formatted_history: list[str] = []

        for original_message in reversed(history):
            # Restructure messages as per model
            message = restructure_history(original_message, model)

            message_length = TokenHandler._get_token_count(self.text_to_tokens(message))
            total_length += message_length
            # if total length of already checked messages is higher than limit
            # then skip all remaining messages (we need to skip from top)
            if total_length > limit:
                logger.debug(
                    "History truncated, it exceeds available %d tokens.", limit
                )
                return formatted_history[::-1], True
            formatted_history.append(message)

        return formatted_history[::-1], False
