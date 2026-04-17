"""Utility to handle tokens."""

import logging
from enum import Enum
from math import ceil

from langchain_core.messages import BaseMessage
from llama_index.core.schema import NodeWithScore
from tiktoken import get_encoding

from ols.app.models.models import RagChunk
from ols.constants import (
    DEFAULT_TOKENIZER_MODEL,
    MINIMUM_CONTEXT_TOKEN_LIMIT,
    RAG_SIMILARITY_CUTOFF,
    TOKEN_BUFFER_WEIGHT,
)

logger = logging.getLogger(__name__)


def format_retrieved_chunk(rag_content: str) -> str:
    """Format a RAG document chunk with the standard prefix."""
    return f"Document:\n{rag_content}"


class TokenCategory(Enum):
    """Named categories for token budget accounting."""

    PROMPT = "prompt"
    HISTORY = "history"
    RAG = "rag"
    SKILL = "skill"
    TOOL_DEFINITIONS = "tool_definitions"
    AI_ROUND = "ai_round"
    TOOL_RESULT = "tool_result"


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
        # Also the provider may add model specific tags.
        self._encoder = get_encoding(encoding_name)

    def text_to_tokens(self, text: str) -> list[int]:
        """Convert text to tokens.

        Args:
            text: context text, ex: "This is my doc"

        Returns:
            List of tokens, ex: [1, 2, 3, 4]
        """
        return self._encoder.encode(text)

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

    def truncate_rag_context(
        self, retrieved_nodes: list[NodeWithScore], max_tokens: int = 500
    ) -> list[RagChunk]:
        """Process retrieved node text and truncate if required.

        Args:
            retrieved_nodes: retrieved nodes object from index
            max_tokens: maximum tokens allowed for rag context

        Returns:
            List of `RagChunk` objects that fit within the token budget.
        """
        rag_chunks = []
        logger.info(
            "Processing %d retrieved nodes for RAG context", len(retrieved_nodes)
        )

        for idx, node in enumerate(retrieved_nodes):
            score = float(node.get_score(raise_error=False))
            doc_title = node.metadata.get("title", "unknown")
            doc_url = node.metadata.get("docs_url", "unknown")
            index_id = node.metadata.get("index_id", "")
            index_origin = node.metadata.get("index_origin", "")

            if score < RAG_SIMILARITY_CUTOFF:
                logger.info(
                    "Document #%d rejected: '%s' (index: %s) - "
                    "similarity score %.4f < threshold %.4f",
                    idx + 1,
                    doc_title,
                    index_origin or index_id or "unknown",
                    score,
                    RAG_SIMILARITY_CUTOFF,
                )
                break

            node_text = format_retrieved_chunk(node.get_text())
            tokens = self.text_to_tokens(node_text)
            tokens_count = TokenHandler._get_token_count(tokens)
            tokens_count += 1  # for new-line char
            logger.debug("RAG content tokens count: %d", tokens_count)

            available_tokens = min(tokens_count, max_tokens)
            logger.debug(
                "Tokens used for this chunk: %d, remaining budget: %d",
                available_tokens,
                max_tokens - available_tokens,
            )

            if available_tokens < MINIMUM_CONTEXT_TOKEN_LIMIT:
                logger.info(
                    "Document #%d rejected: '%s' (index: %s) - "
                    "insufficient tokens (%d < %d minimum)",
                    idx + 1,
                    doc_title,
                    index_origin or index_id or "unknown",
                    available_tokens,
                    MINIMUM_CONTEXT_TOKEN_LIMIT,
                )
                break

            logger.info(
                "Document #%d selected: title='%s', url='%s', index='%s', "
                "score=%.4f, tokens=%d, remaining_context=%d",
                idx + 1,
                doc_title,
                doc_url,
                index_origin or index_id or "unknown",
                score,
                available_tokens,
                max_tokens - available_tokens,
            )

            node_text = self.tokens_to_text(tokens[:available_tokens])
            rag_chunks.append(
                RagChunk(
                    text=node_text,
                    doc_url=doc_url,
                    doc_title=doc_title,
                )
            )

            max_tokens -= available_tokens

        logger.info(
            "Final selection: %d documents chosen for RAG context", len(rag_chunks)
        )
        return rag_chunks

    def limit_conversation_history(
        self, history: list[BaseMessage], limit: int = 0
    ) -> tuple[list[BaseMessage], bool]:
        """Limit conversation history to specified number of tokens."""
        total_length = 0
        index = 0

        for message in reversed(history):
            message_length = TokenHandler._get_token_count(
                self.text_to_tokens(f"{message.type}: {message.content}")
            )
            total_length += message_length + 1  # 1 for new-line char

            # if total length of already checked messages is higher than limit
            # then skip all remaining messages (we need to skip from top)
            if total_length > limit:
                logger.debug(
                    "History truncated, it exceeds available %d tokens.", limit
                )
                return history[len(history) - index :], True
            index += 1

        return history, False

    def calculate_and_check_available_tokens(
        self,
        prompt: str,
        context_window_size: int,
        max_tokens_for_response: int,
        max_tokens_for_tools: int = 0,
    ) -> int:
        """Return tokens still available for context after the formatted prompt.

        Raises:
            PromptTooLongError: If the prompt alone exceeds the allowed budget.
        """
        prompt_tokens = TokenHandler._get_token_count(self.text_to_tokens(prompt))
        context_limit = (
            context_window_size - max_tokens_for_response - max_tokens_for_tools
        )
        if prompt_tokens > context_limit:
            raise PromptTooLongError(
                f"Prompt length {prompt_tokens} exceeds "
                f"LLM available context window limit {context_limit} tokens"
            )
        return context_limit - prompt_tokens


class TokenBudgetTracker:
    """Per-request token budget accounting across the full context window.

    Tracks every token allocation by category so that prompt construction,
    tool-calling rounds, and AI responses all draw from one unified budget.
    """

    def __init__(
        self,
        token_handler: TokenHandler,
        context_window_size: int,
        max_response_tokens: int,
        max_tool_tokens: int,
        round_cap_fraction: float,
    ) -> None:
        """Initialize the tracker with the full context window parameters.

        Args:
            token_handler: Tokenizer used for all token counting.
            context_window_size: Total context window of the LLM.
            max_response_tokens: Tokens reserved for the LLM response.
            max_tool_tokens: Tokens reserved for tool-related usage.
            round_cap_fraction: Fraction of remaining tool budget available
                per round (default DEFAULT_TOOL_ROUND_CAP_FRACTION).
        """
        self._token_handler = token_handler
        self.context_window_size = context_window_size
        self.max_response_tokens = max_response_tokens
        self.max_tool_tokens = max_tool_tokens
        self.round_cap_fraction = round_cap_fraction

        self._usage: dict[TokenCategory, int] = dict.fromkeys(TokenCategory, 0)
        self._tool_loop_max_rounds: int | None = None
        self._last_tools_exec_budget: int | None = None
        self._last_tool_rounds_left: int | None = None

    @property
    def token_handler(self) -> TokenHandler:
        """The underlying tokenizer."""
        return self._token_handler

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the underlying tokenizer."""
        return TokenHandler._get_token_count(self._token_handler.text_to_tokens(text))

    @property
    def history_budget(self) -> int:
        """Tokens available for history after prompt, RAG, and skill overhead."""
        return (
            self.prompt_budget
            - self._usage[TokenCategory.PROMPT]
            - self._usage[TokenCategory.RAG]
            - self._usage[TokenCategory.SKILL]
        )

    def charge(self, category: TokenCategory, tokens: int) -> None:
        """Add a token charge to the specified category.

        Args:
            category: Which budget category to charge.
            tokens: Number of tokens to add.
        """
        self._usage[category] += tokens

    def usage(self, category: TokenCategory) -> int:
        """Return the current token usage for a category."""
        return self._usage[category]

    @property
    def total_used(self) -> int:
        """Total tokens used across all categories."""
        return sum(self._usage.values())

    @property
    def prompt_budget(self) -> int:
        """Tokens available for prompt content (system, history, RAG, skill)."""
        return (
            self.context_window_size - self.max_response_tokens - self.max_tool_tokens
        )

    @property
    def remaining(self) -> int:
        """Tokens remaining in the full context window."""
        return self.context_window_size - self.max_response_tokens - self.total_used

    @property
    def prompt_budget_remaining(self) -> int:
        """Tokens still available within the prompt budget."""
        return self.prompt_budget - self.total_used

    @property
    def tool_budget_used(self) -> int:
        """Tokens consumed from the tool execution slice (per-round AI + results).

        Tool definition tokens are request-side input and are tracked under
        ``TokenCategory.TOOL_DEFINITIONS`` for logging, but they are excluded
        here so ``tool_budget_remaining`` reflects only execution traffic.
        """
        return (
            self._usage[TokenCategory.AI_ROUND] + self._usage[TokenCategory.TOOL_RESULT]
        )

    @property
    def tool_budget_remaining(self) -> int:
        """Tokens still available within the tool budget."""
        return max(0, self.max_tool_tokens - self.tool_budget_used)

    @property
    def tools_round_budget(self) -> int:
        """Budget available for the current tool-calling round.

        Caps at ``round_cap_fraction`` of the remaining tool budget so that
        no single round can exhaust the entire tool allocation.
        """
        return int(self.tool_budget_remaining * self.round_cap_fraction)

    def tools_round_execution_budget(self, tool_rounds_left: int) -> int:
        """Token budget for one tool execution round (adaptive cap).

        Returns the smaller of (1) ``round_cap_fraction`` of the remaining
        tool pool and (2) ``tool_budget_remaining // horizon`` with
        ``horizon = max(1, tool_rounds_left)``. Callers will pass an upper
        bound on tool rounds still to come (for example ``max_rounds -
        round_index``); this method does not read the tool loop state itself.

        Args:
            tool_rounds_left: Upper bound on tool rounds remaining including
                this round; values below 1 are treated as 1.

        Returns:
            Token budget for a single round's tool execution, or 0 if the
            tool pool is exhausted.
        """
        remaining_tool = self.tool_budget_remaining
        if remaining_tool <= 0:
            return 0
        horizon = max(1, tool_rounds_left)
        fraction_cap = int(remaining_tool * self.round_cap_fraction)
        equal_share = remaining_tool // horizon
        return min(fraction_cap, equal_share)

    def set_tool_loop_max_rounds(self, max_rounds: int) -> None:
        """Store the configured tool-loop bound for adaptive budget in :meth:`summary`."""
        self._tool_loop_max_rounds = max_rounds

    @property
    def last_tools_exec_budget(self) -> int | None:
        """Adaptive tool execution budget from the last :meth:`summary` call, if any."""
        return self._last_tools_exec_budget

    @property
    def last_tool_rounds_left(self) -> int | None:
        """Horizon ``max_rounds - round_index`` used for the last :meth:`summary` call."""
        return self._last_tool_rounds_left

    def summary(self, round_index: int) -> str:
        """Return a per-round breakdown of token usage.

        When :meth:`set_tool_loop_max_rounds` was called, appends one
        ``tools_exec_budget`` line using :meth:`tools_round_execution_budget` with
        ``tool_rounds_left = max_rounds - round_index`` and stores the values on
        ``last_tools_exec_budget`` / ``last_tool_rounds_left``.
        """
        parts = [
            f"Token budget after round {round_index}:",
            f"prompt={self._usage[TokenCategory.PROMPT]}",
            f"history={self._usage[TokenCategory.HISTORY]}",
            f"rag={self._usage[TokenCategory.RAG]}",
            f"skill={self._usage[TokenCategory.SKILL]}",
            f"tool_defs={self._usage[TokenCategory.TOOL_DEFINITIONS]}",
            f"ai_rounds={self._usage[TokenCategory.AI_ROUND]}",
            f"tool_results={self._usage[TokenCategory.TOOL_RESULT]} |",
            f"total_used={self.total_used}/{self.context_window_size}",
            f"remaining={self.remaining}",
            f"tool_remaining={self.tool_budget_remaining}/{self.max_tool_tokens}",
        ]
        if self._tool_loop_max_rounds is not None:
            tool_rounds_left = self._tool_loop_max_rounds - round_index
            exec_budget = self.tools_round_execution_budget(tool_rounds_left)
            self._last_tools_exec_budget = exec_budget
            self._last_tool_rounds_left = tool_rounds_left
            parts.append(
                f"tools_exec_budget={exec_budget} tool_rounds_left={tool_rounds_left}"
            )
        else:
            self._last_tools_exec_budget = None
            self._last_tool_rounds_left = None
        return " ".join(parts)
