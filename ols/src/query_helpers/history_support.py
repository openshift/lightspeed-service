"""Conversation history support helpers."""

import asyncio
import logging
import time
from typing import TypeAlias

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ols import config
from ols.app.models.models import CacheEntry
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)

HistorySplit: TypeAlias = tuple[list[CacheEntry], bool]
HISTORY_TOKEN_BUDGET_RATIO = 0.85
DEFAULT_ENTRIES_TO_KEEP = 5


def _count_message_tokens(message: BaseMessage, token_handler: TokenHandler) -> int:
    """Estimate token count for one message with newline overhead.

    Args:
        message: Chat message to estimate token usage for.
        token_handler: Token helper used for text-to-token conversion.

    Returns:
        Estimated token count for the message including newline separator overhead.
    """
    # Mirror the same type/content formatting used by history token limiting.
    message_tokens = TokenHandler._get_token_count(
        token_handler.text_to_tokens(f"{message.type}: {message.content}")
    )
    # Reserve a single token for message separator/newline in joined history text.
    return message_tokens + 1


def _split_entries_by_token_budget(
    entries: list[CacheEntry],
    available_tokens: int,
    token_handler: TokenHandler,
) -> HistorySplit:
    """Split history into fitting newest entries and overflow flag.

    Args:
        entries: Full conversation history ordered oldest to newest.
        available_tokens: Token budget available for history.
        token_handler: Token helper used for per-entry token estimation.

    Returns:
        A tuple containing:
        - Entries that fit in budget, ordered newest to oldest.
        - Boolean indicating whether older entries overflowed the budget.
    """
    if not entries:
        return [], False

    kept_newest_first: list[CacheEntry] = []
    used_tokens = 0
    # Walk from newest to oldest and keep appending until we exceed the budget.
    for entry in reversed(entries):
        # Count one cached turn (user query + assistant response) consistently.
        response_message = entry.response or AIMessage("")
        entry_tokens = _count_message_tokens(entry.query, token_handler)
        entry_tokens += _count_message_tokens(response_message, token_handler)
        if used_tokens + entry_tokens > available_tokens:
            # First non-fitting entry marks token overflow; return what still fits.
            return kept_newest_first, True
        kept_newest_first.append(entry)
        used_tokens += entry_tokens

    return kept_newest_first, False


def _rewrite_cache(
    user_id: str,
    conversation_id: str,
    skip_user_id_check: bool,
    entries: list[CacheEntry],
    context: str,
) -> list[CacheEntry]:
    """Replace conversation cache and return next history state.

    Args:
        user_id: User ID for cache operations.
        conversation_id: Conversation ID for cache operations.
        skip_user_id_check: Whether to bypass user ID validation.
        entries: Full list of entries to persist as replacement history.
        context: Human-readable context label for error logs.

    Returns:
        Persisted entries on success, or an empty list on cache update failure.
    """
    rewrite_start = time.perf_counter()
    try:
        # Replace in two phases: clear previous history first...
        config.conversation_cache.delete(user_id, conversation_id, skip_user_id_check)
        # ...then append the replacement history in order.
        for entry in entries:
            config.conversation_cache.insert_or_append(
                user_id,
                conversation_id,
                entry,
                skip_user_id_check,
            )
        return entries
    except Exception as e:
        logger.error("Failed to update cache with %s: %s", context, e)
        # Strict consistency policy: on cache rewrite failure, return empty history
        # instead of any in-memory fallback that may diverge from persisted state.
        return []
    finally:
        rewrite_duration_ms = (time.perf_counter() - rewrite_start) * 1000
        logger.info(
            "Cache rewrite (%s) finished in %.2f ms for %d entries",
            context,
            rewrite_duration_ms,
            len(entries),
        )


def _retrieve_previous_input(
    user_id: str,
    conversation_id: str,
    skip_user_id_check: bool,
) -> list[CacheEntry]:
    """Retrieve previous conversation history from cache.

    Args:
        user_id: User ID for cache lookup.
        conversation_id: Conversation ID for cache lookup.
        skip_user_id_check: Whether to bypass user ID validation.

    Returns:
        Conversation history entries, or an empty list when none exist.
    """
    previous_input: list[CacheEntry] = []
    if conversation_id:
        # Read full conversation for budgeting/compression decisions.
        cache_content = config.conversation_cache.get(
            user_id,
            conversation_id,
            skip_user_id_check,
        )
        if cache_content:
            previous_input = cache_content
            logger.info(
                "Conversation ID: %s - Retrieved %d messages",
                conversation_id,
                len(previous_input),
            )
    return previous_input


async def summarize_entries(entries: list[CacheEntry], bare_llm: object) -> str | None:
    """Summarize a list of conversation cache entries.

    Args:
        entries: Conversation entries to summarize.
        bare_llm: LLM client with callable async `ainvoke(messages)`.

    Returns:
        Summary text on success, otherwise None.
    """
    if not entries:
        return None

    conversation_text = []
    # Build plain text transcript as alternating User/Assistant turns.
    for entry in entries:
        conversation_text.append(f"User: {entry.query.content}")
        conversation_text.append(f"Assistant: {entry.response.content}")

    full_conversation = "\n".join(conversation_text)
    # Structured prompt for reusable conversational summary output.
    summarization_prompt = f"""Create a comprehensive but concise summary.
Include:
- Main topics and questions asked
- Solutions, commands, or configurations provided
- Decisions made or troubleshooting steps taken
- Important technical details (error messages, resource names, configurations)
- Any tasks or follow-up actions mentioned

Exclude:
- Greetings and pleasantries
- Repetitive information

Write in a clear style suitable for continuing the conversation later.
Keep the summary focused and compact: preserve decisions, errors, fixes, and next steps,
but avoid long narrative explanations.

Conversation history:
{full_conversation}

Summary:"""

    delay = 1.0
    max_attempts = 3
    last_exception: Exception | None = None
    # Retry transient failures with exponential backoff.
    for attempt in range(1, max_attempts + 1):
        try:
            messages = [{"role": "user", "content": summarization_prompt}]
            # Avoid Protocol/ABC dependency: runtime capability check is enough here.
            ainvoke = getattr(bare_llm, "ainvoke", None)
            if not callable(ainvoke):
                raise TypeError("LLM object must provide callable ainvoke(messages)")
            response = await ainvoke(messages)
            content = getattr(response, "content", None)
            # Normalize provider-specific response objects to plain string output.
            summary = content if isinstance(content, str) else str(response)
            logger.info(
                "Summarized %d conversation entries into %d characters",
                len(entries),
                len(summary),
            )
            return summary
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            # Retry only known transient/network/rate-limit style errors.
            is_transient = any(
                keyword in error_msg
                for keyword in [
                    "timeout",
                    "timed out",
                    "connection",
                    "rate limit",
                    "too many requests",
                    "503",
                    "502",
                    "429",
                ]
            )
            if not is_transient or attempt == max_attempts:
                break
            logger.warning(
                "Transient error on attempt %d/%d: %s. Retrying in %.1fs...",
                attempt,
                max_attempts,
                e,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2

    if last_exception is not None:
        logger.error("Failed to summarize conversation entries: %s", last_exception)
    return None


async def compress_conversation_history(
    user_id: str,
    conversation_id: str,
    skip_user_id_check: bool,
    *,
    provider: str,
    model: str,
    bare_llm: object,
    full_cache_entries: list[CacheEntry],
    kept_newest_first: list[CacheEntry],
    entries_to_keep: int = DEFAULT_ENTRIES_TO_KEEP,
) -> list[CacheEntry]:
    """Compress conversation history by summarizing old entries.

    Args:
        user_id: User ID for cache operations.
        conversation_id: Conversation ID for cache operations.
        skip_user_id_check: Whether to bypass user ID validation.
        provider: LLM provider name stored in summary metadata.
        model: LLM model name stored in summary metadata.
        bare_llm: LLM client used for summarization.
        full_cache_entries: Full history ordered oldest to newest.
        kept_newest_first: Entries that fit budget, ordered newest to oldest.
        entries_to_keep: Keep threshold; summarize all when fit entries are at or below this value.

    Returns:
        Compressed persisted history, or fallback entries when summarization fails.
    """
    keep_entries = list(reversed(kept_newest_first))
    # Keep recency only when we have enough entries to justify preserving raw turns.
    if len(keep_entries) <= entries_to_keep:
        keep_entries = keep_entries[:-1]
    else:
        keep_entries = keep_entries[-entries_to_keep:]
    summarize_count = len(full_cache_entries) - len(keep_entries)
    # Summarize everything except whichever recent entries we decided to keep.
    entries_to_summarize = full_cache_entries[:summarize_count]
    logger.info(
        "Compressing conversation with %d entries (keeping last %d unsummarized)",
        len(full_cache_entries),
        len(keep_entries),
    )

    # Measure summarization latency for observability/tuning.
    summarize_start = time.perf_counter()
    summary_text = await summarize_entries(entries_to_summarize, bare_llm)
    summarize_duration_ms = (time.perf_counter() - summarize_start) * 1000
    logger.info(
        "Summarization finished in %.2f ms for %d entries",
        summarize_duration_ms,
        len(entries_to_summarize),
    )

    if not summary_text:
        # If summarization fails, prefer kept recency; otherwise preserve at least last turn.
        fallback_entries = keep_entries or [full_cache_entries[-1]]
        logger.warning(
            "Summarization failed, falling back to %d entry(ies)",
            len(fallback_entries),
        )
        # Persist fallback view so next request sees the same history state.
        return _rewrite_cache(
            user_id,
            conversation_id,
            skip_user_id_check,
            fallback_entries,
            "fallback history",
        )

    current_time = time.time()
    # Materialize summary as synthetic cache entry to preserve conversation continuity.
    summary_entry = CacheEntry(
        query=HumanMessage(
            content="[Previous conversation summary]",
            response_metadata={"created_at": current_time},
        ),
        response=AIMessage(
            content=summary_text,
            response_metadata={
                "created_at": current_time,
                "provider": provider,
                "model": model,
            },
        ),
    )

    # Final shape: summary first, then most recent raw turns.
    compressed_entries = [summary_entry, *keep_entries]
    return _rewrite_cache(
        user_id,
        conversation_id,
        skip_user_id_check,
        compressed_entries,
        "compressed history",
    )


async def prepare_history(
    *,
    user_id: str | None,
    conversation_id: str | None,
    skip_user_id_check: bool,
    available_tokens: int,
    provider: str,
    model: str,
    bare_llm: object,
    token_handler: TokenHandler,
) -> tuple[list[BaseMessage], bool]:
    """Retrieve, optionally compress, and truncate history for prompting.

    Args:
        user_id: User ID for conversation history retrieval.
        conversation_id: Conversation ID for history retrieval.
        skip_user_id_check: Whether to bypass user ID validation for cache access.
        available_tokens: Token budget available for conversation history.
        provider: LLM provider name used for summary metadata.
        model: LLM model name used for summary metadata.
        bare_llm: LLM client used for summarizing overflow history.
        token_handler: Token helper used for history budgeting and truncation.

    Returns:
        A tuple containing:
        - Prepared conversation history as BaseMessage objects.
        - Boolean indicating whether final token-based truncation occurred.
    """
    if not (user_id and conversation_id):
        return [], False

    # Read the full persisted history first; budgeting decisions happen in memory.
    cache_entries = _retrieve_previous_input(
        user_id,
        conversation_id,
        skip_user_id_check,
    )
    # Reserve a slice of budget so we trigger compression before hitting hard limits.
    effective_history_budget = max(
        1, int(available_tokens * HISTORY_TOKEN_BUDGET_RATIO)
    )
    # Determine if history fits the effective history token budget without compression.
    kept_newest_first, overflowed = _split_entries_by_token_budget(
        cache_entries,
        effective_history_budget,
        token_handler,
    )
    if not overflowed:
        return CacheEntry.cache_entries_to_history(cache_entries), False
    logger.info("History exceeded token budget, compressing conversation")
    # Summarize overflow and rewrite cache into summary + recent entries.
    cache_entries = await compress_conversation_history(
        user_id,
        conversation_id,
        skip_user_id_check,
        provider=provider,
        model=model,
        bare_llm=bare_llm,
        full_cache_entries=cache_entries,
        kept_newest_first=kept_newest_first,
    )
    history = CacheEntry.cache_entries_to_history(cache_entries)
    return token_handler.limit_conversation_history(history, available_tokens)
