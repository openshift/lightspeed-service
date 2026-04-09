"""Offload large tool outputs to disk with search + read retrieval."""

import logging
import os
import re
import shutil
import signal
from typing import Optional
from uuid import uuid4

from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

from ols import constants

logger = logging.getLogger(__name__)


def cleanup_offload_storage(storage_path: str) -> None:
    """Remove orphaned offload files from a previous run.

    Call this at application startup to clean up files that may have
    been left behind after a crash (where try/finally didn't execute).

    Args:
        storage_path: Directory used for offloaded files.
    """
    if not os.path.isdir(storage_path):
        return
    try:
        shutil.rmtree(storage_path)
        os.makedirs(storage_path, exist_ok=True)
        logger.info("Cleaned up offload storage directory: %s", storage_path)
    except OSError:
        logger.warning(
            "Failed to clean up offload storage directory: %s",
            storage_path,
            exc_info=True,
        )


_CHARS_PER_TOKEN_ESTIMATE = 4

_RETRIEVAL_TOOL_NAMES = frozenset(
    {"search_offloaded_content", "read_offloaded_content"}
)


class OffloadManager:
    """Manage offloading of large tool outputs to temporary files on disk.

    Each instance is scoped to a single HTTP request. Files are cleaned
    up via ``cleanup()`` in a try/finally block.
    """

    def __init__(
        self,
        storage_path: str,
        max_tokens_per_tool_output: int,
    ) -> None:
        """Initialize the offload manager.

        Args:
            storage_path: Directory to store offloaded files.
            max_tokens_per_tool_output: Token threshold above which
                tool outputs are offloaded to disk.
        """
        self._storage_path = storage_path
        self._max_tokens = max_tokens_per_tool_output
        self._allowlist: dict[str, str] = {}
        self._retrieval_tools_built = False

    @property
    def has_offloaded_content(self) -> bool:
        """Return True if any content has been offloaded."""
        return len(self._allowlist) > 0

    @property
    def retrieval_tools_registered(self) -> bool:
        """Return True if retrieval tools have been built and registered."""
        return self._retrieval_tools_built

    def mark_retrieval_tools_registered(self) -> None:
        """Mark that retrieval tools have been added to the tool loop."""
        self._retrieval_tools_built = True

    def try_offload(self, text: str, tool_name: str) -> str:
        """Offload text to disk if it exceeds the token threshold.

        Args:
            text: The full tool output text.
            tool_name: Name of the tool that produced the output.

        Returns:
            Either the original text (when under threshold, fallback, or
            skipped) or a placeholder string with retrieval instructions.
        """
        if tool_name in _RETRIEVAL_TOOL_NAMES:
            return text

        estimated_tokens = len(text) // _CHARS_PER_TOKEN_ESTIMATE
        if estimated_tokens <= self._max_tokens:
            return text

        byte_size = len(text.encode("utf-8"))
        if byte_size > constants.OFFLOAD_MAX_FILE_SIZE_BYTES:
            logger.warning(
                "Tool '%s' output exceeds 50 MB hard cap (%d bytes); "
                "falling back to truncation",
                tool_name,
                byte_size,
            )
            return text

        ref_id = str(uuid4())
        file_path = os.path.join(self._storage_path, f"{ref_id}.txt")

        try:
            os.makedirs(self._storage_path, exist_ok=True)
            fd = os.open(
                file_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.write(fd, text.encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            logger.warning(
                "Failed to write offloaded content for tool '%s'; "
                "falling back to truncation",
                tool_name,
                exc_info=True,
            )
            return text

        self._allowlist[ref_id] = file_path
        line_count = text.count("\n") + 1

        return _build_placeholder(ref_id, tool_name, line_count, byte_size)

    def cleanup(self) -> None:
        """Delete all offloaded files. Safe to call multiple times."""
        for ref_id, file_path in self._allowlist.items():
            try:
                os.unlink(file_path)
            except OSError:
                logger.warning(
                    "Failed to delete offloaded file for ref_id '%s': %s",
                    ref_id,
                    file_path,
                    exc_info=True,
                )
        self._allowlist.clear()

    def build_retrieval_tools(self) -> list[StructuredTool]:
        """Build the search and read retrieval tools.

        Returns:
            List of two StructuredTool instances.
        """
        manager = self

        class SearchArgs(BaseModel):
            """Arguments for search_offloaded_content."""

            ref_id: str = Field(description="Reference ID from the placeholder message")
            pattern: str = Field(description="Regex pattern to search for")
            context_lines: int = Field(
                default=3,
                ge=0,
                le=10,
                description="Number of context lines before and after each match",
            )

        class ReadArgs(BaseModel):
            """Arguments for read_offloaded_content."""

            ref_id: str = Field(description="Reference ID from the placeholder message")
            start_line: int = Field(description="1-based start line number")
            end_line: int = Field(description="1-based end line number (inclusive)")

        def _search_sync(**kwargs: object) -> str:
            return _search_offloaded(manager, **kwargs)

        async def _search_async(**kwargs: object) -> str:
            return _search_offloaded(manager, **kwargs)

        def _read_sync(**kwargs: object) -> str:
            return _read_offloaded(manager, **kwargs)

        async def _read_async(**kwargs: object) -> str:
            return _read_offloaded(manager, **kwargs)

        search_tool = StructuredTool(
            name="search_offloaded_content",
            description=(
                "Search offloaded tool output using a regex pattern. "
                "Returns matching lines with line numbers and context, "
                "similar to grep -n -C N. Use this to find where "
                "something is in a large tool output."
            ),
            func=_search_sync,
            coroutine=_search_async,
            args_schema=SearchArgs,
            metadata={"annotations": {"readOnlyHint": True}},
        )

        read_tool = StructuredTool(
            name="read_offloaded_content",
            description=(
                "Read a specific line range from offloaded tool output. "
                "Returns lines with line numbers. Use this after "
                "search_offloaded_content to examine a section in detail."
            ),
            func=_read_sync,
            coroutine=_read_async,
            args_schema=ReadArgs,
            metadata={"annotations": {"readOnlyHint": True}},
        )

        return [search_tool, read_tool]


def _build_placeholder(
    ref_id: str, tool_name: str, line_count: int, byte_size: int
) -> str:
    """Build a placeholder string for offloaded content."""
    return (
        f"[Offloaded: {tool_name}]\n"
        f"ref_id: {ref_id}\n"
        f"lines: {line_count} | size: {byte_size} bytes\n"
        f"\n"
        f"The full output has been saved. To find information, use:\n"
        f'  search_offloaded_content(ref_id="{ref_id}", pattern="<regex>")\n'
        f"To read a specific section found by search, use:\n"
        f'  read_offloaded_content(ref_id="{ref_id}", start_line=N, end_line=M)\n'
    )


def _load_lines_or_error(
    manager: OffloadManager, ref_id: str
) -> tuple[Optional[list[str]], Optional[str]]:
    """Validate ref_id and load lines, returning (lines, None) or (None, error_message)."""
    if ref_id not in manager._allowlist:
        available = list(manager._allowlist.keys())
        return None, (
            f"Error: unknown reference '{ref_id}'. "
            f"Available references: {available}"
        )
    file_path = manager._allowlist[ref_id]
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.readlines(), None
    except OSError as e:
        return None, f"Error: could not read offloaded content for '{ref_id}': {e}"


def _find_matches_with_timeout(
    lines: list[str], compiled: re.Pattern[str], pattern: str
) -> tuple[Optional[list[int]], Optional[str]]:
    """Run regex search with SIGALRM timeout. Return (indices, None) or (None, error)."""

    def _timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError()

    match_indices: list[int] = []
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(constants.OFFLOAD_REGEX_TIMEOUT_SECONDS)
    try:
        for i, line in enumerate(lines):
            if compiled.search(line):
                match_indices.append(i)
    except TimeoutError:
        return None, (
            f"Error: search pattern '{pattern}' timed out. Try a simpler pattern."
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return match_indices, None


def _format_search_results(
    lines: list[str],
    match_indices: list[int],
    ref_id: str,
    pattern: str,
    context_lines: int,
) -> str:
    """Format matched lines with context into grep-style output."""
    total_matches = len(match_indices)
    capped = match_indices[: constants.OFFLOAD_MAX_SEARCH_MATCHES]
    shown = len(capped)

    included: set[int] = set()
    for idx in capped:
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        included.update(range(start, end))

    match_set = set(match_indices)
    result_parts: list[str] = [
        f"Showing {shown} of {total_matches} total matches "
        f"for '{pattern}' in {ref_id}",
        "",
    ]

    sorted_included = sorted(included)
    prev_line_idx = -2
    for line_idx in sorted_included:
        if line_idx != prev_line_idx + 1 and prev_line_idx >= 0:
            result_parts.append("--")
        line_text = lines[line_idx].rstrip("\n\r")
        marker = ":" if line_idx in match_set else "-"
        result_parts.append(f"{line_idx + 1}{marker}{line_text}")
        prev_line_idx = line_idx

    return "\n".join(result_parts)


def _search_offloaded(manager: OffloadManager, **kwargs: object) -> str:
    """Execute a regex search against an offloaded file."""
    ref_id = str(kwargs.get("ref_id", ""))
    pattern = str(kwargs.get("pattern", ""))
    context_lines = int(str(kwargs.get("context_lines", 3)))

    lines, error = _load_lines_or_error(manager, ref_id)
    if error is not None or lines is None:
        return error or f"Error: unknown reference '{ref_id}'."

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid search pattern '{pattern}': {e}"

    match_indices, timeout_error = _find_matches_with_timeout(lines, compiled, pattern)
    if timeout_error is not None or match_indices is None:
        return timeout_error or f"Error: search failed for '{ref_id}'."

    if len(match_indices) == 0:
        return (
            f"No matches found for pattern '{pattern}' in "
            f"{ref_id} ({len(lines)} lines)"
        )

    return _format_search_results(lines, match_indices, ref_id, pattern, context_lines)


def _read_offloaded(manager: OffloadManager, **kwargs: object) -> str:
    """Read a line range from an offloaded file."""
    ref_id = str(kwargs.get("ref_id", ""))
    start_line = int(str(kwargs.get("start_line", 1)))
    end_line = int(str(kwargs.get("end_line", 1)))

    lines, error = _load_lines_or_error(manager, ref_id)
    if error is not None or lines is None:
        return error or f"Error: unknown reference '{ref_id}'."

    total_lines = len(lines)
    start_idx = max(0, start_line - 1)
    end_idx = min(total_lines, end_line)

    if end_idx - start_idx > constants.OFFLOAD_MAX_READ_LINES:
        end_idx = start_idx + constants.OFFLOAD_MAX_READ_LINES

    result_parts: list[str] = []
    for i in range(start_idx, end_idx):
        line_text = lines[i].rstrip("\n\r")
        result_parts.append(f"{i + 1}:{line_text}")

    return "\n".join(result_parts)
