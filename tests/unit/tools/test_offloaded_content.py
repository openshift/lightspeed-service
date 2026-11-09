"""Unit tests for the offloaded_content module."""

import asyncio
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest

from ols import constants
from ols.src.tools import offload_search_worker
from ols.src.tools.offloaded_content import (
    OffloadManager,
    _build_placeholder,
    _find_matches_in_subprocess,
    _read_offloaded,
    _search_offloaded,
    cleanup_offload_storage,
)


@pytest.fixture
def storage_path(tmp_path):
    """Return a temporary directory for offloaded files."""
    return str(tmp_path / "offload")


@pytest.fixture
def manager(storage_path):
    """Return an OffloadManager."""
    return OffloadManager(storage_path=storage_path)


_SMALL_BUDGET = 100
_LARGE_BUDGET = 100_000


def _large_text(lines: int = 500) -> str:
    """Generate multi-line text exceeding a small token budget."""
    return "\n".join(f"line {i}: content here" for i in range(1, lines + 1))


class TestTryOffload:
    """Tests for OffloadManager.try_offload."""

    def test_under_budget_returns_original(self, manager):
        """Test that small outputs pass through unchanged."""
        small = "hello world"
        result = manager.try_offload(small, "my_tool", _LARGE_BUDGET)
        assert result is small
        assert not manager.has_offloaded_content

    def test_over_budget_writes_file_and_returns_placeholder(self, manager):
        """Test that large outputs are written to disk with a placeholder returned."""
        text = _large_text(500)
        result = manager.try_offload(text, "big_tool", _SMALL_BUDGET)

        assert result is not text
        assert "ref_id:" in result
        assert "big_tool" in result
        assert "search_offloaded_content" in result
        assert "read_offloaded_content" in result
        assert manager.has_offloaded_content

        ref_id = result.split("ref_id: ")[1].split("\n")[0]
        file_path = manager._allowlist[ref_id]
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            assert f.read() == text

    def test_retrieval_tool_output_never_offloaded(self, manager):
        """Test that retrieval tool outputs are never offloaded."""
        text = _large_text(500)
        result = manager.try_offload(text, "search_offloaded_content", _SMALL_BUDGET)
        assert result is text

        result = manager.try_offload(text, "read_offloaded_content", _SMALL_BUDGET)
        assert result is text

    def test_over_50mb_falls_back_to_original(self, manager):
        """Test that content exceeding 50 MB returns original text."""
        text = "x" * (50 * 1024 * 1024 + 1)
        result = manager.try_offload(text, "huge_tool", _SMALL_BUDGET)
        assert result is text
        assert not manager.has_offloaded_content

    def test_disk_write_failure_falls_back(self, manager):
        """Test that disk write failure returns original text."""
        text = _large_text(500)
        with patch("os.open", side_effect=OSError("disk full")):
            result = manager.try_offload(text, "fail_tool", _SMALL_BUDGET)
        assert result is text
        assert not manager.has_offloaded_content

    def test_multiple_offloads_create_separate_files(self, manager):
        """Test that multiple offloads produce separate ref_ids and files."""
        text1 = _large_text(500)
        text2 = _large_text(600)
        r1 = manager.try_offload(text1, "tool_a", _SMALL_BUDGET)
        r2 = manager.try_offload(text2, "tool_b", _SMALL_BUDGET)

        assert r1 != r2
        assert len(manager._allowlist) == 2

    def test_session_dir_created_on_first_offload(self, manager):
        """Test that the session directory is created only on first offload."""
        assert manager._session_dir is None
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        assert manager._session_dir is not None
        assert os.path.isdir(manager._session_dir)


class TestCleanup:
    """Tests for OffloadManager.cleanup."""

    def test_cleanup_removes_session_dir(self, manager):
        """Test that cleanup removes the entire session directory."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        session_dir = manager._session_dir
        assert session_dir is not None
        assert os.path.isdir(session_dir)

        manager.cleanup()
        assert not os.path.isdir(session_dir)
        assert not manager.has_offloaded_content

    def test_cleanup_idempotent(self, manager):
        """Test that cleanup tolerates already-deleted session dir."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        manager.cleanup()
        manager.cleanup()

    def test_cleanup_tolerates_missing_dir(self, manager):
        """Test that cleanup handles session dir deleted externally."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        shutil.rmtree(manager._session_dir)
        manager.cleanup()


class TestPlaceholder:
    """Tests for placeholder message content."""

    def test_placeholder_contains_required_fields(self):
        """Test placeholder includes ref_id, tool name, line count, size, instructions."""
        placeholder = _build_placeholder("ref-123", "my_tool", 42, 1234)
        assert "ref-123" in placeholder
        assert "my_tool" in placeholder
        assert "42" in placeholder
        assert "1234" in placeholder
        assert "search_offloaded_content" in placeholder
        assert "read_offloaded_content" in placeholder

    def test_placeholder_is_small(self):
        """Test placeholder is under 100 tokens (~400 chars)."""
        placeholder = _build_placeholder("ref-uuid", "tool", 100, 5000)
        estimated_tokens = len(placeholder) // 4
        assert estimated_tokens < 100


class TestSearchOffloaded:
    """Tests for the search retrieval tool."""

    def test_search_finds_matches_with_context(self, manager):
        """Test search returns matching lines with context."""
        text = _large_text(100)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        result = _search_offloaded(
            manager, ref_id=ref_id, pattern="line 50:", context_lines=2
        )
        assert "line 50:" in result
        assert "1 of 1 total matches" in result
        assert "48:" in result or "49:" in result

    def test_search_multiple_matches(self, manager):
        """Test search with a pattern matching multiple lines."""
        text = _large_text(100)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        result = _search_offloaded(
            manager, ref_id=ref_id, pattern="content here", context_lines=0
        )
        assert "50 of 100 total matches" in result

    def test_search_no_matches(self, manager):
        """Test search with no matches returns informative message."""
        text = _large_text(100)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        result = _search_offloaded(
            manager, ref_id=ref_id, pattern="nonexistent_xyz", context_lines=0
        )
        assert "No matches found" in result

    def test_search_invalid_ref_id(self, manager):
        """Test search with unknown ref_id returns error."""
        result = _search_offloaded(
            manager, ref_id="bad-ref", pattern="test", context_lines=0
        )
        assert "Error: unknown reference 'bad-ref'" in result

    def test_search_invalid_regex(self, manager):
        """Test search with invalid regex returns error."""
        text = _large_text(100)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        result = _search_offloaded(
            manager, ref_id=ref_id, pattern="[invalid", context_lines=0
        )
        assert "Error: invalid search pattern" in result

    def test_search_capped_at_max_matches(self, manager):
        """Test search caps results at OFFLOAD_MAX_SEARCH_MATCHES."""
        lines = "\n".join(f"match line {i}" for i in range(200))
        manager.try_offload(lines, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        result = _search_offloaded(
            manager, ref_id=ref_id, pattern="match line", context_lines=0
        )
        assert "50 of 200 total matches" in result

    def test_search_context_windows_merged(self, storage_path):
        """Test that overlapping context windows are merged (no duplicate lines)."""
        mgr = OffloadManager(storage_path=storage_path)
        text = "\n".join(f"line {i}" for i in range(20))
        mgr.try_offload(text, "tool", 1)
        ref_id = next(iter(mgr._allowlist.keys()))

        result = _search_offloaded(
            mgr, ref_id=ref_id, pattern="line [89]", context_lines=3
        )
        line_numbers = re.findall(r"^(\d+)[:-]", result, re.MULTILINE)
        assert len(line_numbers) == len(set(line_numbers))

    def test_search_regex_timeout(self, manager):
        """Test search returns timeout error when regex execution exceeds time limit."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        with patch(
            "ols.src.tools.offloaded_content._find_matches_in_subprocess",
            return_value=(
                None,
                "Error: search pattern 'bad' timed out. Try a simpler pattern.",
            ),
        ):
            result = _search_offloaded(
                manager, ref_id=ref_id, pattern="bad", context_lines=0
            )
        assert "timed out" in result
        assert "simpler pattern" in result

    def test_search_offloaded_succeeds_in_worker_thread(self, manager):
        """Test search succeeds when the request runs outside the main thread."""
        text = _large_text(100)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))
        outcome: dict[str, object] = {}

        async def call() -> None:
            outcome["result"] = _search_offloaded(
                manager, ref_id=ref_id, pattern="line 42", context_lines=0
            )

        def run_in_worker() -> None:
            asyncio.run(call())

        thread = threading.Thread(target=run_in_worker)
        thread.start()
        thread.join(timeout=30)
        assert not thread.is_alive(), "search thread did not finish"

        assert "line 42" in str(outcome["result"])


class TestReadOffloaded:
    """Tests for the read retrieval tool."""

    def test_read_returns_line_range(self, manager):
        """Test read returns the correct line range with line numbers."""
        text = _large_text(100)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        result = _read_offloaded(manager, ref_id=ref_id, start_line=10, end_line=15)
        lines = result.strip().split("\n")
        assert len(lines) == 6
        assert lines[0].startswith("10:")
        assert lines[-1].startswith("15:")

    def test_read_clamps_out_of_bounds(self, storage_path):
        """Test read clamps line range to file boundaries."""
        mgr = OffloadManager(storage_path=storage_path)
        text = _large_text(10)
        mgr.try_offload(text, "tool", 1)
        ref_id = next(iter(mgr._allowlist.keys()))

        result = _read_offloaded(mgr, ref_id=ref_id, start_line=1, end_line=999)
        lines = result.strip().split("\n")
        assert len(lines) == 10

    def test_read_caps_at_max_lines(self, manager):
        """Test read caps at OFFLOAD_MAX_READ_LINES."""
        text = _large_text(1000)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        result = _read_offloaded(manager, ref_id=ref_id, start_line=1, end_line=1000)
        lines = result.strip().split("\n")
        assert len(lines) == 500

    def test_read_invalid_ref_id(self, manager):
        """Test read with unknown ref_id returns error."""
        result = _read_offloaded(manager, ref_id="bad-ref", start_line=1, end_line=10)
        assert "Error: unknown reference 'bad-ref'" in result


class TestBuildRetrievalTools:
    """Tests for OffloadManager.build_retrieval_tools."""

    def test_returns_two_tools(self, manager):
        """Test build_retrieval_tools returns search and read tools."""
        tools = manager.build_retrieval_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"search_offloaded_content", "read_offloaded_content"}

    def test_tools_have_readonly_annotation(self, manager):
        """Test both tools have readOnlyHint annotation."""
        tools = manager.build_retrieval_tools()
        for tool in tools:
            assert tool.metadata["annotations"]["readOnlyHint"] is True

    @pytest.mark.asyncio
    async def test_search_tool_executes(self, manager):
        """Test search tool can be called via coroutine."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        tools = manager.build_retrieval_tools()
        search_tool = next(t for t in tools if t.name == "search_offloaded_content")
        result = await search_tool.coroutine(
            ref_id=ref_id, pattern="line 100:", context_lines=1
        )
        assert "line 100:" in result

    @pytest.mark.asyncio
    async def test_read_tool_executes(self, manager):
        """Test read tool can be called via coroutine."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))

        tools = manager.build_retrieval_tools()
        read_tool = next(t for t in tools if t.name == "read_offloaded_content")
        result = await read_tool.coroutine(ref_id=ref_id, start_line=1, end_line=5)
        assert "1:" in result
        assert "5:" in result


class TestSearchThenRead:
    """Tests for AC 4: model can call retrieval tools multiple times."""

    @pytest.mark.asyncio
    async def test_search_then_read_same_ref(self, manager):
        """Test search then read on the same ref_id both return correct results."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))
        tools = manager.build_retrieval_tools()
        search_tool = next(t for t in tools if t.name == "search_offloaded_content")
        read_tool = next(t for t in tools if t.name == "read_offloaded_content")

        search_result = await search_tool.coroutine(
            ref_id=ref_id, pattern="line 250:", context_lines=0
        )
        assert "line 250:" in search_result

        read_result = await read_tool.coroutine(
            ref_id=ref_id, start_line=248, end_line=252
        )
        assert "248:" in read_result
        assert "252:" in read_result

    @pytest.mark.asyncio
    async def test_multiple_searches_different_patterns(self, manager):
        """Test multiple search calls with different patterns on same ref."""
        text = _large_text(500)
        manager.try_offload(text, "tool", _SMALL_BUDGET)
        ref_id = next(iter(manager._allowlist.keys()))
        tools = manager.build_retrieval_tools()
        search_tool = next(t for t in tools if t.name == "search_offloaded_content")

        r1 = await search_tool.coroutine(
            ref_id=ref_id, pattern="line 100:", context_lines=0
        )
        r2 = await search_tool.coroutine(
            ref_id=ref_id, pattern="line 200:", context_lines=0
        )
        assert "line 100:" in r1
        assert "line 200:" in r2


class TestFindMatchesInSubprocess:
    """Tests for _find_matches_in_subprocess."""

    @pytest.fixture
    def file_path(self, tmp_path):
        """Return a small file to search."""
        path = tmp_path / "content.txt"
        path.write_text("foo\nbar\nfoo\n", encoding="utf-8")
        return str(path)

    def test_returns_matching_line_indices(self, file_path):
        """Test matching lines are returned as 0-based indices."""
        assert _find_matches_in_subprocess(file_path, "ref", "fo+") == ([0, 2], None)

    def test_no_match_returns_empty_list(self, file_path):
        """Test a pattern with no matches returns an empty list."""
        assert _find_matches_in_subprocess(file_path, "ref", "baz") == ([], None)

    def test_invalid_pattern_reports_re_error(self, file_path):
        """Test an invalid pattern returns the re compile error."""
        indices, error = _find_matches_in_subprocess(file_path, "ref", "[x")
        assert indices is None
        assert error is not None
        assert error.startswith("Error: invalid search pattern '[x': ")
        assert "unterminated character set" in error

    def test_timeout_stops_slow_pattern_off_main_thread(self, tmp_path):
        """Test a slow pattern is stopped at the deadline outside the main thread."""
        path = tmp_path / "slow.txt"
        path.write_text(("a" * 40 + "b\n") * 3, encoding="utf-8")
        outcome: dict[str, object] = {}

        def run_search() -> None:
            started = time.monotonic()
            outcome["value"] = _find_matches_in_subprocess(
                str(path), "ref", r"(a|aa)+$"
            )
            outcome["elapsed"] = time.monotonic() - started

        with patch.object(constants, "OFFLOAD_REGEX_TIMEOUT_SECONDS", 0.2):
            thread = threading.Thread(target=run_search)
            thread.start()
            thread.join(timeout=30)
            assert not thread.is_alive(), "search thread did not finish"

        assert outcome["value"] == (
            None,
            "Error: search pattern '(a|aa)+$' timed out. Try a simpler pattern.",
        )
        assert outcome["elapsed"] < 5  # type: ignore[operator]

    def test_memory_over_limit_returns_error(self, file_path):
        """Test a search exceeding the memory budget fails without harming the caller."""
        pattern = "a" * (64 * 1024 * 1024)
        with patch.object(
            constants, "OFFLOAD_SEARCH_MEMORY_LIMIT_BYTES", 8 * 1024 * 1024
        ):
            indices, error = _find_matches_in_subprocess(file_path, "ref", pattern)

        assert indices is None
        assert error is not None
        assert error.endswith("needs too much memory. Try a simpler pattern.")

    def test_child_launch_failure_returns_error(self, file_path):
        """Test a child process that cannot start returns a tool error."""
        with patch(
            "ols.src.tools.offloaded_content.subprocess.run",
            side_effect=OSError("cannot start"),
        ):
            indices, error = _find_matches_in_subprocess(file_path, "ref", "foo")

        assert indices is None
        assert error == "Error: search failed for 'ref'."

    def test_unexpected_child_exit_returns_error(self, file_path):
        """Test an unrecognized child exit code returns a tool error."""
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom"
        )
        with patch(
            "ols.src.tools.offloaded_content.subprocess.run",
            return_value=failed,
        ):
            indices, error = _find_matches_in_subprocess(file_path, "ref", "foo")

        assert indices is None
        assert error == "Error: search failed for 'ref'."


class TestOffloadSearchWorker:
    """Tests for the stdlib-only child process entrypoint."""

    @pytest.fixture
    def content_file(self, tmp_path):
        """Return a small file to search."""
        path = tmp_path / "content.txt"
        path.write_text("foo\nbar\nfoo\n", encoding="utf-8")
        return str(path)

    def _run_main(self, content_file, stdin):
        """Run the worker entrypoint with resource limits and stdio patched."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(offload_search_worker, "_set_limit"),
            patch.object(sys, "argv", ["worker", content_file, "8000000", "6"]),
            patch.object(sys, "stdin", io.StringIO(stdin)),
            patch.object(sys, "stdout", stdout),
            patch.object(sys, "stderr", stderr),
        ):
            return_code = offload_search_worker.main()
        return return_code, stdout.getvalue(), stderr.getvalue()

    def test_main_prints_matching_line_indices(self, content_file):
        """Test the entrypoint prints 0-based indices of matching lines."""
        assert self._run_main(content_file, "fo+") == (0, "0\n2\n", "")

    def test_main_prints_nothing_without_matches(self, content_file):
        """Test the entrypoint prints no indices when nothing matches."""
        assert self._run_main(content_file, "baz") == (0, "", "")

    def test_main_reports_invalid_pattern(self, content_file):
        """Test the entrypoint reports invalid patterns on stderr."""
        return_code, stdout, stderr = self._run_main(content_file, "[x")
        assert return_code == offload_search_worker.EXIT_INVALID_PATTERN
        assert stdout == ""
        assert "unterminated character set" in stderr

    def test_main_reports_memory_error(self, content_file):
        """Test the entrypoint reports a memory error through its exit code."""
        with patch.object(offload_search_worker.re, "compile", side_effect=MemoryError):
            return_code, stdout, stderr = self._run_main(content_file, "foo")
        assert return_code == offload_search_worker.EXIT_OUT_OF_MEMORY
        assert (stdout, stderr) == ("", "")

    @pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/status")
    def test_current_virtual_size_reads_proc_status(self):
        """Test the child reports a positive virtual size on Linux."""
        assert offload_search_worker._current_virtual_size() > 0

    def test_current_virtual_size_returns_zero_without_proc(self):
        """Test the child falls back to zero when /proc is unavailable."""
        with patch("builtins.open", side_effect=OSError):
            assert offload_search_worker._current_virtual_size() == 0

    def test_set_limit_clamps_to_hard_limit(self):
        """Test a budget above the hard limit is clamped to the hard limit."""
        rlimit = offload_search_worker.resource
        with (
            patch.object(rlimit, "getrlimit", return_value=(0, 2048)),
            patch.object(rlimit, "setrlimit") as setrlimit,
        ):
            offload_search_worker._set_limit(rlimit.RLIMIT_AS, 4096)
        setrlimit.assert_called_once_with(rlimit.RLIMIT_AS, (2048, 2048))

    def test_set_limit_keeps_request_without_hard_limit(self):
        """Test the requested budget is kept when the hard limit is unbounded."""
        rlimit = offload_search_worker.resource
        with (
            patch.object(rlimit, "getrlimit", return_value=(0, rlimit.RLIM_INFINITY)),
            patch.object(rlimit, "setrlimit") as setrlimit,
        ):
            offload_search_worker._set_limit(rlimit.RLIMIT_CPU, 6)
        setrlimit.assert_called_once_with(rlimit.RLIMIT_CPU, (6, 6))


class TestRetrievalToolRegistration:
    """Tests for retrieval tool registration lifecycle."""

    def test_retrieval_tools_registered_flag(self, manager):
        """Test mark_retrieval_tools_registered sets the flag."""
        assert not manager.retrieval_tools_registered
        manager.mark_retrieval_tools_registered()
        assert manager.retrieval_tools_registered


class TestCleanupOffloadStorage:
    """Tests for the startup cleanup function."""

    def test_cleanup_removes_orphaned_files(self, tmp_path):
        """Test that cleanup removes files from a previous run."""
        storage = str(tmp_path / "offload")
        os.makedirs(storage)
        orphan = os.path.join(storage, "orphan.txt")
        with open(orphan, "w") as f:
            f.write("stale data")

        cleanup_offload_storage(storage)

        assert os.path.isdir(storage)
        assert os.listdir(storage) == []

    def test_cleanup_nonexistent_directory(self, tmp_path):
        """Test cleanup is a no-op when directory doesn't exist."""
        storage = str(tmp_path / "nonexistent")
        cleanup_offload_storage(storage)
        assert not os.path.isdir(storage)

    def test_cleanup_tolerates_permission_error(self, tmp_path):
        """Test cleanup logs warning on permission error."""
        storage = str(tmp_path / "offload")
        os.makedirs(storage)
        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            cleanup_offload_storage(storage)
