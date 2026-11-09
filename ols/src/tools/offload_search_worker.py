"""Child process that runs one offloaded-content regex search under resource limits.

Usage: python -I -S offload_search_worker.py FILE MEMORY_BYTES CPU_SECONDS

The search pattern is read from stdin. The 0-based indices of matching lines
are written to stdout, one per line. The service starts this script with
``-I -S``, so it must import only the standard library.

Running the match in a separate process lets the service kill a runaway
pattern at its deadline from any thread and caps the memory a pattern can use
without risking the service process.
"""

import re
import resource
import sys

EXIT_INVALID_PATTERN = 2
EXIT_OUT_OF_MEMORY = 3


def _current_virtual_size() -> int:
    """Return this process's virtual memory size in bytes, or 0 if unknown."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _set_limit(kind: int, value: int) -> None:
    """Lower both limits of a resource to value, within the existing hard limit."""
    _, hard = resource.getrlimit(kind)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, value))


def main() -> int:
    """Search FILE for the stdin pattern and print matching line indices."""
    file_path = sys.argv[1]
    memory_budget = int(sys.argv[2])
    cpu_seconds = int(sys.argv[3])

    # The interpreter's startup address space varies by build, so the budget
    # is added on top of it rather than used as an absolute cap.
    _set_limit(resource.RLIMIT_AS, _current_virtual_size() + memory_budget)
    _set_limit(resource.RLIMIT_CPU, cpu_seconds)

    try:
        pattern = sys.stdin.read()
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            sys.stderr.write(str(e))
            return EXIT_INVALID_PATTERN
        with open(file_path, encoding="utf-8") as lines:
            for index, line in enumerate(lines):
                if compiled.search(line):
                    sys.stdout.write(f"{index}\n")
        sys.stdout.flush()
    except MemoryError:
        return EXIT_OUT_OF_MEMORY
    return 0


if __name__ == "__main__":
    sys.exit(main())
