#!/usr/bin/env python3
"""Generate report-generator logs with a failure window between 03:00-03:05."""

import random
import sys
import time
from datetime import datetime, timedelta

TICK = timedelta(seconds=10)
FAILURE_START_HOUR = 3
FAILURE_START_MIN = 0
FAILURE_END_MIN = 5
API_URL = "https://reports-upstream.internal/v2/ingest"


def ts(dt):
    """Format a datetime as an ISO 8601 timestamp with millisecond precision."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def in_failure_window(dt):
    """Return True if dt falls within the 03:00-03:05 failure window."""
    return (
        dt.hour == FAILURE_START_HOUR
        and FAILURE_START_MIN <= dt.minute <= FAILURE_END_MIN
    )


def emit(dt, level, msg, **extra):
    """Print a structured log line for the report-generator component."""
    parts = [
        f"ts={ts(dt)}",
        f"level={level}",
        "component=report-generator",
        f"msg={msg}",
    ]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts))


def run():
    """Generate report-generator logs spanning 24 hours with a failure window."""
    now = datetime.utcnow()
    two_days_ago = now - timedelta(days=2)
    cursor = two_days_ago - timedelta(hours=24)
    iteration = 0
    warning_emitted = False

    while cursor < two_days_ago:
        if in_failure_window(cursor):
            emit(
                cursor,
                "ERROR",
                "Report generation failed",
                reason="upstream unreachable",
                url=API_URL,
                err="timeout after 5000ms",
            )

            if not warning_emitted:
                warning_emitted = True
                print(
                    f"ts={ts(cursor)} level=WARN component=report-generator "
                    f"msg=Detected repeated failures during 03:00-03:05 window "
                    f"hint=possible scheduled maintenance upstream"
                )
        else:
            duration = random.randint(80, 420)  # noqa: S311
            emit(
                cursor,
                "INFO",
                f"Report batch completed in {duration}ms",
                rows=random.randint(50, 9000),  # noqa: S311
            )

        if iteration % 90 == 0:
            emit(
                cursor,
                "INFO",
                "System health check passed",
                latency_ms=random.randint(1, 12),  # noqa: S311
            )

        cursor += TICK
        if random.random() < 0.08:  # noqa: S311
            cursor += timedelta(seconds=random.randint(1, 4))  # noqa: S311
        iteration += 1

    # sentinel so setup.sh knows logs are fully written
    cursor += timedelta(seconds=random.randint(1, 5))  # noqa: S311
    emit(cursor, "INFO", "Job executed successfully in 167ms.", final="true")
    sys.stdout.flush()

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    run()
