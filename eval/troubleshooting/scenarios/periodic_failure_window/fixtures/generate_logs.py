#!/usr/bin/env python3
"""Generate batch-processor logs showing a 03:00-03:05 failure window."""
import csv
import io
import random
import sys
import time
from datetime import datetime, timedelta

INTERVAL = timedelta(minutes=2)
FIELDS = ["timestamp", "severity", "source", "event", "detail"]
UPSTREAM = "https://pipeline-intake.corp.net/submit"


def write_row(writer, dt, severity, event, detail=""):
    writer.writerow([dt.isoformat() + "Z", severity, "batch-processor", event, detail])


def generate():
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)

    origin = datetime.utcnow()
    cursor = origin - timedelta(hours=24)
    seq = 0
    diag_done = False

    while cursor < origin:
        h, m = cursor.hour, cursor.minute
        is_bad = h == 3 and 0 <= m <= 5

        if is_bad:
            write_row(writer, cursor, "ERROR",
                      "Batch submission failed",
                      f"POST {UPSTREAM} => connect timeout 5s")

            if not diag_done:
                diag_done = True
                # This exact message is checked by setup/verify scripts
                write_row(writer, cursor, "WARN",
                          "Detected repeated failures during 03:00-03:05 window",
                          "recurring pattern – likely upstream maintenance")
        else:
            ms = random.randint(90, 480)
            write_row(writer, cursor, "INFO",
                      f"Batch processed in {ms}ms",
                      f"records={random.randint(100, 5000)}")

        if seq % 18 == 0:
            write_row(writer, cursor, "INFO",
                      "System health check passed",
                      f"ping={random.randint(1, 15)}ms")

        cursor += INTERVAL
        if random.random() < 0.12:
            cursor += timedelta(seconds=random.randint(1, 6))
        seq += 1

    # sentinel line
    cursor += timedelta(seconds=random.randint(1, 5))
    write_row(writer, cursor, "INFO",
              "Job executed successfully in 167ms.",
              "shutdown=graceful")

    sys.stdout.write(buf.getvalue())
    sys.stdout.flush()

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    generate()
