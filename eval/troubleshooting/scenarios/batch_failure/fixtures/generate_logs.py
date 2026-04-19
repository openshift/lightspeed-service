#!/usr/bin/env python3
"""Simulate an inventory sync process that fails to reach its database."""

import logging
import sys

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("inventory-sync")

DB_HOST = "prod-db"
DB_PORT = 3333
MAX_RETRIES = 4


def try_connect(attempt):
    """Simulate a single TCP connection attempt to the database."""
    log.info(
        "Opening TCP socket to %s:%d (attempt %d/%d)",
        DB_HOST,
        DB_PORT,
        attempt,
        MAX_RETRIES,
    )
    log.error("Socket connect returned errno 111 - connection refused")
    print(f"Target host: {DB_HOST}, port: {DB_PORT}")
    log.warning("Pool stats: waiting=32, active=0, idle=0")


def main():
    """Run the inventory-sync failure simulation."""
    log.info("inventory-sync-validator starting up")
    log.info("Reading database coordinates from environment")
    log.info("Resolved endpoint %s:%d via service discovery", DB_HOST, DB_PORT)

    for attempt in range(1, MAX_RETRIES + 1):
        try_connect(attempt)
        log.error("Retry %d of %d exhausted", attempt, MAX_RETRIES)

    log.critical("All connection attempts failed after %d retries", MAX_RETRIES)
    print("FATAL: Unable to connect to required database")
    log.info("Cleaning up resources before exit")
    sys.exit(1)


if __name__ == "__main__":
    main()
