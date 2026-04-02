#!/usr/bin/env python3
"""Simulate a gateway-proxy that reloads config and starts hitting wrong endpoints."""
import time


def emit(ts, level, msg):
    print(f"ts={ts} level={level} component=gateway-proxy msg={msg}")


def main():
    # Healthy startup
    emit("2026-01-15T10:00:00.000Z", "INFO", "gateway-proxy v3.1.0 booting")
    emit("2026-01-15T10:00:00.150Z", "INFO", "resolved upstream db-prod.internal:5432")
    emit("2026-01-15T10:00:00.300Z", "INFO", "resolved upstream redis-prod.internal:6379")
    emit("2026-01-15T10:00:00.450Z", "INFO", "upstream health=OK for all backends")
    emit("2026-01-15T10:00:01.200Z", "INFO", "listening on :8080, ready to accept traffic")

    # Normal requests
    for path, ms in [("/api/users/123", 45), ("/api/orders", 89), ("/api/products", 34)]:
        emit(f"2026-01-15T10:01:{ms:02d}.000Z", "INFO", f"200 {path} ({ms}ms)")
    emit("2026-01-15T10:01:15.000Z", "INFO", "periodic probe result=healthy")

    # Root cause: config drift
    print("2026-01-15T10:05:00.000Z [WARN] Configuration file change detected")
    emit("2026-01-15T10:05:00.100Z", "INFO", "hot-reload triggered by inotify event on /config/app.yaml")
    emit("2026-01-15T10:05:00.250Z", "WARN", "new upstream db-staging.internal:5432 differs from previous db-prod.internal:5432")
    emit("2026-01-15T10:05:00.400Z", "WARN", "new upstream redis-staging.internal:6379 differs from previous redis-prod.internal:6379")
    emit("2026-01-15T10:05:00.550Z", "WARN", "environment=PRODUCTION but config references staging hosts")
    emit("2026-01-15T10:05:00.700Z", "INFO", "draining existing connection pools")
    emit("2026-01-15T10:05:00.850Z", "INFO", "attempting to establish pools to new upstreams")

    # Connection refused flood
    emit("2026-01-15T10:05:01.000Z", "ERROR", "tcp dial db-staging.internal:5432 => connection refused")
    emit("2026-01-15T10:05:01.100Z", "ERROR", "tcp dial redis-staging.internal:6379 => connection refused")
    emit("2026-01-15T10:05:01.200Z", "ERROR", "no healthy upstreams, entering degraded mode")

    for i in range(1, 51):
        s = f"{i + 1:02d}"
        emit(f"2026-01-15T10:05:{s}.000Z", "ERROR", f"502 GET /api/users/{1000+i} upstream_err=connection_refused")
        emit(f"2026-01-15T10:05:{s}.150Z", "ERROR", f"pool reconnect db-staging.internal:5432 => refused")
        emit(f"2026-01-15T10:05:{s}.300Z", "ERROR", f"pool reconnect redis-staging.internal:6379 => refused")
        emit(f"2026-01-15T10:05:{s}.450Z", "DEBUG", "backoff wait before next reconnect attempt")
        emit(f"2026-01-15T10:05:{s}.500Z", "ERROR", f"502 POST /api/orders upstream_err=connection_refused")

    for i in range(1, 51):
        s = f"{i:02d}"
        emit(f"2026-01-15T10:06:{s}.000Z", "ERROR", f"upstream db-staging.internal:5432 still refusing")
        emit(f"2026-01-15T10:06:{s}.100Z", "ERROR", f"502 GET /api/products upstream_err=connection_refused")
        emit(f"2026-01-15T10:06:{s}.200Z", "ERROR", "all queries routed to fallback=none")
        emit(f"2026-01-15T10:06:{s}.300Z", "WARN", "circuit breaker OPEN for db pool")
        emit(f"2026-01-15T10:06:{s}.400Z", "ERROR", "probe result=unhealthy reason=no_upstream")

    emit("2026-01-15T10:07:00.000Z", "CRIT", "gateway degraded for 120s, paging oncall")
    emit("2026-01-15T10:07:00.200Z", "INFO", "active config dump: db=db-staging.internal cache=redis-staging.internal env=PRODUCTION")
    emit("2026-01-15T10:07:00.500Z", "ERROR", "staging network unreachable from production VPC")

    for i in range(1, 31):
        s = f"{i:02d}"
        emit(f"2026-01-15T10:07:{s}.000Z", "ERROR", f"upstream db-staging.internal:5432 connection_refused")
        emit(f"2026-01-15T10:07:{s}.100Z", "ERROR", f"upstream redis-staging.internal:6379 connection_refused")
        print(f"2026-01-15T10:07:{s}.200Z [HTTP] 500 GET /api/health - Connection refused")

    # Keep pod running
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
