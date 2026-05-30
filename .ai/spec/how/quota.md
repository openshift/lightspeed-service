# Quota Management -- Architecture

The quota system enforces per-user and per-cluster token limits via PostgreSQL-backed counters. A factory creates limiter instances from config, the request flow checks and deducts tokens, and a background scheduler thread periodically replenishes quotas.

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `src/quota/quota_limiter.py` | `QuotaLimiter` (ABC) | Contract: `available_quota`, `ensure_available_quota`, `consume_tokens`, `revoke_quota`, `increase_quota` |
| `src/quota/revokable_quota_limiter.py` | `RevokableQuotaLimiter(QuotaLimiter, PostgresBase)` | Core implementation: PostgreSQL CRUD, schema creation, budget accounting |
| `src/quota/user_quota_limiter.py` | `UserQuotaLimiter(RevokableQuotaLimiter)` | Per-user quota: `subject_type="u"`, subject_id is user_id |
| `src/quota/cluster_quota_limiter.py` | `ClusterQuotaLimiter(RevokableQuotaLimiter)` | Cluster-wide quota: `subject_type="c"`, subject_id forced to `""` |
| `src/quota/quota_limiter_factory.py` | `QuotaLimiterFactory` | Factory: creates limiters from config via `match` on type string |
| `src/quota/quota_exceed_error.py` | `QuotaExceedError` | Exception with user/cluster-specific messages |
| `src/quota/token_usage_history.py` | `TokenUsageHistory(PostgresBase)` | Analytics: upsert per-user/provider/model token counts (separate from enforcement) |
| `runners/quota_scheduler.py` | `quota_scheduler()`, `start_quota_scheduler()` | Daemon thread: periodic quota replenishment via direct SQL |

## Data Flow

### Request-time quota enforcement

```
process_request()
  -> check_tokens_available(config.quota_limiters, user_id)
       for each limiter:
         limiter.ensure_available_quota(subject_id=user_id)
           SELECT available FROM quota_limits WHERE id=user_id AND subject='u'
           if available <= 0: raise QuotaExceedError -> HTTP 500
  ... LLM processing ...
  -> consume_tokens(config.quota_limiters, config.token_usage_history, ...)
       token_usage_history.consume_tokens(user_id, provider, model, in, out)  [if enabled]
         INSERT ... ON CONFLICT DO UPDATE SET input_tokens=input_tokens+N
       for each limiter:
         limiter.consume_tokens(input_tokens, output_tokens, subject_id=user_id)
           UPDATE quota_limits SET available=available-(in+out) WHERE id=user_id AND subject='u'
  -> get_available_quotas(config.quota_limiters, user_id)
       for each limiter: limiter.available_quota(user_id) -> int
       returns {"UserQuotaLimiter": 450, "ClusterQuotaLimiter": 8950}
```

### Background quota replenishment (scheduler)

```
start_quota_scheduler(config)  [called from runner.py at startup]
  -> Thread(target=quota_scheduler, daemon=True).start()
       while True:
         for each limiter in config:
           quota_revocation(connection, name, limiter)
             if increase_by configured:
               UPDATE quota_limits SET available=available+N
                 WHERE subject='u'|'c' AND revoked_at < NOW() - INTERVAL period
             if initial_quota configured:
               UPDATE quota_limits SET available=initial_quota
                 WHERE subject='u'|'c' AND revoked_at < NOW() - INTERVAL period
         sleep(config.scheduler.period)
```

The scheduler runs direct SQL via psycopg2 (not through the `QuotaLimiter` abstraction) with `autocommit=True`. The `revoked_at < NOW() - INTERVAL period` condition ensures quotas are only replenished after their configured period has elapsed.

## Key Abstractions

### Database schema

```sql
CREATE TABLE quota_limits (
    id          text NOT NULL,       -- user_id for per-user, "" for cluster
    subject     char(1) NOT NULL,    -- 'u' for user, 'c' for cluster
    quota_limit int NOT NULL,        -- configured initial quota
    available   int,                 -- remaining tokens (decremented on use)
    updated_at  timestamp with time zone,
    revoked_at  timestamp with time zone,  -- last replenishment timestamp
    PRIMARY KEY(id, subject)
);

CREATE TABLE token_usage (
    user_id       text NOT NULL,
    provider      text NOT NULL,
    model         text NOT NULL,
    input_tokens  int,
    output_tokens int,
    updated_at    timestamp with time zone,
    PRIMARY KEY(user_id, provider, model)
);
```

### User vs cluster limiter

Both inherit from `RevokableQuotaLimiter`. The only difference is the `subject_type` constructor argument:
- `UserQuotaLimiter`: `subject_type="u"`, uses `user_id` as the database key -- each user gets independent quota
- `ClusterQuotaLimiter`: `subject_type="c"`, forces `subject_id=""` in every method -- single shared row for the entire cluster

### Lazy initialization

New quota records are created on first access. `available_quota()` calls `_init_quota()` when no database row exists for the subject, inserting a row with `available = initial_quota` and `revoked_at = NOW()`.

### Token usage history vs quota enforcement

`TokenUsageHistory` is an independent analytics system that records cumulative token consumption per `(user_id, provider, model)` using PostgreSQL `ON CONFLICT DO UPDATE` upserts. It only adds tokens, never decrements. It is controlled by `enable_token_history` in config and is completely separate from quota enforcement.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `app/endpoints/ols.py` | `QuotaLimiter` instances | `check_tokens_available()` pre-request, `consume_tokens()` post-response |
| `app/endpoints/streaming_ols.py` | Same | Same functions, called from streaming wrapper |
| `utils/config.py` | `QuotaLimiterFactory` | `config.quota_limiters` lazy property creates limiter list |
| `utils/config.py` | `TokenUsageHistory` | `config.token_usage_history` lazy property, gated on `enable_token_history` |
| `runner.py` | `runners/quota_scheduler.py` | `start_quota_scheduler()` spawns daemon thread at startup |

## Implementation Notes

### Scheduler uses direct SQL, not the limiter abstraction

The quota scheduler runs `INCREASE_QUOTA_STATEMENT` and `RESET_QUOTA_STATEMENT` directly via psycopg2, not through `RevokableQuotaLimiter` methods. This is because the scheduler operates on all rows matching a subject type at once (bulk update), while the limiter methods operate on individual subjects.

### Connection management differs between scheduler and limiters

Limiters use the `@connection` decorator from `PostgresBase` for per-call connection management. The scheduler creates a single long-lived `psycopg2.connect()` with `autocommit=True` and reuses it across all iterations.

### Quota check returns HTTP 500, not 429

When `QuotaExceedError` is raised, the endpoint catches it and returns HTTP 500 (not 429 Rate Limit). This is a known inconsistency -- the error is caught alongside database errors in a broad exception handler.

### Daemon thread lifecycle

The scheduler thread is created with `daemon=True`, so it is automatically killed when the main Uvicorn process exits. There is no graceful shutdown mechanism -- the thread simply stops between sleep cycles.
