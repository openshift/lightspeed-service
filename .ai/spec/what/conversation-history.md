# Conversation History

The conversation history subsystem preserves prior exchanges within a conversation so that multi-turn interactions benefit from earlier context, and exposes CRUD operations for managing conversations per user.

## Behavioral Rules

1. Every cache operation requires both a user_id and a conversation_id. Both must be valid UUIDs. Operations with an invalid ID must be rejected with a validation error before reaching the storage backend.

2. When a query-response exchange is stored, the system must either append it to an existing conversation (if one exists for the given user_id + conversation_id) or create a new conversation. Each stored exchange contains the user query (HumanMessage), the AI response (AIMessage), any attachments submitted with the query, any tool calls requested by the model, and any tool results returned from tool execution.

3. [PLANNED: OLS-3442] The AI response stored in the cache must include both text and reasoning content accumulated during streaming. Reasoning chunks (`StreamChunkType.REASONING`) must be accumulated into the response string alongside text chunks, so the model has access to its own reasoning within the current conversation turn. The response is stored as a plain-string `AIMessage` — no structured reasoning blocks or provider-specific signatures are preserved. Cache schema changes for structured reasoning storage are deferred until testing and evals show they improve multi-turn quality.

3. Retrieving history for a user_id + conversation_id must return all stored exchanges ordered oldest to newest. If no conversation exists for the given key, the system must return an empty result.

4. Listing conversations for a user must return metadata for every conversation belonging to that user, ordered by last message timestamp (most recent first). Each entry includes: conversation_id, topic_summary, last_message_timestamp, and message_count.

5. Deleting a conversation must remove both the message history and the conversation metadata. The operation must return whether the conversation existed prior to deletion.

6. The topic summary for a conversation may be set or replaced at any time via the update operation. Setting a topic summary on a conversation that has no metadata record yet must create one.

7. The readiness check must verify that the storage backend is operational. For in-memory storage, readiness is always true. For PostgreSQL, readiness requires a live, responsive database connection.

8. When capacity is exceeded after storing an exchange, the system must evict the oldest message from the least-recently-used conversation. If that conversation has no remaining messages after eviction, the conversation and its metadata must be removed entirely. Eviction continues until total entries are within capacity.

9. When history compression is enabled and conversation history exceeds the available token budget, the system must attempt LLM-based summarization of the oldest entries. A configurable number of the most recent entries are preserved verbatim and never summarized.

10. If LLM summarization succeeds, the system must replace the summarized entries in the cache with a single synthetic summary entry (a HumanMessage containing "[Previous conversation summary]" paired with an AIMessage containing the summary text), followed by the preserved recent entries.

11. If LLM summarization fails after retries, the system must fall back to simple truncation: use only the entries that fit within the token budget, discarding the oldest entries without any summary.

12. If the cache rewrite after summarization fails, the compressed entries must still be used for the current request even though they are not persisted.

13. When compression occurs during a streaming response, the system must emit a history_compression_start event before summarization begins and a history_compression_end event (with duration) after it completes.

14. Summarization retries must use exponential backoff and must only retry on transient errors (timeouts, connection failures, rate limits). Non-transient errors must not be retried.

15. The token budget allocated for history must include a safety margin (reduced from the raw available tokens) to prevent boundary overflows.

16. Messages must be serialized to JSON for PostgreSQL storage using a custom encoder/decoder that preserves message type (human/ai), content, response metadata, and additional keyword arguments.

17. Concurrent access to the same conversation must be serialized. For PostgreSQL, this must use database-level advisory locks scoped to the user_id + conversation_id, combined with a thread-level mutex. For in-memory storage, a thread-level mutex must serialize all mutations.

18. The in-memory cache must be a singleton: all threads within a process share one cache instance.

### PostgreSQL Resilience [NEW: OLS-3221]

19. Cache operations must distinguish between *connection errors* (broken TCP, connection refused, closed connection) and *operational errors* (SQL failures on a live connection such as constraint violations, disk full, or query syntax errors). Connection errors trigger the reconnection path via the `@connection` decorator. Operational errors are wrapped in `CacheError` and propagated immediately — reconnection would not resolve them.

20. When a PostgreSQL cache operation fails due to a connection error, the `@connection` decorator must attempt to re-establish the connection transparently before retrying the operation. The decorator must detect closed or broken connections (via `psycopg2` connection status checks) and call `connect()` to obtain a fresh connection. If reconnection itself fails, the operation must raise a `CacheError` with the original cause preserved. Connection re-establishment must be idempotent and thread-safe; the existing `_tx_lock` mutex serializes reconnection attempts.

21. All PostgreSQL operations (queries, advisory lock acquisitions) must have a statement-level timeout (`statement_timeout`). If an operation exceeds the timeout, it must be cancelled, the transaction rolled back, and a `CacheError` raised. This prevents indefinite blocking when PostgreSQL is degraded (e.g., hung transactions, slow I/O). The Python-level `_tx_lock` must use a bounded acquisition timeout. If the lock cannot be acquired within the timeout, the operation must raise a `CacheError` rather than blocking indefinitely.

22. When the PostgreSQL cache backend is configured, the system must run a background health-check loop that periodically verifies database connectivity and attempts reconnection when the connection is lost. The loop runs every `cache_health_check_interval` seconds (default: 30). The loop must use a dedicated lightweight connection, independent of the connection and `_tx_lock` used by cache operations. This ensures the health-check loop remains responsive even when the cache operation path is blocked or deadlocked. When the connection is healthy, the loop records a healthy status and is otherwise a no-op. When the connection is broken, the loop attempts reconnection and logs the result.

23. The health status maintained by the background health-check loop is the single source of truth for database health consumed by the readiness and liveness probes. Neither probe performs its own database query or acquires `_tx_lock`. The health status is an in-memory flag updated by two sources: (a) the background loop on each iteration, and (b) cache operations that encounter connection errors, which immediately mark the status as unhealthy (dual-feed model). Only the background loop may restore the status to healthy after successfully reconnecting on its independent connection. This ensures near-instant detection (first failed user query flips to unhealthy) and controlled recovery (background loop confirms before restoring healthy).

## Configuration Surface

| Field path | Description |
|---|---|
| `ols_config.conversation_cache.type` | Storage backend type: `"memory"` or `"postgres"` |
| `ols_config.conversation_cache.memory.max_entries` | Maximum total message entries for in-memory cache |
| `ols_config.conversation_cache.postgres.host` | PostgreSQL server hostname |
| `ols_config.conversation_cache.postgres.port` | PostgreSQL server port (1-65535) |
| `ols_config.conversation_cache.postgres.dbname` | PostgreSQL database name |
| `ols_config.conversation_cache.postgres.user` | PostgreSQL connection user |
| `ols_config.conversation_cache.postgres.password_path` | Path to file containing PostgreSQL password |
| `ols_config.conversation_cache.postgres.ssl_mode` | PostgreSQL SSL connection mode |
| `ols_config.conversation_cache.postgres.ca_cert_path` | Path to CA certificate for PostgreSQL TLS |
| `ols_config.conversation_cache.postgres.max_entries` | Maximum total message entries for PostgreSQL cache |
| `ols_config.history_compression_enabled` | Whether to use LLM-based history compression (default: true) |
| `ols_config.cache_health_check_interval` | Interval in seconds for the background PostgreSQL health-check loop (default: 30) [NEW: OLS-3221] |
| `ols_config.conversation_cache.postgres.statement_timeout` | Statement-level timeout in milliseconds for PostgreSQL operations (default: 5000) [NEW: OLS-3221] |
| `ols_config.conversation_cache.postgres.lock_timeout` | Timeout in seconds for Python-level `_tx_lock` acquisition (default: 10) [NEW: OLS-3221] |

## Constraints

1. **User isolation is a security invariant.** A user must never be able to read, modify, list, or delete another user's conversations. Every cache operation must scope access by the authenticated user_id. There is no administrative bypass for cross-user access.

2. Both user_id and conversation_id must be valid UUIDs (with or without dashes). Operations with malformed IDs must be rejected before any storage access occurs.

3. Capacity is measured in total message entries across all conversations, not in number of conversations.

4. PostgreSQL storage must use transactions with advisory locks for insert-or-append operations to prevent lost updates from concurrent writes to the same conversation.

5. Cache errors from the storage backend must be wrapped in a domain-specific CacheError exception, preserving the original cause.

6. The in-memory backend does not survive process restarts. It is suitable only for development and single-instance deployments.

7. When compression is disabled, history that exceeds the token budget is truncated by the token handler (newest messages kept, oldest dropped) with no summarization attempt.

## Planned Changes

- [PLANNED: OLS-3221] PostgreSQL resilience — auto-reconnection, background health-check loop, dual-feed health status, operation timeouts, and liveness probe DB check. See Rules 19–23.
- [PLANNED: OLS-2713] Persistent chat -- UI-side conversation persistence, enabling users to resume conversations across browser sessions.
- [PLANNED: OLS-141] Encrypting conversation state cache data, on disk and in memory.
- [PLANNED: OLS-251] Scale Postgres Conversation Cache Backend for high-availability and multi-instance production deployments.
