# Cache -- Architecture

The cache subsystem provides persistent conversation history storage, keyed by
user ID and conversation ID. It ships two backends (in-memory LRU and
PostgreSQL) behind a common abstract interface, selected at startup via factory.

## Module Map

| File | Key symbols | Responsibility |
|---|---|---|
| `ols/src/cache/cache.py` | `Cache` (ABC) | Abstract interface. Defines `get`, `insert_or_append`, `delete`, `list`, `set_topic_summary`, `ready`. Also provides `construct_key` (static) for compound key creation and ID validation via `check_suid`. |
| `ols/src/cache/in_memory_cache.py` | `InMemoryCache` | Thread-safe singleton LRU cache backed by a `deque` (for recency ordering) and a `dict` (for O(1) lookup). Capacity is measured in total message entries across all conversations. |
| `ols/src/cache/postgres_cache.py` | `PostgresCache` | PostgreSQL-backed cache using `psycopg2`. Stores serialized JSON in a `bytea` column. Uses advisory locks for write serialization and a separate `conversations` metadata table. |
| `ols/src/cache/cache_factory.py` | `CacheFactory.conversation_cache()` | Static factory. Maps config type string (`"memory"` / `"postgres"`) to concrete `Cache` subclass. |
| `ols/src/cache/cache_error.py` | `CacheError` | Domain exception wrapping any database or cache operation failure. |
| `ols/utils/postgres.py` | `PostgresBase`, `connection` decorator | Base class for all Postgres-backed components. Handles connect/reconnect, DDL execution under an advisory lock, and the `@connection` decorator that transparently reconnects before method calls. |
| `ols/app/models/models.py` | `CacheEntry`, `ConversationData`, `MessageEncoder`, `MessageDecoder` | Data models. `CacheEntry` wraps a `HumanMessage`/`AIMessage` pair plus attachments and tool call data. Encoder/Decoder handle JSON serialization of LangChain message objects. |

## Data Flow

### Initialization

```
olsconfig.yaml
  -> OLSConfig.__init__ builds ConversationCacheConfig (validates type + sub-config)
  -> AppConfig.conversation_cache property (lazy)
     -> CacheFactory.conversation_cache(config)
        -> match config.type:
             "memory"   -> InMemoryCache(config.memory)   [singleton]
             "postgres" -> PostgresCache(config.postgres)  [connects + runs DDL]
        -> returns Cache instance stored on AppConfig
```

### Read (get)

```
Endpoint / HistorySupport
  -> config.conversation_cache.get(user_id, conversation_id, skip_user_id_check)
  -> construct_key validates IDs (UUID format via check_suid)
  -> InMemory: dict lookup, promote key to front of deque, return list[CacheEntry]
  -> Postgres: @connection ensures live connection, SELECT value WHERE (user_id, conversation_id),
               deserialize JSON via MessageDecoder, return list[CacheEntry]
```

### Write (insert_or_append)

```
ols.py endpoint
  -> config.conversation_cache.insert_or_append(user_id, conversation_id, cache_entry, ...)
  -> CacheEntry.to_dict() produces {"human_query": HumanMessage, "ai_response": AIMessage, ...}
  -> InMemory: append to dict list, promote in deque, increment total_entries,
               update ConversationData metadata, evict from tail if over capacity
  -> Postgres: acquire advisory lock -> SELECT existing -> append to JSON array ->
               UPDATE (or INSERT if new) -> upsert conversations metadata ->
               _cleanup evicts oldest message if over capacity -> COMMIT
```

### List / Delete / SetTopicSummary

All follow the same pattern: validate IDs, acquire lock (thread mutex for
in-memory, `_tx_lock` for Postgres), perform the operation, return result.

## Key Abstractions

### Abstract Cache Interface

All methods receive `user_id`, `conversation_id`, and `skip_user_id_check`.
The compound cache key is `"{user_id}:{conversation_id}"`.

| Method | Signature | Contract |
|---|---|---|
| `get` | `(user_id, conversation_id, skip_user_id_check) -> list[CacheEntry]` | Returns list of cache entries or `None`/`[]` if not found. |
| `insert_or_append` | `(user_id, conversation_id, cache_entry, skip_user_id_check) -> None` | Creates new conversation or appends to existing. Triggers capacity eviction. |
| `delete` | `(user_id, conversation_id, skip_user_id_check) -> bool` | Deletes all entries for a conversation. Returns `True` if something was deleted. |
| `list` | `(user_id, skip_user_id_check) -> list[ConversationData]` | Returns all conversations for a user, sorted by `last_message_timestamp` descending. |
| `set_topic_summary` | `(user_id, conversation_id, topic_summary, skip_user_id_check) -> None` | Upserts a human-readable summary for the conversation. |
| `ready` | `() -> bool` | Health check. In-memory always returns `True`; Postgres checks connection liveness. |

### CacheEntry

A Pydantic model holding one exchange:

- `query`: `HumanMessage` (LangChain)
- `response`: `AIMessage` (LangChain), defaults to `AIMessage("")`
- `attachments`: `list[Attachment]`
- `tool_calls`: `list[dict]`
- `tool_results`: `list[dict]`

`to_dict()` serializes to `{"human_query": HumanMessage, "ai_response": AIMessage, "attachments": [...], "tool_calls": [...], "tool_results": [...]}`.

`from_dict()` reconstructs from that shape.

### ConversationData

Metadata model returned by `list()`:

- `conversation_id`: str
- `topic_summary`: str (default `""`)
- `last_message_timestamp`: float (Unix epoch)
- `message_count`: int

### Factory Pattern

`CacheFactory.conversation_cache(config)` uses `match config.type` to dispatch.
The config type constants are `"memory"` and `"postgres"` (defined in
`ols/constants.py`).

## Integration Points

| Consumer | Usage |
|---|---|
| `ols/app/endpoints/ols.py` | Calls `insert_or_append` after generating a response to persist the exchange. |
| `ols/app/endpoints/conversations.py` | Calls `list`, `get`, `delete`, `set_topic_summary` for the conversations REST API. |
| `ols/src/query_helpers/history_support.py` | Calls `get` to retrieve history for context, `delete` + `insert_or_append` to rewrite compressed history. |
| `ols/app/endpoints/health.py` | Calls `ready()` for the `/readiness` health check. |
| `ols/utils/config.py` | `AppConfig.conversation_cache` property lazily creates the cache via `CacheFactory`. |

## Implementation Notes

### PostgreSQL Table Schema

Two tables, created with `CREATE TABLE IF NOT EXISTS`:

```sql
CREATE TABLE IF NOT EXISTS cache (
    user_id         text NOT NULL,
    conversation_id text NOT NULL,
    value           bytea,
    updated_at      timestamp,
    PRIMARY KEY(user_id, conversation_id)
);
CREATE INDEX IF NOT EXISTS timestamps ON cache (updated_at);

CREATE TABLE IF NOT EXISTS conversations (
    user_id                text NOT NULL,
    conversation_id        text NOT NULL,
    topic_summary          text DEFAULT '',
    last_message_timestamp timestamp NOT NULL,
    message_count          integer DEFAULT 0,
    PRIMARY KEY(user_id, conversation_id)
);
```

The `value` column in `cache` stores a JSON array of serialized `CacheEntry`
dicts, encoded as UTF-8 bytes.

### Advisory Lock Key Derivation

Write operations acquire a PostgreSQL transaction-level advisory lock before
reading+modifying the row:

```sql
SELECT pg_advisory_xact_lock(hashtext(user_id || conversation_id))
```

`hashtext` produces a 32-bit integer from the concatenated `user_id` and
`conversation_id` strings. The lock is automatically released when the
transaction commits or rolls back. This serializes concurrent writers to the
same conversation without blocking unrelated conversations.

Schema DDL is also guarded by a separate advisory lock:
`pg_advisory_xact_lock(hashtext('ols_schema_init'))`.

### JSON Serialization Format

LangChain `HumanMessage` and `AIMessage` objects are serialized via
`MessageEncoder`:

```json
{
  "type": "human",
  "content": "...",
  "response_metadata": {},
  "additional_kwargs": {}
}
```

A full `CacheEntry` serializes as:

```json
{
  "human_query": {"type": "human", "content": "...", ...},
  "ai_response": {"type": "ai", "content": "...", ...},
  "attachments": [...],
  "tool_calls": [...],
  "tool_results": [...]
}
```

The `value` column stores a JSON array of these objects. `MessageDecoder` uses
an `object_hook` that inspects the `"type"` key to reconstruct `HumanMessage`
or `AIMessage`, and `"__type__": "CacheEntry"` to reconstruct `CacheEntry`
objects. Deserialization reads the `bytea` column as `memoryview`, converts to
`str` via `str(value, "utf-8")`, then passes through `json.loads` with
`cls=MessageDecoder`.

### In-Memory LRU Implementation

- **Singleton**: `__new__` + `threading.Lock` ensures one instance per process.
- **Data structures**: `dict[str, list[dict]]` for storage, `deque[str]` for
  LRU ordering, `dict[str, ConversationData]` for metadata.
- **Capacity**: Measured in total individual message entries across all
  conversations (not number of conversations).
- **LRU promotion**: On `get` or `insert_or_append`, the key is moved to the
  front of the deque.
- **Eviction**: When `total_entries > capacity`, the oldest single message
  (first element of the list at the tail of the deque) is removed. If that
  conversation's list becomes empty, the entire conversation is removed from
  all data structures.

### Thread Safety

- **InMemoryCache**: A class-level `threading.Lock` guards singleton creation
  and all mutation operations (`insert_or_append`, `delete`, `list`,
  `set_topic_summary`). Note that `get` does not acquire the lock -- it only
  does deque reordering which is not guarded.
- **PostgresCache**: A per-instance `threading.Lock` (`_tx_lock`) serializes
  all database operations at the application level. Within Postgres,
  `pg_advisory_xact_lock` provides row-level write serialization across
  processes/pods.
- **Connection decorator**: The `@connection` decorator on `PostgresBase`
  checks connection liveness and reconnects before each public method call.

### Capacity Eviction (Postgres)

After each `insert_or_append`, `_cleanup` runs:

1. `SELECT COALESCE(SUM(json_array_length(...)), 0) FROM cache` to count total
   message entries across all rows.
2. If over capacity, select the oldest row (by `updated_at`).
3. If that row has multiple messages, pop the first (oldest) message and update.
4. If that row has only one message, delete the entire row.

This removes one message per insert, keeping the total at or below capacity.

### How to Add a New Cache Backend

1. Create a new file in `ols/src/cache/` (e.g., `redis_cache.py`).
2. Subclass `Cache` and implement all abstract methods: `get`,
   `insert_or_append`, `delete`, `list`, `set_topic_summary`, `ready`.
3. Add a config model for the new backend in `ols/app/models/config.py`.
4. Add a constant for the new cache type in `ols/constants.py`.
5. Add a `case` branch in `CacheFactory.conversation_cache()`.
6. Add the corresponding config validation in `ConversationCacheConfig`.
