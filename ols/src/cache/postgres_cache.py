"""Cache that uses Postgres to store cached values."""

import json
import logging
import threading
from typing import Any

import psycopg2

from ols.app.models.config import PostgresConfig
from ols.app.models.models import (
    CacheEntry,
    ConversationData,
    MessageDecoder,
    MessageEncoder,
)
from ols.src.cache.cache import Cache
from ols.src.cache.cache_error import CacheError
from ols.utils.postgres import PostgresBase, connection

logger = logging.getLogger(__name__)


class PostgresCache(Cache, PostgresBase):
    """Cache that uses Postgres to store cached values.

    The cache itself is stored in following tables:

    Cache table:
    ```
         Column      |            Type             | Nullable | Default | Storage  |
    -----------------+-----------------------------+----------+---------+----------+
     user_id         | text                        | not null |         | extended |
     conversation_id | text                        | not null |         | extended |
     value           | bytea                       |          |         | extended |
     updated_at      | timestamp without time zone |          |         | plain    |
    Indexes:
        "cache_pkey" PRIMARY KEY, btree (user_id, conversation_id)
        "timestamps" btree (updated_at)
    ```

    Conversations metadata table:
    ```
         Column                 |            Type             | Nullable | Default |
    ---------------------------+-----------------------------+----------+---------+
     user_id                    | text                        | not null |         |
     conversation_id            | text                        | not null |         |
     topic_summary              | text                        |          | ''      |
     last_message_timestamp     | timestamp without time zone | not null |         |
     message_count              | integer                     |          | 0       |
    Indexes:
        "conversations_pkey" PRIMARY KEY, btree (user_id, conversation_id)
    ```
    """

    CREATE_CACHE_TABLE = """
        CREATE TABLE IF NOT EXISTS cache (
            user_id         text NOT NULL,
            conversation_id text NOT NULL,
            value           bytea,
            updated_at      timestamp,
            PRIMARY KEY(user_id, conversation_id)
        );
        """

    CREATE_CONVERSATIONS_TABLE = """
        CREATE TABLE IF NOT EXISTS conversations (
            user_id                text NOT NULL,
            conversation_id        text NOT NULL,
            topic_summary          text DEFAULT '',
            last_message_timestamp timestamp NOT NULL,
            message_count          integer DEFAULT 0,
            PRIMARY KEY(user_id, conversation_id)
        );
        """

    CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS timestamps
            ON cache (updated_at)
        """

    SELECT_CONVERSATION_HISTORY_STATEMENT = """
        SELECT value
          FROM cache
         WHERE user_id=%s AND conversation_id=%s LIMIT 1
        """

    UPDATE_CONVERSATION_HISTORY_STATEMENT = """
        UPDATE cache
           SET value=%s, updated_at=CURRENT_TIMESTAMP
         WHERE user_id=%s AND conversation_id=%s
        """

    INSERT_CONVERSATION_HISTORY_STATEMENT = """
        INSERT INTO cache(user_id, conversation_id, value, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """

    DELETE_CONVERSATION_HISTORY_STATEMENT = """
        DELETE FROM cache
         WHERE (user_id, conversation_id) in
               (SELECT user_id, conversation_id FROM cache ORDER BY updated_at LIMIT
        """

    QUERY_TOTAL_ENTRIES = """
        SELECT COALESCE(SUM(json_array_length(convert_from(value, 'utf-8')::json)), 0) FROM cache;
        """

    SELECT_OLDEST_ROW = """
        SELECT user_id, conversation_id, value FROM cache ORDER BY updated_at LIMIT 1;
        """

    DELETE_SINGLE_CONVERSATION_STATEMENT = """
        DELETE FROM cache
         WHERE user_id=%s AND conversation_id=%s
        """

    LIST_CONVERSATIONS_STATEMENT = """
        SELECT conversation_id, topic_summary,
               EXTRACT(EPOCH FROM last_message_timestamp) as last_message_timestamp,
               message_count
        FROM conversations
        WHERE user_id=%s
        ORDER BY last_message_timestamp DESC
    """

    INSERT_OR_UPDATE_TOPIC_SUMMARY_STATEMENT = """
        INSERT INTO conversations
            (user_id, conversation_id, topic_summary, last_message_timestamp, message_count)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP, 0)
        ON CONFLICT (user_id, conversation_id)
        DO UPDATE SET topic_summary = EXCLUDED.topic_summary,
                      last_message_timestamp = EXCLUDED.last_message_timestamp
    """

    UPSERT_CONVERSATION_STATEMENT = """
        INSERT INTO conversations
            (user_id, conversation_id, topic_summary, last_message_timestamp, message_count)
        VALUES (%s, %s, '', CURRENT_TIMESTAMP, 1)
        ON CONFLICT (user_id, conversation_id)
        DO UPDATE SET last_message_timestamp = CURRENT_TIMESTAMP,
                      message_count = conversations.message_count + 1
    """

    DELETE_CONVERSATION_METADATA_STATEMENT = """
        DELETE FROM conversations
         WHERE user_id=%s AND conversation_id=%s
    """

    ADVISORY_LOCK_STATEMENT = """
        SELECT pg_advisory_xact_lock(hashtext(%s || %s))
    """

    def __init__(self, config: PostgresConfig) -> None:
        """Create a new instance of Postgres cache."""
        self._tx_lock = threading.Lock()
        self.capacity = config.max_entries
        super().__init__(config)

    @property
    def _ddl_statements(self) -> list[str]:
        """Return DDL statements for cache tables and indexes."""
        return [
            self.CREATE_CACHE_TABLE,
            self.CREATE_CONVERSATIONS_TABLE,
            self.CREATE_INDEX,
        ]

    @connection
    def get(
        self, user_id: str, conversation_id: str, skip_user_id_check: bool = False
    ) -> list[CacheEntry]:
        """Get the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            The value associated with the key, or None if not found.
        """
        # just check if user_id and conversation_id are UUIDs
        super().construct_key(user_id, conversation_id, skip_user_id_check)

        with self._tx_lock:
            with self.connection.cursor() as cursor:
                try:
                    value = PostgresCache._select(cursor, user_id, conversation_id)
                    if value is None:
                        return []
                    history = [CacheEntry.from_dict(ce) for ce in value]
                    return history
                except psycopg2.DatabaseError as e:
                    logger.error("PostgresCache.get %s", e)
                    raise CacheError("PostgresCache.get", e) from e

    @connection
    def insert_or_append(
        self,
        user_id: str,
        conversation_id: str,
        cache_entry: CacheEntry,
        skip_user_id_check: bool = False,
    ) -> None:
        """Set the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            cache_entry: The `CacheEntry` object to store.
            skip_user_id_check: Skip user_id suid check.

        """
        value = cache_entry.to_dict()
        # autocommit=True makes each execute() its own transaction, so
        # pg_advisory_xact_lock would be released immediately after the first
        # execute.  Disable autocommit for a real multi-statement transaction.
        # The lock serialises concurrent callers that share this connection.
        with self._tx_lock:
            self.connection.autocommit = False
            with self.connection.cursor() as cursor:
                try:
                    cursor.execute(
                        self.ADVISORY_LOCK_STATEMENT,
                        (user_id, conversation_id),
                    )
                    old_value = self._select(cursor, user_id, conversation_id)
                    if old_value:
                        old_value.append(value)
                        PostgresCache._update(
                            cursor,
                            user_id,
                            conversation_id,
                            json.dumps(old_value, cls=MessageEncoder).encode("utf-8"),
                        )
                    else:
                        PostgresCache._insert(
                            cursor,
                            user_id,
                            conversation_id,
                            json.dumps([value], cls=MessageEncoder).encode("utf-8"),
                        )
                    cursor.execute(
                        PostgresCache.UPSERT_CONVERSATION_STATEMENT,
                        (user_id, conversation_id),
                    )
                    PostgresCache._cleanup(cursor, self.capacity)
                    self.connection.commit()
                except psycopg2.DatabaseError as e:
                    self.connection.rollback()
                    logger.error("PostgresCache.insert_or_append: %s", e)
                    raise CacheError("PostgresCache.insert_or_append", e) from e
                finally:
                    self.connection.autocommit = True

    @connection
    def delete(
        self, user_id: str, conversation_id: str, skip_user_id_check: bool = False
    ) -> bool:
        """Delete conversation history for a given user_id and conversation_id.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            bool: True if the conversation was deleted, False if not found.

        """
        with self._tx_lock:
            self.connection.autocommit = False
            with self.connection.cursor() as cursor:
                try:
                    deleted = PostgresCache._delete(cursor, user_id, conversation_id)
                    cursor.execute(
                        PostgresCache.DELETE_CONVERSATION_METADATA_STATEMENT,
                        (user_id, conversation_id),
                    )
                    self.connection.commit()
                    return deleted
                except psycopg2.DatabaseError as e:
                    self.connection.rollback()
                    logger.error("PostgresCache.delete: %s", e)
                    raise CacheError("PostgresCache.delete", e) from e
                finally:
                    self.connection.autocommit = True

    @connection
    def list(
        self, user_id: str, skip_user_id_check: bool = False
    ) -> list[ConversationData]:
        """List all conversations for a given user_id.

        Args:
            user_id: User identification.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            A list of ConversationData objects containing conversation_id,
            topic_summary, last_message_timestamp, and message_count.

        """
        with self._tx_lock:
            with self.connection.cursor() as cursor:
                try:
                    cursor.execute(
                        PostgresCache.LIST_CONVERSATIONS_STATEMENT, (user_id,)
                    )
                    rows = cursor.fetchall()
                    return [
                        ConversationData(
                            conversation_id=row[0],
                            topic_summary=row[1] or "",
                            last_message_timestamp=float(row[2]),
                            message_count=row[3] or 0,
                        )
                        for row in rows
                    ]
                except psycopg2.DatabaseError as e:
                    logger.error("PostgresCache.list: %s", e)
                    raise CacheError("PostgresCache.list", e) from e

    @connection
    def set_topic_summary(
        self,
        user_id: str,
        conversation_id: str,
        topic_summary: str,
        skip_user_id_check: bool = False,
    ) -> None:
        """Set or update the topic summary for a conversation.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            topic_summary: The topic summary to store.
            skip_user_id_check: Skip user_id suid check.
        """
        with self._tx_lock:
            with self.connection.cursor() as cursor:
                try:
                    cursor.execute(
                        PostgresCache.INSERT_OR_UPDATE_TOPIC_SUMMARY_STATEMENT,
                        (user_id, conversation_id, topic_summary),
                    )
                except psycopg2.DatabaseError as e:
                    logger.error("PostgresCache.set_topic_summary: %s", e)
                    raise CacheError("PostgresCache.set_topic_summary", e) from e

    def ready(self) -> bool:
        """Check if the cache is ready.

        Postgres cache checks if the connection is alive.

        Returns:
            True if the cache is ready, False otherwise.
        """
        # TODO: when the connection is closed and the database is back online,
        # we need to reestablish the connection => implement this
        if not self.connection or self.connection.closed == 1:
            return False
        try:
            return self.connection.poll() == psycopg2.extensions.POLL_OK
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            # OperationalError - the once alive connection is closed
            # InterfaceError - cannot reach the database server
            return False

    @staticmethod
    def _select(
        cursor: psycopg2.extensions.cursor,
        user_id: str,
        conversation_id: str,
        skip_user_id_check: bool = False,
    ) -> Any:
        """Select conversation history for given user_id and conversation_id."""
        cursor.execute(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        )
        value = cursor.fetchone()

        # check if history exists at all
        if value is None:
            return None

        # check the retrieved value
        if len(value) != 1:
            raise ValueError("Invalid value read from cache:", value)

        # convert from memoryview object to a string
        text_value = str(value[0], "utf-8")
        deserialized = json.loads(text_value, cls=MessageDecoder)

        # try to deserialize the value
        return deserialized

    @staticmethod
    def _update(
        cursor: psycopg2.extensions.cursor,
        user_id: str,
        conversation_id: str,
        value: bytes,
    ) -> None:
        """Update conversation history for given user_id and conversation_id."""
        cursor.execute(
            PostgresCache.UPDATE_CONVERSATION_HISTORY_STATEMENT,
            (value, user_id, conversation_id),
        )

    @staticmethod
    def _insert(
        cursor: psycopg2.extensions.cursor,
        user_id: str,
        conversation_id: str,
        value: bytes,
    ) -> None:
        """Insert new conversation history for given user_id and conversation_id."""
        cursor.execute(
            PostgresCache.INSERT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id, value),
        )

    @staticmethod
    def _cleanup(cursor: psycopg2.extensions.cursor, capacity: int) -> None:
        """Perform cleanup by evicting oldest messages until total_entries <= capacity."""
        cursor.execute(PostgresCache.QUERY_TOTAL_ENTRIES)
        result = cursor.fetchone()
        if result is None:
            total_entries = 0
        else:
            val = result[0]
            try:
                total_entries = int(val)
            except (TypeError, ValueError):
                total_entries = 0

        if total_entries > capacity:
            cursor.execute(PostgresCache.SELECT_OLDEST_ROW)
            row = cursor.fetchone()
            if not row:
                return
            user_id, conversation_id, value_bytes = row
            text_value = str(value_bytes, "utf-8")
            value = json.loads(text_value, cls=MessageDecoder)

            if len(value) > 1:
                value.pop(0)
                new_value_bytes = json.dumps(value, cls=MessageEncoder).encode("utf-8")
                PostgresCache._update(cursor, user_id, conversation_id, new_value_bytes)
            else:
                PostgresCache._delete(cursor, user_id, conversation_id)
            total_entries -= 1

    @staticmethod
    def _delete(
        cursor: psycopg2.extensions.cursor, user_id: str, conversation_id: str
    ) -> bool:
        """Delete conversation history for given user_id and conversation_id."""
        cursor.execute(
            PostgresCache.DELETE_SINGLE_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        )
        return cursor.rowcount > 0
