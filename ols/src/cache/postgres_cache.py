"""Cache that uses Postgres to store cached values."""

import json
import logging
from typing import Any

import psycopg2

from ols.app.models.config import PostgresConfig
from ols.app.models.models import CacheEntry, MessageDecoder, MessageEncoder
from ols.src.cache.cache import Cache
from ols.src.cache.cache_error import CacheError

logger = logging.getLogger(__name__)


class PostgresCache(Cache):
    """Cache that uses Postgres to store cached values.

    The cache itself is stored in following table:

    ```
         Column      |            Type             | Nullable | Default | Storage  |
    -----------------+-----------------------------+----------+---------+----------+
     user_id         | text                        | not null |         | extended |
     conversation_id | text                        | not null |         | extended |
     value           | bytea                       |          |         | extended |
     updated_at      | timestamp without time zone |          |         | plain    |
    Indexes:
        "cache_pkey" PRIMARY KEY, btree (user_id, conversation_id)
        "cache_key_key" UNIQUE CONSTRAINT, btree (key)
        "timestamps" btree (updated_at)
    Access method: heap
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

    QUERY_CACHE_SIZE = """
        SELECT count(*) FROM cache;
        """

    DELETE_SINGLE_CONVERSATION_STATEMENT = """
        DELETE FROM cache
         WHERE user_id=%s AND conversation_id=%s
        """

    LIST_CONVERSATIONS_STATEMENT = """
        SELECT conversation_id
        FROM cache
        WHERE user_id=%s
        ORDER BY updated_at DESC
    """

    def __init__(self, config: PostgresConfig) -> None:
        """Create a new instance of Postgres cache."""
        # initialize connection to DB
        self.conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            dbname=config.dbname,
            sslmode=config.ssl_mode,
            sslrootcert=config.ca_cert_path,
        )
        self.conn.autocommit = True
        try:
            self.initialize_cache()
        except Exception as e:
            self.conn.close()
            logger.exception("Error initializing Postgres cache:\n%s", e)
            raise
        self.capacity = config.max_entries

    def initialize_cache(self) -> None:
        """Initialize cache - clean it up etc."""
        cur = self.conn.cursor()
        cur.execute(PostgresCache.CREATE_CACHE_TABLE)
        cur.execute(PostgresCache.CREATE_INDEX)
        cur.close()
        self.conn.commit()

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
        with self.conn.cursor() as cursor:
            try:
                value = PostgresCache._select(cursor, user_id, conversation_id)
                if value is None:
                    return []
                return [CacheEntry.from_dict(cache_entry) for cache_entry in value]
            except psycopg2.DatabaseError as e:
                logger.error("PostgresCache.get %s", e)
                raise CacheError("PostgresCache.get", e) from e

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
        # the whole operation is run in one transaction
        with self.conn.cursor() as cursor:
            try:
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
                    PostgresCache._cleanup(cursor, self.capacity)
                # commit is implicit at this point
            except psycopg2.DatabaseError as e:
                logger.error("PostgresCache.insert_or_append: %s", e)
                raise CacheError("PostgresCache.insert_or_append", e) from e

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
        with self.conn.cursor() as cursor:
            try:
                return PostgresCache._delete(cursor, user_id, conversation_id)
            except psycopg2.DatabaseError as e:
                logger.error("PostgresCache.delete: %s", e)
                raise CacheError("PostgresCache.delete", e) from e

    def list(self, user_id: str, skip_user_id_check: bool = False) -> list[str]:
        """List all conversations for a given user_id.

        Args:
            user_id: User identification.
            skip_user_id_check: Skip user_id suid check.

        Returns:
            A list of conversation ids from the cache

        """
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(PostgresCache.LIST_CONVERSATIONS_STATEMENT, (user_id,))
                rows = cursor.fetchall()
                return [row[0] for row in rows]
            except psycopg2.DatabaseError as e:
                logger.error("PostgresCache.list: %s", e)
                raise CacheError("PostgresCache.list", e) from e

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

        # try to deserialize the value
        return json.loads(value[0], cls=MessageDecoder)

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
        """Perform cleanup old conversation histories."""
        cursor.execute(PostgresCache.QUERY_CACHE_SIZE)
        value = cursor.fetchone()
        if value is not None:
            count = value[0]
            limit = count - capacity
            if limit > 0:
                cursor.execute(
                    f"{PostgresCache.DELETE_CONVERSATION_HISTORY_STATEMENT} {count - capacity})"
                )

    @staticmethod
    def _delete(
        cursor: psycopg2.extensions.cursor, user_id: str, conversation_id: str
    ) -> bool:
        """Delete conversation history for given user_id and conversation_id."""
        cursor.execute(
            PostgresCache.DELETE_SINGLE_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        )
        return cursor.fetchone() is not None
