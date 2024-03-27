"""Cache that uses Postgres to store cached values."""

import logging
import pickle
from typing import Optional

from langchain_core.messages.base import BaseMessage

from ols.app.models.config import PostgresConfig
from ols.src.cache.cache import Cache

logger = logging.getLogger(__name__)


class PostgresCache(Cache):
    """Cache that uses Postgres to store cached values."""

    CREATE_CACHE_TABLE = """
        CREATE TABLE IF NOT EXISTS cache (
            user_id         text NOT NULL,
            conversation_id text NOT NULL,
            key             text UNIQUE NOT NULL,
            value           bytea,
            updated_at      timestamp,
            PRIMARY KEY(user_id, conversation_id)
        );
        """

    CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS timestamps
            ON cache (updated_at)
        """

    SELECT_STATEMENT = """
        SELECT value
          FROM cache
         WHERE user_id=%s AND conversation_id=%s LIMIT 1
        """

    UPDATE_STATEMENT = """
        UPDATE cache
           SET value=%s, updated_at=CURRENT_TIMESTAMP
         WHERE user_id=%s AND conversation_id=%s
        """

    INSERT_STATEMENT = """
        INSERT INTO cache(user_id, conversation_id, value, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """

    DELETE_STATEMENT = """
        DELETE FROM cache
         WHERE (user_id, conversation_id) in
               (SELECT user_id, conversation_id FROM cache ORDER BY updated_at LIMIT
        """

    def __init__(self, config: PostgresConfig):
        """Create a new instance of Postgres cache."""
        import psycopg2

        # initialize connection to DB
        connection_string = "postgresql://localhost:5432"  # TODO from config
        self.conn = psycopg2.connect(connection_string)
        try:
            self.initialize_cache()
        except Exception as e:
            self.conn.close()
            logger.exception(f"Error initializing Postgres cache:\n{e}")
            raise e
        self.capacity = 1000  # TODO from config

    def initialize_cache(self) -> None:
        """Initialize cache - clean it up etc."""
        cur = self.conn.cursor()
        cur.execute(PostgresCache.CREATE_CACHE_TABLE)
        cur.execute(PostgresCache.CREATE_INDEX)
        cur.execute("create new cache table")
        cur.close()
        self.conn.commit()

    def get(self, user_id: str, conversation_id: str) -> Optional[list[BaseMessage]]:
        """Get the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.

        Returns:
            The value associated with the key, or None if not found.
        """
        cur = self.conn.cursor()
        cur.execute(PostgresCache.SELECT_STATEMENT, (user_id, conversation_id))
        value = cur.fetchone()
        cur.close()
        if value is None:
            return None
        return pickle.loads(value, errors="strict")  # noqa S301

    def insert_or_append(
        self, user_id: str, conversation_id: str, value: list[BaseMessage]
    ) -> None:
        """Set the value associated with the given key.

        Args:
            user_id: User identification.
            conversation_id: Conversation ID unique for given user.
            value: The value to set.

        Raises:
            OutOfMemoryError: If item is evicted when Redis allocated
                memory is higher than maxmemory.
        """
        old_value = self.get(user_id, conversation_id)
        if old_value:
            old_value.extend(value)
            self._update(
                user_id,
                conversation_id,
                pickle.dumps(old_value, protocol=pickle.HIGHEST_PROTOCOL),
            )
        else:
            self._insert(
                user_id,
                conversation_id,
                pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
            )
            self._cleanup()
        self.conn.commit()

    def _update(self, user_id: str, conversation_id: str, value: str) -> None:
        cur = self.conn.cursor()
        cur.execute(PostgresCache.UPDATE_STATEMENT, (value, user_id, conversation_id))
        cur.close()

    def _insert(self, user_id: str, conversation_id: str, value: str) -> None:
        cur = self.conn.cursor()
        cur.execute(PostgresCache.INSERT_STATEMENT, (user_id, conversation_id, value))
        cur.close()

    def _cleanup(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT count(*) FROM cache;")
        count = cur.fetchone()[0]
        limit = count - self.capacity
        if limit > 0:
            cur.execute(f"{PostgresCache.DELETE_STATEMENT} {count-self.capacity})")
        cur.close()
