"""Shared base class for all Postgres-backed components.

Provides connection lifecycle (connect, reconnect, health check)
and an auto-reconnect decorator for public methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import psycopg2

from ols.app.models.config import PostgresConfig

logger = logging.getLogger(__name__)


def connection(f: Callable) -> Callable:
    """Ensure the object is connected before calling the wrapped method.

    If the connection is lost, reconnect transparently.
    """

    def wrapper(connectable: Any, *args: Any, **kwargs: Any) -> Callable:
        if not connectable.connected():
            connectable.connect()
        return f(connectable, *args, **kwargs)

    return wrapper


class PostgresBase(ABC):
    """Base class for components that store data in PostgreSQL.

    Subclasses declare their DDL via the ``_ddl_statements`` property.
    The base class handles connecting, executing DDL, committing, and
    health-checking.
    """

    def __init__(self, config: PostgresConfig) -> None:
        """Initialize Postgres connection and run DDL."""
        self.connection_config = config
        self.connection: Any = None
        self.connect()

    @property
    @abstractmethod
    def _ddl_statements(self) -> list[str]:
        """Return the DDL statements to execute during initialization."""

    INIT_ADVISORY_LOCK = """
        SELECT pg_advisory_xact_lock(hashtext('ols_schema_init'))
    """

    def connect(self) -> None:
        """Establish connection and initialize schema."""
        logger.info("Establishing connection to Postgres")
        self.connection = None
        config = self.connection_config
        self.connection = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            dbname=config.dbname,
            sslmode=config.ssl_mode,
            sslrootcert=config.ca_cert_path,
            gssencmode=config.gss_encmode,
        )
        try:
            cursor = self.connection.cursor()
            cursor.execute("SET LOCAL lock_timeout = '60s'")
            logger.info("Acquiring advisory lock for schema initialization")
            cursor.execute(self.INIT_ADVISORY_LOCK)
            for statement in self._ddl_statements:
                cursor.execute(statement)
            cursor.close()
            self.connection.commit()
        except Exception as e:
            self.connection.close()
            logger.exception("Error initializing Postgres schema:\n%s", e)
            raise
        self.connection.autocommit = True

    def connected(self) -> bool:
        """Check if the connection to Postgres is alive."""
        if self.connection is None:
            logger.warning("Not connected, need to reconnect later")
            return False
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            logger.info("Connection to storage is ok")
            return True
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logger.error("Disconnected from storage: %s", e)
            return False
