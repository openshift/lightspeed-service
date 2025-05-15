"""Abstract class that is parent for all quota limiter implementations."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import psycopg2

if TYPE_CHECKING:
    from ols.app.models.config import PostgresConfig

logger = logging.getLogger(__name__)


class QuotaLimiter(ABC):
    """Abstract class that is parent for all quota limiter implementations."""

    @abstractmethod
    def available_quota(self, subject_id: str) -> int:
        """Retrieve available quota for given user."""

    @abstractmethod
    def revoke_quota(self) -> None:
        """Revoke quota for given user."""

    @abstractmethod
    def increase_quota(self) -> None:
        """Increase quota for given user."""

    @abstractmethod
    def ensure_available_quota(self, subject_id: str = "") -> None:
        """Ensure that there's avaiable quota left."""

    @abstractmethod
    def consume_tokens(
        self, input_tokens: int, output_tokens: int, subject_id: str = ""
    ) -> None:
        """Consume tokens by given user."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize connection config."""
        self.connection_config: Optional[PostgresConfig] = None

    @abstractmethod
    def _initialize_tables(self) -> None:
        """Initialize tables and indexes."""

    # pylint: disable=W0201
    def connect(self) -> None:
        """Initialize connection to database."""
        logger.info("Establishing connection to storage")
        config = self.connection_config
        # make sure the connection will have known state
        self.connection = None
        self.connection = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            dbname=config.dbname,
            sslmode=config.ssl_mode,
            # sslrootcert=config.ca_cert_path,
            gssencmode=config.gss_encmode,
        )

        try:
            self._initialize_tables()
        except Exception as e:
            self.connection.close()
            logger.exception("Error initializing Postgres database:\n%s", e)
            raise

        self.connection.autocommit = True

    def connected(self) -> bool:
        """Check if connection to cache is alive."""
        if self.connection is None:
            logger.warning("Not connected, need to reconnect later")
            return False
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            logger.info("Connection to storage is ok")
            return True
        except psycopg2.OperationalError as e:
            logger.error("Disconnected from storage: %s", e)
            return False
