"""Repository interface and implementation for quota metrics data access."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import psycopg2

from ols.app.models.config import PostgresConfig

logger = logging.getLogger(__name__)


@dataclass
class QuotaRecord:
    """Data class representing a quota record from the database."""

    id: str
    subject: str
    quota_limit: int
    available: int
    updated_at: datetime

    @property
    def utilization_percent(self) -> float:
        """Calculate quota utilization as a percentage."""
        if self.quota_limit == 0:
            return 0.0
        used = self.quota_limit - self.available
        return (used / self.quota_limit) * 100.0


@dataclass
class TokenUsageRecord:
    """Data class representing a token usage record from the database."""

    user_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    updated_at: datetime

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens (input + output)."""
        return self.input_tokens + self.output_tokens


class QuotaMetricsRepository(ABC):
    """Abstract repository interface for quota metrics data access."""

    @abstractmethod
    def get_quota_records(self) -> List[QuotaRecord]:
        """Retrieve all quota records from the database."""
        raise NotImplementedError

    @abstractmethod
    def get_token_usage_records(self) -> List[TokenUsageRecord]:
        """Retrieve all token usage records from the database."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        raise NotImplementedError


class PostgresQuotaMetricsRepository(QuotaMetricsRepository):
    """PostgreSQL implementation of the quota metrics repository."""

    SELECT_QUOTA_RECORDS = """
        SELECT id, subject, quota_limit, available, updated_at
        FROM quota_limits
        WHERE revoked_at IS NULL
        ORDER BY subject, id
    """

    SELECT_USAGE_RECORDS = """
        SELECT user_id, provider, model, input_tokens, output_tokens, updated_at
        FROM token_usage
        ORDER BY user_id, provider, model
    """

    def __init__(self, config: PostgresConfig) -> None:
        """Initialize the repository with database connection configuration."""
        self.connection_config = config
        self.connection: Optional[psycopg2.extensions.connection] = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to the PostgreSQL database."""
        logger.info("Establishing connection to PostgreSQL for quota metrics")

        try:
            self.connection = psycopg2.connect(
                host=self.connection_config.host,
                port=self.connection_config.port,
                user=self.connection_config.user,
                password=self.connection_config.password,
                dbname=self.connection_config.dbname,
                sslmode=self.connection_config.ssl_mode,
                gssencmode=self.connection_config.gss_encmode,
            )
            self.connection.autocommit = True
            logger.info("Successfully connected to PostgreSQL for quota metrics")
        except Exception as e:
            if self.connection:
                self.connection.close()
            logger.exception("Error connecting to PostgreSQL for quota metrics: %s", e)
            raise

    def _ensure_connected(self) -> None:
        """Ensure database connection is established."""
        if self.connection is None or self.connection.closed:
            logger.warning("Database connection lost, reconnecting...")
            self._connect()

    def get_quota_records(self) -> List[QuotaRecord]:
        """Retrieve all quota records from the database."""
        self._ensure_connected()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(self.SELECT_QUOTA_RECORDS)
                rows = cursor.fetchall()

                return [
                    QuotaRecord(
                        id=row[0],
                        subject=row[1],
                        quota_limit=row[2],
                        available=row[3],
                        updated_at=row[4],
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error("Error retrieving quota records: %s", e)
            raise

    def get_token_usage_records(self) -> List[TokenUsageRecord]:
        """Retrieve all token usage records from the database."""
        self._ensure_connected()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(self.SELECT_USAGE_RECORDS)
                rows = cursor.fetchall()

                return [
                    TokenUsageRecord(
                        user_id=row[0],
                        provider=row[1],
                        model=row[2],
                        input_tokens=row[3],
                        output_tokens=row[4],
                        updated_at=row[5],
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error("Error retrieving token usage records: %s", e)
            raise

    def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            self._ensure_connected()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            logger.debug("Database health check passed")
            return True
        except Exception as e:
            logger.error("Database health check failed: %s", e)
            return False
