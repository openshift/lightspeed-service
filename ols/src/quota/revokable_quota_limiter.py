"""Simple quota limiter where quota can be revoked."""

import logging
from datetime import datetime

from ols.app.models.config import PostgresConfig
from ols.src.quota.quota_exceed_error import QuotaExceedError
from ols.src.quota.quota_limiter import QuotaLimiter
from ols.utils.connection_decorator import connection

logger = logging.getLogger(__name__)


class RevokableQuotaLimiter(QuotaLimiter):
    """Simple quota limiter where quota can be revoked."""

    CREATE_QUOTA_TABLE = """
        CREATE TABLE IF NOT EXISTS quota_limits (
            id              text NOT NULL,
            subject         char(1) NOT NULL,
            quota_limit     int NOT NULL,
            available       int,
            updated_at      timestamp with time zone,
            revoked_at      timestamp with time zone,
            PRIMARY KEY(id, subject)
        );
        """

    INIT_QUOTA = """
        INSERT INTO quota_limits (id, subject, quota_limit, available, revoked_at)
        VALUES (%s, %s, %s, %s, %s)
        """

    SELECT_QUOTA = """
        SELECT available
          FROM quota_limits
         WHERE id=%s and subject=%s LIMIT 1
        """

    SET_AVAILABLE_QUOTA = """
        UPDATE quota_limits
           SET available=%s, revoked_at=%s
         WHERE id=%s and subject=%s
        """

    UPDATE_AVAILABLE_QUOTA = """
        UPDATE quota_limits
           SET available=available+%s, updated_at=%s
         WHERE id=%s and subject=%s
        """

    # Reconciliation SQL for user quotas - updates all user rows
    RECONCILE_USER_QUOTA_LIMITS = """
        UPDATE quota_limits
           SET available = available + (%s - quota_limit),
               quota_limit = %s,
               updated_at = NOW()
         WHERE subject = 'u'
           AND quota_limit != %s
     RETURNING id, quota_limit, available
        """

    # Reconciliation SQL for cluster quotas - updates cluster row
    RECONCILE_CLUSTER_QUOTA_LIMITS = """
        UPDATE quota_limits
           SET available = available + (%s - quota_limit),
               quota_limit = %s,
               updated_at = NOW()
         WHERE subject = 'c'
           AND quota_limit != %s
     RETURNING id, quota_limit, available
        """

    def __init__(
        self,
        initial_quota: int,
        increase_by: int,
        subject_type: str,
        connection_config: PostgresConfig,
    ) -> None:
        """Initialize quota limiter."""
        self.subject_type = subject_type
        self.initial_quota = initial_quota
        self.increase_by = increase_by
        self.connection_config = connection_config

    def reconcile_quota_limits(self) -> None:
        """Reconcile quota limits with current configuration.

        Public method to trigger reconciliation after initialization.
        Should be called after quota limiters are created to ensure
        database quota limits match OLSConfig after updates.
        """
        try:
            self._reconcile_quota_limits()
        except Exception as e:
            logger.warning(
                "Quota reconciliation failed (will retry on scheduler): %s", e
            )

    @connection
    def available_quota(self, subject_id: str = "") -> int:
        """Retrieve available quota for given subject."""
        if self.subject_type == "c":
            subject_id = ""
        with self.connection.cursor() as cursor:
            cursor.execute(
                RevokableQuotaLimiter.SELECT_QUOTA,
                (subject_id, self.subject_type),
            )
            value = cursor.fetchone()
            if value is None:
                self._init_quota(subject_id)
                return self.initial_quota
            return value[0]

    @connection
    def revoke_quota(self, subject_id: str = "") -> None:
        """Revoke quota for given subject."""
        if self.subject_type == "c":
            subject_id = ""
        # timestamp to be used
        revoked_at = datetime.now()

        with self.connection.cursor() as cursor:
            cursor.execute(
                RevokableQuotaLimiter.SET_AVAILABLE_QUOTA,
                (self.initial_quota, revoked_at, subject_id, self.subject_type),
            )
            self.connection.commit()

    @connection
    def increase_quota(self, subject_id: str = "") -> None:
        """Increase quota for given subject."""
        if self.subject_type == "c":
            subject_id = ""
        # timestamp to be used
        updated_at = datetime.now()

        with self.connection.cursor() as cursor:
            cursor.execute(
                RevokableQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
                (self.increase_by, updated_at, subject_id, self.subject_type),
            )
            self.connection.commit()

    def ensure_available_quota(self, subject_id: str = "") -> None:
        """Ensure that there's avaiable quota left."""
        if self.subject_type == "c":
            subject_id = ""
        available = self.available_quota(subject_id)
        logger.info("Available quota for subject %s is %d", subject_id, available)
        # check if ID still have available tokens to be consumed
        if available <= 0:
            e = QuotaExceedError(subject_id, self.subject_type, available)
            logger.exception("Quota exceed: %s", e)
            raise e

    @connection
    def consume_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        subject_id: str = "",
    ) -> None:
        """Consume tokens by given subject."""
        if self.subject_type == "c":
            subject_id = ""
        logger.info(
            "Consuming %d input and %d output tokens for subject %s",
            input_tokens,
            output_tokens,
            subject_id,
        )
        to_be_consumed = input_tokens + output_tokens

        with self.connection.cursor() as cursor:
            # timestamp to be used
            updated_at = datetime.now()

            cursor.execute(
                RevokableQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
                (-to_be_consumed, updated_at, subject_id, self.subject_type),
            )
            self.connection.commit()

    def _initialize_tables(self) -> None:
        """Initialize tables used by quota limiter."""
        logger.info("Initializing tables for quota limiter")
        cursor = self.connection.cursor()
        cursor.execute(RevokableQuotaLimiter.CREATE_QUOTA_TABLE)
        cursor.close()
        self.connection.commit()

    def _init_quota(self, subject_id: str = "") -> None:
        """Initialize quota for given ID."""
        # timestamp to be used
        revoked_at = datetime.now()

        with self.connection.cursor() as cursor:
            cursor.execute(
                RevokableQuotaLimiter.INIT_QUOTA,
                (
                    subject_id,
                    self.subject_type,
                    self.initial_quota,
                    self.initial_quota,
                    revoked_at,
                ),
            )
            self.connection.commit()

    def _reconcile_quota_limits(self) -> None:
        """Reconcile quota limits with current configuration.

        Updates existing quota records when initial_quota changes while preserving
        consumed tokens.
        """
        # Select appropriate SQL based on subject type
        if self.subject_type == "u":
            reconcile_sql = RevokableQuotaLimiter.RECONCILE_USER_QUOTA_LIMITS
        elif self.subject_type == "c":
            reconcile_sql = RevokableQuotaLimiter.RECONCILE_CLUSTER_QUOTA_LIMITS
        else:
            logger.error(
                "Unknown subject type '%s', skipping reconciliation", self.subject_type
            )
            return

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    reconcile_sql,
                    (
                        self.initial_quota,  # used in calculation: available + (new - old)
                        self.initial_quota,  # new quota_limit to set
                        self.initial_quota,  # WHERE quota_limit != ?
                    ),
                )
                updated_rows = cursor.fetchall()
                if updated_rows:
                    for row in updated_rows:
                        subject_id, new_limit, new_available = row
                        logger.info(
                            "Reconciled quota for subject='%s' type='%s': "
                            "quota_limit=%d, available=%d",
                            subject_id if subject_id else "(cluster)",
                            self.subject_type,
                            new_limit,
                            new_available,
                        )
                    logger.info(
                        "Quota reconciliation complete for subject type '%s': "
                        "updated %d record(s)",
                        self.subject_type,
                        len(updated_rows),
                    )
                else:
                    logger.debug(
                        "No quota updates needed for subject type '%s' "
                        "(no records or all match current configuration)",
                        self.subject_type,
                    )

                self.connection.commit()

        except Exception as e:
            logger.error(
                "Failed to reconcile quota limits for subject type '%s': %s",
                self.subject_type,
                e,
            )
            raise
