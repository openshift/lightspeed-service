"""Simple user quota limiter where each user have fixed quota."""

import logging

from ols.app.models.config import PostgresConfig
from ols.src.quota.revokable_quota_limiter import RevokableQuotaLimiter

logger = logging.getLogger(__name__)


class UserQuotaLimiter(RevokableQuotaLimiter):
    """Simple user quota limiter where each user have fixed quota."""

    def __init__(
        self,
        config: PostgresConfig,
        initial_quota: int = 0,
        increase_by: int = 0,
    ) -> None:
        """Initialize quota limiter storage."""
        subject = "u"  # user
        super().__init__(initial_quota, increase_by, subject, config)

        # initialize connection to DB
        self.connect()

        try:
            self._initialize_tables()
        except Exception as e:
            self.connection.close()
            logger.exception("Error initializing Postgres database:\n%s", e)
            raise
