"""Simple cluster quota limiter where quota is fixed for the whole cluster."""

import logging

from ols.app.models.config import PostgresConfig
from ols.src.quota.revokable_quota_limiter import RevokableQuotaLimiter

logger = logging.getLogger(__name__)


class ClusterQuotaLimiter(RevokableQuotaLimiter):
    """Simple cluster quota limiter where quota is fixed for the whole cluster."""

    def __init__(
        self,
        config: PostgresConfig,
        initial_quota: int = 0,
        increase_by: int = 0,
    ) -> None:
        """Initialize quota limiter storage."""
        subject = "c"  # cluster
        super().__init__(initial_quota, increase_by, subject)

        # initialize connection to DB
        self.connect(config)

        try:
            self._initialize_tables()
        except Exception as e:
            self.connection.close()
            logger.exception("Error initializing Postgres database:\n%s", e)
            raise
