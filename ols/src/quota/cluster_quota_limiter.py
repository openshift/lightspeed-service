"""Simple cluster quota limiter where quota is fixed for the whole cluster."""

from ols.app.models.config import PostgresConfig
from ols.src.quota.revokable_quota_limiter import RevokableQuotaLimiter


class ClusterQuotaLimiter(RevokableQuotaLimiter):
    """Simple cluster quota limiter where quota is fixed for the whole cluster."""

    def __init__(
        self,
        config: PostgresConfig,
        initial_quota: int = 0,
        increase_by: int = 0,
    ) -> None:
        """Initialize quota limiter storage."""
        super().__init__(initial_quota, increase_by, "c", config)
