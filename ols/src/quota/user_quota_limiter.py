"""Simple user quota limiter where each user have fixed quota."""

from ols.app.models.config import PostgresConfig
from ols.src.quota.revokable_quota_limiter import RevokableQuotaLimiter


class UserQuotaLimiter(RevokableQuotaLimiter):
    """Simple user quota limiter where each user have fixed quota."""

    def __init__(
        self,
        config: PostgresConfig,
        initial_quota: int = 0,
        increase_by: int = 0,
    ) -> None:
        """Initialize quota limiter storage."""
        super().__init__(initial_quota, increase_by, "u", config)
