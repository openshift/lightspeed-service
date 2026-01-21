"""Quota limiter factory class."""

import logging

from ols import constants
from ols.app.models.config import PostgresConfig, QuotaHandlersConfig
from ols.src.quota.cluster_quota_limiter import ClusterQuotaLimiter
from ols.src.quota.quota_limiter import QuotaLimiter
from ols.src.quota.revokable_quota_limiter import RevokableQuotaLimiter
from ols.src.quota.user_quota_limiter import UserQuotaLimiter

logger = logging.getLogger(__name__)


class QuotaLimiterFactory:
    """Quota limiter factory class."""

    @staticmethod
    def quota_limiters(config: QuotaHandlersConfig) -> list[QuotaLimiter]:
        """Create instances of quota limiters based on loaded configuration.

        Returns:
            List of instances of 'QuotaLimiter',
        """
        limiters: list[QuotaLimiter] = []

        # storage (Postgres) configuration
        storage_config = config.storage
        if storage_config is None:
            logger.warning("Storage configuration for quota limiters not specified")
            return limiters

        limiters_config = config.limiters
        if limiters_config is None:
            logger.warning("Quota limiters are not specified in configuration")
            return limiters

        # fill-in list of initialized quota limiters
        for name, limiter_config in limiters_config.limiters.items():
            limiter_type = limiter_config.type
            initial_quota = limiter_config.initial_quota
            increase_by = limiter_config.quota_increase
            limiter = QuotaLimiterFactory.create_limiter(
                storage_config, limiter_type, initial_quota, increase_by
            )
            if isinstance(limiter, RevokableQuotaLimiter):
                limiter.reconcile_quota_limits()
            limiters.append(limiter)
            logger.info("Set up quota limiter '%s'", name)
        return limiters

    @staticmethod
    def create_limiter(
        storage_config: PostgresConfig,
        limiter_type: str,
        initial_quota: int,
        increase_by: int,
    ) -> QuotaLimiter:
        """Create selected quota limiter."""
        match limiter_type:
            case constants.USER_QUOTA_LIMITER:
                return UserQuotaLimiter(storage_config, initial_quota, increase_by)
            case constants.CLUSTER_QUOTA_LIMITER:
                return ClusterQuotaLimiter(storage_config, initial_quota, increase_by)
            case _:
                raise ValueError(f"Invalid limiter type: {limiter_type}.")
