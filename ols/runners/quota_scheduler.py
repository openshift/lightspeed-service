"""User and cluster quota scheduler runner."""

import logging
from threading import Thread
from time import sleep
from typing import Any, Optional

import psycopg2

from ols import constants
from ols.app.models.config import LimiterConfig, PostgresConfig, QuotaHandlersConfig
from ols.utils.config import AppConfig

logger: logging.Logger = logging.getLogger(__name__)


INCREASE_QUOTA_STATEMENT = """
    UPDATE quota_limits
       SET available=available+%s, revoked_at=NOW()
     WHERE subject=%s
       AND revoked_at < NOW() - INTERVAL %s ;
    """

RESET_QUOTA_STATEMENT = """
    UPDATE quota_limits
       SET available=%s, revoked_at=NOW()
     WHERE subject=%s
       AND revoked_at < NOW() - INTERVAL %s ;
    """


RECONCILE_QUOTA_LIMITS_STATEMENT = """
    UPDATE quota_limits
       SET quota_limit = %s,
           available = available + (%s - quota_limit),
           updated_at = NOW()
     WHERE subject = %s
       AND quota_limit != %s
 RETURNING id, quota_limit, available
    """


def quota_scheduler(config: Optional[QuotaHandlersConfig]) -> bool:
    """Quota scheduler task."""
    if config is None:
        logger.warning("Quota limiters are not configured, skipping")
        return False

    if config.storage is None:
        logger.warning("Storage for quota limiter is not set, skipping")
        return False

    connection = connect(config.storage)
    if connection is None:
        logger.warning("Unable to connect to Postgres, skipping")
        return False

    if config.limiters is None:
        logger.warning("No limiters are setup, skipping")
        return False

    period = config.scheduler.period

    logger.info(
        "Quota scheduler started in separated thread with period set to %d seconds",
        period,
    )

    while True:
        logger.info("Quota scheduler sync started")
        for name, limiter in config.limiters.limiters.items():
            try:
                # Reconcile quota limits first to handle config changes
                reconcile_quota_limits(connection, name, limiter)
                # Then handle periodic quota revocation
                quota_revocation(connection, name, limiter)
            except Exception as e:
                logger.error("Quota scheduler error for '%s': %s", name, e)
        logger.info("Quota scheduler sync finished")
        sleep(period)
    # unreachable code
    connection.close()
    return True


def quota_revocation(connection: Any, name: str, quota_limiter: LimiterConfig) -> None:
    """Quota revocation mechanism."""
    logger.info(
        "Quota revocation mechanism for limiter '%s' of type '%s'",
        name,
        quota_limiter.type,
    )

    if quota_limiter.type is None:
        raise Exception("Limiter type not set, skipping revocation")

    if quota_limiter.period is None:
        raise Exception("Limiter period not set, skipping revocation")

    subject_id = get_subject_id(quota_limiter.type)

    if quota_limiter.quota_increase is not None:
        increase_quota(
            connection, subject_id, quota_limiter.quota_increase, quota_limiter.period
        )

    if quota_limiter.initial_quota is not None and quota_limiter.initial_quota > 0:
        reset_quota(
            connection, subject_id, quota_limiter.initial_quota, quota_limiter.period
        )


def increase_quota(
    connection: Any, subject_id: str, increase_by: int, period: str
) -> None:
    """Increase quota by specified amount."""
    logger.info(
        "Increasing quota for subject '%s' by %d when period %s is reached",
        subject_id,
        increase_by,
        period,
    )

    update_statement = INCREASE_QUOTA_STATEMENT

    with connection.cursor() as cursor:
        cursor.execute(
            update_statement,
            (
                increase_by,
                subject_id,
                period,
            ),
        )
        logger.info("Changed %d rows in database", cursor.rowcount)


def reset_quota(connection: Any, subject_id: str, reset_to: int, period: str) -> None:
    """Reset quota to specified amount."""
    logger.info(
        "Reseting quota for subject '%s' to %d when period %s is reached",
        subject_id,
        reset_to,
        period,
    )

    update_statement = RESET_QUOTA_STATEMENT

    with connection.cursor() as cursor:
        cursor.execute(
            update_statement,
            (
                reset_to,
                subject_id,
                period,
            ),
        )
        logger.info("Changed %d rows in database", cursor.rowcount)


def reconcile_quota_limits(
    connection: Any, name: str, quota_limiter: LimiterConfig
) -> None:
    """Reconcile quota limits with current configuration.

    Updates existing quota records when initial_quota changes while preserving
    consumed tokens.
    """
    if quota_limiter.type is None:
        logger.warning("Limiter type not set for '%s', skipping reconciliation", name)
        return

    if quota_limiter.initial_quota is None:
        logger.debug("No initial quota set for '%s', skipping reconciliation", name)
        return

    subject_id = get_subject_id(quota_limiter.type)
    new_quota = quota_limiter.initial_quota

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                RECONCILE_QUOTA_LIMITS_STATEMENT,
                (
                    new_quota,  # new quota_limit value
                    new_quota,  # used in calculation: available + (new - old)
                    subject_id,  # WHERE subject = ?
                    new_quota,  # WHERE quota_limit != ?
                ),
            )

            # Process returned rows for logging
            updated_rows = cursor.fetchall()
            if updated_rows:
                for row in updated_rows:
                    user_id, new_limit, new_available = row
                    logger.info(
                        "Reconciled quota for limiter='%s' subject='%s' type='%s': "
                        "quota_limit=%d, available=%d",
                        name,
                        user_id if user_id else "(cluster)",
                        quota_limiter.type,
                        new_limit,
                        new_available,
                    )
                logger.info(
                    "Quota reconciliation complete for '%s': updated %d record(s)",
                    name,
                    len(updated_rows),
                )
            else:
                logger.debug(
                    "No quota updates needed for '%s' "
                    "(no records or all match current configuration)",
                    name,
                )
    except Exception as e:
        logger.error(
            "Failed to reconcile quota limits for '%s': %s",
            name,
            e,
        )


def get_subject_id(limiter_type: str) -> str:
    """Get subject ID based on quota limiter type."""
    match limiter_type:
        case constants.USER_QUOTA_LIMITER:
            return "u"
        case constants.CLUSTER_QUOTA_LIMITER:
            return "c"
        case _:
            return "?"


def connect(config: PostgresConfig) -> Any:
    """Initialize connection to database."""
    logger.info("Initializing connection to quota limiter database")
    connection = psycopg2.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        dbname=config.dbname,
        sslmode=config.ssl_mode,
        # sslrootcert=config.ca_cert_path,
        gssencmode=config.gss_encmode,
    )
    if connection is not None:
        connection.autocommit = True
    return connection


def start_quota_scheduler(config: AppConfig) -> None:
    """Start user and cluster quota scheduler in separate thread."""
    logger.info("Starting quota scheduler")
    thread = Thread(
        target=quota_scheduler, daemon=True, args=(config.ols_config.quota_handlers,)
    )
    thread.start()
