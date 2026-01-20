"""Config status extraction and storage functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytz
from pydantic import BaseModel

from ols.app.models.config import Config
from ols.utils import suid

logger = logging.getLogger(__name__)


class ConfigStatus(BaseModel):
    """Feature-level configuration status representation.

    This class captures user-configurable settings exposed through the operator CRD.
    Does not expose any secrets or sensitive information.
    """

    providers: dict[str, list[str]]
    models: dict[str, list[str]]

    rag_indexes: list[str]

    query_redactor_enabled: bool
    query_filter_count: int

    providers_with_tls_config: list[str]

    mcp_servers: dict[str, str]

    quota_management_enabled: bool
    token_history_enabled: bool

    proxy_enabled: bool
    extra_ca_count: int


def extract_config_status(cfg: Config) -> ConfigStatus:
    """Extract feature-level configuration status from Config object.

    Args:
        cfg: The Config object to extract status from.

    Returns:
        ConfigStatus object representing user-configurable settings.
    """
    ols_cfg = cfg.ols_config
    llm_cfg = cfg.llm_providers
    mcp_cfg = cfg.mcp_servers

    providers: dict[str, list[str]] = {}
    for name, p in llm_cfg.providers.items():
        if p.type is not None:
            if p.type not in providers:
                providers[p.type] = []
            providers[p.type].append(name)
    models = {name: list(p.models.keys()) for name, p in llm_cfg.providers.items()}

    rag_indexes = (
        [
            idx.product_docs_index_id
            for idx in ols_cfg.reference_content.indexes
            if idx.product_docs_index_id
        ]
        if ols_cfg.reference_content and ols_cfg.reference_content.indexes
        else []
    )

    query_redactor_enabled = (
        ols_cfg.query_filters is not None and len(ols_cfg.query_filters) > 0
    )
    query_filter_count = len(ols_cfg.query_filters) if ols_cfg.query_filters else 0

    providers_with_tls_config = [
        name
        for name, p in llm_cfg.providers.items()
        if p.tls_security_profile and p.tls_security_profile.profile_type
    ]

    mcp_servers = {
        s.name: s.url.split("://")[0] if "://" in s.url else "http"
        for s in mcp_cfg.servers
    }

    quota_management_enabled = ols_cfg.quota_handlers is not None and (
        ols_cfg.quota_handlers.limiters is not None
        and len(ols_cfg.quota_handlers.limiters.limiters) > 0
    )
    token_history_enabled = (
        ols_cfg.quota_handlers is not None
        and ols_cfg.quota_handlers.enable_token_history is True
    )

    proxy_enabled = (
        ols_cfg.proxy_config is not None and ols_cfg.proxy_config.proxy_url is not None
    )
    extra_ca_count = len(ols_cfg.extra_ca)

    return ConfigStatus(
        providers=providers,
        models=models,
        rag_indexes=rag_indexes,
        query_redactor_enabled=query_redactor_enabled,
        query_filter_count=query_filter_count,
        providers_with_tls_config=providers_with_tls_config,
        mcp_servers=mcp_servers,
        quota_management_enabled=quota_management_enabled,
        token_history_enabled=token_history_enabled,
        proxy_enabled=proxy_enabled,
        extra_ca_count=extra_ca_count,
    )


def store_config_status(storage_path: str, config_status: ConfigStatus) -> None:
    """Store config status in the local filesystem.

    Args:
        storage_path: Path to the directory where config status will be stored.
        config_status: The ConfigStatus object to store.
    """
    path = Path(storage_path)
    path.mkdir(parents=True, exist_ok=True)

    data_to_store = {
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        **config_status.model_dump(),
    }

    config_status_file_path = path / f"{suid.get_suid()}.json"
    with open(config_status_file_path, "w", encoding="utf-8") as config_status_file:
        json.dump(data_to_store, config_status_file)

    logger.info("config status stored in '%s'", config_status_file_path)
