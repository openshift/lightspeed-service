"""Unit tests for config status collection."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ols.app.models.config import (
    AuthenticationConfig,
    Config,
    ConversationCacheConfig,
    DevConfig,
    InMemoryCacheConfig,
    LLMProviders,
    MCPServerConfig,
    MCPServers,
    OLSConfig,
    ProviderConfig,
    QueryFilter,
    ReferenceContent,
    ReferenceContentIndex,
    StdioTransportConfig,
    TLSSecurityProfile,
    UserDataCollection,
)
from ols.src.config_status import (
    ConfigStatus,
    extract_config_status,
    store_config_status,
)


def create_minimal_config() -> Config:
    """Create a minimal Config object for testing."""
    config = Config()
    config.llm_providers = LLMProviders()

    provider = ProviderConfig()
    provider.name = "test_provider"
    provider.type = "openai"
    provider.models = {}
    config.llm_providers.providers = {"test_provider": provider}

    config.ols_config = OLSConfig()
    config.ols_config.default_provider = "test_provider"
    config.ols_config.default_model = "test_model"
    config.ols_config.conversation_cache = ConversationCacheConfig()
    config.ols_config.conversation_cache.type = "memory"
    config.ols_config.conversation_cache.memory = InMemoryCacheConfig()
    config.ols_config.authentication_config = AuthenticationConfig()
    config.ols_config.authentication_config.module = "k8s"
    config.ols_config.user_data_collection = UserDataCollection()

    config.dev_config = DevConfig()
    config.mcp_servers = MCPServers()

    return config


class TestConfigStatus:
    """Tests for ConfigStatus model."""

    def test_config_status_model_creation(self):
        """Test ConfigStatus model can be created with all fields."""
        status = ConfigStatus(
            providers={"openai": ["my_openai"]},
            models={"my_openai": ["gpt-4"]},
            rag_indexes=[],
            query_redactor_enabled=False,
            query_filter_count=0,
            providers_with_tls_config=[],
            mcp_servers={},
            quota_management_enabled=False,
            token_history_enabled=False,
            proxy_enabled=False,
            extra_ca_count=0,
        )

        assert status.providers == {"openai": ["my_openai"]}
        assert status.models == {"my_openai": ["gpt-4"]}
        assert status.rag_indexes == []


class TestExtractConfigStatus:
    """Tests for extract_config_status function."""

    def test_extract_config_status_minimal_config(self):
        """Test extracting config status from a minimal config."""
        config = create_minimal_config()
        status = extract_config_status(config)

        assert status.providers == {"openai": ["test_provider"]}
        assert status.models == {"test_provider": []}
        assert status.rag_indexes == []
        assert status.query_redactor_enabled is False
        assert status.mcp_servers == {}
        assert status.quota_management_enabled is False
        assert status.proxy_enabled is False

    def test_extract_config_status_with_rag_enabled(self):
        """Test extracting config status with RAG enabled."""
        config = create_minimal_config()
        config.ols_config.reference_content = ReferenceContent()
        index = ReferenceContentIndex()
        index.product_docs_index_id = "ocp-product-docs-4_17"
        config.ols_config.reference_content.indexes = [index]
        config.ols_config.reference_content.embeddings_model_path = Path(
            "/models/embeddings"
        )

        status = extract_config_status(config)

        assert status.rag_indexes == ["ocp-product-docs-4_17"]

    def test_extract_config_status_with_query_filters(self):
        """Test extracting config status with query filters enabled."""
        config = create_minimal_config()
        query_filter = QueryFilter()
        query_filter.name = "test_filter"
        query_filter.pattern = "test"
        query_filter.replace_with = "replacement"
        config.ols_config.query_filters = [query_filter]

        status = extract_config_status(config)

        assert status.query_redactor_enabled is True
        assert status.query_filter_count == 1

    def test_extract_config_status_with_mcp_servers(self):
        """Test extracting config status with MCP servers configured."""
        config = create_minimal_config()
        mcp_server = MCPServerConfig(
            name="test_server",
            transport="stdio",
            stdio=StdioTransportConfig(command="python", args=["test.py"]),
        )
        config.mcp_servers.servers = [mcp_server]

        status = extract_config_status(config)

        assert status.mcp_servers == {"test_server": "stdio"}

    def test_extract_config_status_with_provider_tls_config(self):
        """Test extracting config status with provider TLS security profile."""
        config = create_minimal_config()
        provider = config.llm_providers.providers["test_provider"]
        provider.tls_security_profile = TLSSecurityProfile()
        provider.tls_security_profile.profile_type = "Custom"

        status = extract_config_status(config)

        assert status.providers_with_tls_config == ["test_provider"]

    def test_extract_config_status_with_multiple_providers(self):
        """Test extracting config status with multiple providers and models."""
        config = create_minimal_config()

        provider1 = ProviderConfig()
        provider1.name = "provider1"
        provider1.type = "openai"
        provider1.models = {"model1": MagicMock(), "model2": MagicMock()}

        provider2 = ProviderConfig()
        provider2.name = "provider2"
        provider2.type = "azure_openai"
        provider2.models = {"model3": MagicMock()}

        config.llm_providers.providers = {
            "provider1": provider1,
            "provider2": provider2,
        }

        status = extract_config_status(config)

        assert status.providers == {
            "openai": ["provider1"],
            "azure_openai": ["provider2"],
        }
        assert status.models == {
            "provider1": ["model1", "model2"],
            "provider2": ["model3"],
        }


class TestStoreConfigStatus:
    """Tests for store_config_status function."""

    def test_store_config_status(self, tmpdir):
        """Test storing config status to filesystem."""
        storage_path = tmpdir.strpath
        status = ConfigStatus(
            providers={"openai": ["my_openai"]},
            models={"my_openai": ["gpt-4"]},
            rag_indexes=["ocp-docs-4_17", "user-docs-v1"],
            query_redactor_enabled=True,
            query_filter_count=1,
            providers_with_tls_config=["my_openai"],
            mcp_servers={"openshift": "stdio"},
            quota_management_enabled=False,
            token_history_enabled=False,
            proxy_enabled=False,
            extra_ca_count=0,
        )

        with patch("ols.src.config_status.config_status.suid.get_suid") as mock_suid:
            mock_suid.return_value = "test-uuid"
            with patch("ols.src.config_status.config_status.datetime") as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = (
                    "2024-01-01T00:00:00+00:00"
                )
                store_config_status(storage_path, status)

        config_status_file = Path(storage_path) / "test-uuid.json"
        assert config_status_file.exists()

        with open(config_status_file) as f:
            stored_data = json.load(f)

        assert stored_data["timestamp"] == "2024-01-01T00:00:00+00:00"
        assert stored_data["providers"] == {"openai": ["my_openai"]}
        assert stored_data["models"] == {"my_openai": ["gpt-4"]}
        assert stored_data["rag_indexes"] == ["ocp-docs-4_17", "user-docs-v1"]
        assert stored_data["mcp_servers"] == {"openshift": "stdio"}

    def test_store_config_status_creates_directory(self, tmpdir):
        """Test that store_config_status creates the storage directory if needed."""
        storage_path = str(Path(tmpdir) / "nested" / "path")
        status = ConfigStatus(
            providers={},
            models={},
            rag_indexes=[],
            query_redactor_enabled=False,
            query_filter_count=0,
            providers_with_tls_config=[],
            mcp_servers={},
            quota_management_enabled=False,
            token_history_enabled=False,
            proxy_enabled=False,
            extra_ca_count=0,
        )

        with patch("ols.src.config_status.config_status.suid.get_suid") as mock_suid:
            mock_suid.return_value = "test-uuid"
            store_config_status(storage_path, status)

        assert Path(storage_path).exists()
        assert (Path(storage_path) / "test-uuid.json").exists()
