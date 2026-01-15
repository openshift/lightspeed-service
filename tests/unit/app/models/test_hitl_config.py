"""Unit tests for HITL configuration model."""

import pytest
from pydantic import ValidationError

from ols.app.models.config import HITLConfig


class TestHITLConfig:
    """Tests for HITLConfig model."""

    def test_default_config(self):
        """Test default HITL configuration values."""
        config = HITLConfig()

        assert config.enabled is False
        assert config.approval_timeout == 300
        assert config.tools_requiring_approval == []
        assert config.auto_approve_read_only is True
        assert config.default_on_timeout == "reject"

    def test_enabled_config(self):
        """Test enabled HITL configuration."""
        config = HITLConfig(enabled=True)

        assert config.enabled is True

    def test_custom_timeout(self):
        """Test custom approval timeout."""
        config = HITLConfig(approval_timeout=60)

        assert config.approval_timeout == 60

    def test_specific_tools_requiring_approval(self):
        """Test specifying tools that require approval."""
        config = HITLConfig(
            tools_requiring_approval=["kubectl_apply", "delete_pod"]
        )

        assert config.tools_requiring_approval == ["kubectl_apply", "delete_pod"]

    def test_auto_approve_read_only_disabled(self):
        """Test disabling auto-approve for read-only tools."""
        config = HITLConfig(auto_approve_read_only=False)

        assert config.auto_approve_read_only is False

    def test_default_on_timeout_approve(self):
        """Test auto-approve on timeout."""
        config = HITLConfig(default_on_timeout="approve")

        assert config.default_on_timeout == "approve"

    def test_invalid_default_on_timeout(self):
        """Test that invalid default_on_timeout raises error."""
        with pytest.raises(ValidationError):
            HITLConfig(default_on_timeout="invalid")

    def test_equality(self):
        """Test equality comparison."""
        config1 = HITLConfig(enabled=True, approval_timeout=60)
        config2 = HITLConfig(enabled=True, approval_timeout=60)
        config3 = HITLConfig(enabled=False, approval_timeout=60)

        assert config1 == config2
        assert config1 != config3

    def test_equality_with_non_hitl_config(self):
        """Test equality comparison with non-HITLConfig object."""
        config = HITLConfig()

        assert config != "not a config"
        assert config != None  # noqa: E711
        assert config != {}

    def test_full_custom_config(self):
        """Test creating a fully customized configuration."""
        config = HITLConfig(
            enabled=True,
            approval_timeout=120,
            tools_requiring_approval=["dangerous_tool", "another_tool"],
            auto_approve_read_only=False,
            default_on_timeout="approve",
        )

        assert config.enabled is True
        assert config.approval_timeout == 120
        assert config.tools_requiring_approval == ["dangerous_tool", "another_tool"]
        assert config.auto_approve_read_only is False
        assert config.default_on_timeout == "approve"



