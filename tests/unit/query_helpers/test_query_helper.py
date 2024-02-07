"""Unit tests for query helper class."""

from ols.src.query_helpers import QueryHelper
from ols.utils import config


class TestQueryHelper:
    """Test the query helper class."""

    def test_defaults_used(self):
        """Test that the defaults are used when no inputs are provided."""
        config.init_config("tests/config/valid_config.yaml")

        qh = QueryHelper()

        assert qh.provider == config.ols_config.default_provider
        assert qh.model == config.ols_config.default_model

    def test_inputs_are_used(self):
        """Test that the inputs are used when provided."""
        test_provider = "test_provider"
        test_model = "test_model"
        qh = QueryHelper(provider=test_provider, model=test_model)

        assert qh.provider == test_provider
        assert qh.model == test_model
