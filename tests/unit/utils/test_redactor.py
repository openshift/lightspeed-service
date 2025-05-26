"""Unit test for the question filter."""

import re
from unittest import TestCase

from ols.utils.config import AppConfig
from ols.utils.redactor import Redactor, RegexFilter


class TestRedactor(TestCase):
    """Test the filter class."""

    def setUp(self):
        """Set up the test."""
        config = AppConfig()
        config.reload_from_yaml_file("tests/config/valid_config_with_query_filter.yaml")
        # make sure the query filters are specified in configuration
        assert config.ols_config.query_filters is not None
        # construct redactor class
        self.query_filter = Redactor(config.ols_config.query_filters)

    def test_redact_question_image_ip(self):
        """Test redact question with perfect word  and ip."""
        self.query_filter.regex_filters = [
            RegexFilter(re.compile(r"\b(?:image)\b"), "perfect_word", "REDACTED_image"),
            RegexFilter(
                re.compile(r"(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"),
                "ip_address",
                "REDACTED_IP",
            ),
        ]
        query = (
            "write a deployment yaml for the mongodb image with nodeip as 1.123.0.99"
        )
        redacted_question = self.query_filter.redact("test_id", query)
        expected_output = (
            "write a deployment yaml for the mongodb REDACTED_image with nodeip "
            "as REDACTED_IP"
        )
        assert redacted_question == expected_output

    def test_redact_question_mongopart_url_phone(self):
        """Test redact question with partial_word, url and phone number."""
        self.query_filter.regex_filters = [
            RegexFilter(re.compile(r"(?:mongo)"), "any_string_match", "REDACTED_MONGO"),
            RegexFilter(
                re.compile(r"(?:https?://)?(?:www\.)?[\w\.-]+\.\w+"),
                "url",
                "",
            ),
            RegexFilter(
                re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
                "phone_number",
                "REDACTED_PHONE_NUMBER",
            ),
        ]
        query = "write a deployment yaml for\
        the mongodb image from www.mongodb.com and call me at 123-456-7890"
        redacted_question = self.query_filter.redact("test_id", query)
        expected_output = "write a deployment yaml for\
        the REDACTED_MONGOdb image from  and call me at REDACTED_PHONE_NUMBER"
        assert redacted_question == expected_output

    def test_redact_query_with_empty_filters(self):
        """Test redact query with empty filters."""
        query = "write a deployment yaml for the mongodb image"
        self.query_filter.regex_filters = []
        redacted_query = self.query_filter.redact("test_id", query)
        assert redacted_query == query
