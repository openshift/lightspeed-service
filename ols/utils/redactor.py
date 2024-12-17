"""A class helps redact the question based on the regex filters provided in the config file."""

import logging
import re
from collections import namedtuple

from ols.app.models.config import QueryFilter

logger = logging.getLogger(__name__)

RegexFilter = namedtuple("RegexFilter", "pattern, name, replace_with")


# TODO: OLS-380 Config object mirrors configuration


class Redactor:
    """Redact the input based on the regex filters provided in the config file."""

    def __init__(self, filters: list[QueryFilter]) -> None:
        """Initialize the class instance."""
        regex_filters: list[RegexFilter] = []
        self.regex_filters = regex_filters
        logger.debug("Filters : %s", filters)
        if not filters:
            return
        for regex_filter in filters:
            pattern = re.compile(str(regex_filter.pattern))
            regex_filters.append(
                RegexFilter(
                    pattern=pattern,
                    name=regex_filter.name,
                    replace_with=regex_filter.replace_with,
                )
            )
        self.regex_filters = regex_filters

    def redact(self, conversation_id: str, text_input: str) -> str:
        """Redact the input using regex built."""
        logger.debug("Redacting conversation %s", conversation_id)
        for regex_filter in self.regex_filters:
            text_input, count = regex_filter.pattern.subn(
                regex_filter.replace_with, text_input
            )
            logger.debug(
                "Replaced: %d matched with filter: %s", count, regex_filter.name
            )
        logger.debug("Redacted conversation %s input: %s", conversation_id, text_input)
        return text_input
