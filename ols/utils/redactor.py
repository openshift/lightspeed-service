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
        logger.debug(f"Filters : {filters}")
        if not filters:
            return
        for filter in filters:
            pattern = re.compile(str(filter.pattern))
            regex_filters.append(
                RegexFilter(
                    pattern=pattern,
                    name=filter.name,
                    replace_with=filter.replace_with,
                )
            )
        self.regex_filters = regex_filters

    def redact(self, conversation_id: str, input: str) -> str:
        """Redact the input using regex built."""
        logger.debug(f"Redacting conversation {conversation_id} input: {input}")
        for filter in self.regex_filters:
            input, count = filter.pattern.subn(filter.replace_with, input)
            logger.debug(f"Replaced: {count} matched with filter : {filter.name}")
        logger.debug(f"Redacted conversation {conversation_id} input: {input}")
        return input
