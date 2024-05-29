"""A class helps redact the question based on the regex filters provided in the config file."""

import logging
import re
from collections import namedtuple

from ols.app.models.config import QueryFilter

logger = logging.getLogger(__name__)

RegexFilter = namedtuple("RegexFilter", "pattern, name, replace_with")


# TODO: OLS-380 Config object mirrors configuration


class QueryFilters:
    """Redact the query based on the regex filters provided in the config file."""

    def __init__(self, query_filters: list[QueryFilter]) -> None:
        """Initialize the class instance."""
        regex_filters: list[RegexFilter] = []
        self.regex_filters = regex_filters
        logger.debug(f"Query filters : {query_filters}")
        if not query_filters:
            return
        for query_filter in query_filters:
            pattern = re.compile(str(query_filter.pattern))
            regex_filters.append(
                RegexFilter(
                    pattern=pattern,
                    name=query_filter.name,
                    replace_with=query_filter.replace_with,
                )
            )
        self.regex_filters = regex_filters

    def redact_query(self, conversation_id: str, query: str) -> str:
        """Redact the query using regex built."""
        logger.debug(f"Redacting conversation {conversation_id} query: {query}")
        for query_filter in self.regex_filters:
            query, count = query_filter.pattern.subn(query_filter.replace_with, query)
            logger.debug(f"Replaced: {count} matched with filter : {query_filter.name}")
        logger.debug(f"Redacted conversation {conversation_id} query: {query}")
        return query
