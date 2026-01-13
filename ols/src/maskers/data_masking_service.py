"""Data masking service for sensitive MCP server data.

This module provides the core data masking functionality to prevent
sensitive data from reaching the LLM, logging, and storage systems.
"""

import importlib
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Pattern

from ols.app.models.masking_config import MaskingConfig, MaskingPattern
from ols.src.maskers.base_masker import BaseMasker
from ols.src.maskers.builtin_config import (
    BUILTIN_CODE_MASKERS,
    BUILTIN_MASKING_PATTERNS,
    BUILTIN_PATTERN_GROUPS,
)

logger = logging.getLogger(__name__)


class DataMaskingService:
    """Service for masking sensitive data in MCP server responses.

    This service applies both regex patterns and code-based maskers to mask
    sensitive information before it reaches the LLM, logging, or storage systems.

    Code-based maskers are applied first (more specific), followed by regex
    patterns (more general).
    """

    def __init__(
        self,
        get_server_masking_config: Optional[Callable[[str], Optional[MaskingConfig]]] = None,
    ):
        """Initialize the data masking service.

        Args:
            get_server_masking_config: Callback function to retrieve masking config
                for a given server name. If None, masking will be disabled for all servers.
        """
        self._get_server_masking_config = get_server_masking_config
        self.compiled_patterns: Dict[str, Pattern[str]] = {}
        self.custom_pattern_metadata: Dict[str, Dict[str, str]] = {}
        self.code_based_maskers: Dict[str, BaseMasker] = {}
        self._load_builtin_patterns()
        self._load_builtin_maskers()

        logger.info("DataMaskingService initialized")

    def _load_builtin_patterns(self) -> None:
        """Load and compile built-in regex patterns.

        This method compiles all built-in patterns for performance.
        Patterns that fail to compile will be logged and skipped.
        """
        logger.debug("Loading built-in masking patterns")

        for pattern_name, pattern_config in BUILTIN_MASKING_PATTERNS.items():
            try:
                compiled_pattern = re.compile(
                    pattern_config["pattern"], re.DOTALL | re.MULTILINE
                )
                self.compiled_patterns[pattern_name] = compiled_pattern
                logger.debug(f"Compiled pattern: {pattern_name}")
            except re.error as e:
                logger.error(f"Failed to compile built-in pattern '{pattern_name}': {e}")

        logger.info(f"Loaded {len(self.compiled_patterns)} built-in patterns")

    def _load_builtin_maskers(self) -> None:
        """Load and instantiate built-in code-based maskers.

        This method dynamically imports and instantiates code-based maskers
        registered in BUILTIN_CODE_MASKERS. Maskers that fail to load will
        be logged and skipped.
        """
        logger.debug("Loading built-in code-based maskers")

        for masker_name, import_path in BUILTIN_CODE_MASKERS.items():
            try:
                # Parse import path: "module.path.ClassName"
                module_path, class_name = import_path.rsplit(".", 1)

                # Dynamically import the module
                module = importlib.import_module(module_path)

                # Get the class
                masker_class = getattr(module, class_name)

                # Verify it's a BaseMasker subclass before instantiation
                if not issubclass(masker_class, BaseMasker):
                    logger.error(
                        f"Masker '{masker_name}' from '{import_path}' is not a BaseMasker subclass"
                    )
                    continue

                # Instantiate the masker
                masker_instance = masker_class()

                # Validate that the masker's name() matches the registry key
                actual_name = masker_instance.name()
                if actual_name != masker_name:
                    logger.warning(
                        f"Masker name mismatch: registry key is '{masker_name}' but "
                        f"masker.name() returns '{actual_name}'. This could indicate "
                        f"configuration drift. Skipping this masker."
                    )
                    continue

                # Register the masker
                self.code_based_maskers[masker_name] = masker_instance
                logger.debug(f"Loaded code-based masker: {masker_name}")

            except (ImportError, AttributeError, Exception) as e:
                logger.error(
                    f"Failed to load built-in masker '{masker_name}' from '{import_path}': "
                    f"{type(e).__name__}: {e}"
                )

        logger.info(f"Loaded {len(self.code_based_maskers)} built-in code-based maskers")

    def _compile_and_add_custom_patterns(
        self, custom_patterns: List[MaskingPattern]
    ) -> List[str]:
        """Compile custom patterns and add them to the compiled patterns dictionary.

        Args:
            custom_patterns: List of MaskingPattern objects to compile and add

        Returns:
            List of pattern names that were successfully compiled and added
        """
        compiled_pattern_names = []

        for custom_pattern in custom_patterns:
            if not custom_pattern.enabled:
                logger.debug(f"Skipping disabled custom pattern: {custom_pattern.name}")
                continue

            try:
                compiled_pattern = re.compile(
                    custom_pattern.pattern, re.DOTALL | re.MULTILINE
                )

                # Store both compiled pattern and replacement text
                # Use a unique key to avoid conflicts with built-in patterns
                pattern_key = f"custom_{custom_pattern.name}"
                self.compiled_patterns[pattern_key] = compiled_pattern

                # Store custom pattern metadata for replacement lookup
                self.custom_pattern_metadata[pattern_key] = {
                    "replacement": custom_pattern.replacement,
                    "description": custom_pattern.description,
                }

                compiled_pattern_names.append(pattern_key)
                logger.debug(
                    f"Compiled custom pattern: {custom_pattern.name} -> {pattern_key}"
                )

            except re.error as e:
                logger.error(
                    f"Failed to compile custom pattern '{custom_pattern.name}': {e}"
                )
                continue

        logger.debug(f"Successfully compiled {len(compiled_pattern_names)} custom patterns")
        return compiled_pattern_names

    def mask_data(self, data: Dict[str, Any], pattern_group: str = "security") -> Dict[str, Any]:
        """Mask sensitive data in data payloads using specified pattern group.

        This method provides standalone data masking without requiring
        MCP server configuration. It's used for masking data
        at the API entry point.

        Uses the same core masking mechanism as mask_response() to apply
        both code-based maskers and regex patterns to data structures.

        Args:
            data: The data dictionary to mask
            pattern_group: Pattern group name (default: "security")
                          Available groups: basic, secrets, security, kubernetes, all

        Returns:
            Masked data dictionary with sensitive information replaced

        Raises:
            ValueError: If pattern_group is not recognized
        """
        logger.debug(f"mask_data called with pattern_group: {pattern_group}")

        try:
            # Validate pattern group exists
            if pattern_group not in BUILTIN_PATTERN_GROUPS:
                available_groups = list(BUILTIN_PATTERN_GROUPS.keys())
                error_msg = (
                    f"Unknown pattern group '{pattern_group}'. Available: {available_groups}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Expand pattern group to individual patterns
            patterns = self._expand_pattern_groups([pattern_group])

            if not patterns:
                logger.warning(
                    f"Pattern group '{pattern_group}' expanded to no patterns - returning data unchanged"
                )
                return data

            logger.debug(
                f"Applying {len(patterns)} patterns from group '{pattern_group}' to data"
            )

            # Use the same core masking mechanism as mask_response()
            masked_data = self._mask_data_structure(data, patterns)

            logger.debug(f"Data masking completed for pattern group '{pattern_group}'")
            return masked_data

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Log error but don't fail - return original data for reliability
            logger.error(
                f"Error during data masking with pattern group '{pattern_group}': {e}",
                exc_info=True,
            )
            logger.warning(
                "Returning original data due to masking error (fail-open for reliability)"
            )
            return data

    def mask_response(self, response: Dict[str, Any], server_name: str) -> Dict[str, Any]:
        """Apply server-specific masking patterns to response data.

        This method looks up the masking configuration for the specified server
        and applies all configured patterns to the response data.

        Args:
            response: The response data from the MCP server
            server_name: Name of the MCP server that generated the response

        Returns:
            The response data with sensitive information masked
        """
        logger.debug(f"mask_response called for server: {server_name}")

        try:
            # Step 1: Get masking configuration for the server
            masking_config = self._get_masking_config_for_server(server_name)
            if not masking_config or not masking_config.enabled:
                logger.debug(f"Masking disabled for server: {server_name}")
                return response

            # Step 2: Expand pattern groups to individual patterns
            all_patterns: List[str] = []
            if masking_config.pattern_groups:
                expanded_patterns = self._expand_pattern_groups(
                    masking_config.pattern_groups
                )
                all_patterns.extend(expanded_patterns)

            # Step 3: Add individual patterns
            if masking_config.patterns:
                all_patterns.extend(masking_config.patterns)

            # Step 4: Add custom patterns
            custom_pattern_names: List[str] = []
            if masking_config.custom_patterns:
                custom_pattern_names = self._compile_and_add_custom_patterns(
                    masking_config.custom_patterns
                )
                logger.debug(
                    f"Compiled and added {len(custom_pattern_names)} custom patterns: "
                    f"{custom_pattern_names}"
                )

            # Combine all patterns (built-in + custom)
            all_patterns.extend(custom_pattern_names)

            if not all_patterns:
                logger.debug(f"No patterns configured for server: {server_name}")
                return response

            # Remove duplicates while preserving order
            unique_patterns: List[str] = []
            seen: set[str] = set()
            for pattern in all_patterns:
                if pattern not in seen:
                    unique_patterns.append(pattern)
                    seen.add(pattern)

            logger.debug(
                f"Applying {len(unique_patterns)} patterns to response for server: {server_name}"
            )

            # Step 5: Apply masking to the response data
            masked_response = self._mask_data_structure(response, unique_patterns)

            logger.debug(f"Masking completed for server: {server_name}")
            return masked_response

        except Exception as e:
            logger.error(f"Error during masking for server '{server_name}': {e}")
            # Fail-safe behavior: mask the entire response content
            logger.warning(f"Applying fail-safe masking for server: {server_name}")
            return self._apply_failsafe_masking(response)

    def _mask_data_structure(self, data: Any, patterns: List[str]) -> Any:
        """Recursively traverse and mask data structures.

        Args:
            data: The data structure to mask (can be dict, list, str, or other types)
            patterns: List of pattern names to apply

        Returns:
            The data structure with sensitive information masked
        """
        if isinstance(data, dict):
            # Recursively mask dictionary values
            masked_dict = {}
            for key, value in data.items():
                masked_dict[key] = self._mask_data_structure(value, patterns)
            return masked_dict

        elif isinstance(data, list):
            # Recursively mask list elements
            return [self._mask_data_structure(item, patterns) for item in data]

        elif isinstance(data, str):
            # Apply patterns to string content
            return self._apply_patterns(data, patterns)

        else:
            # For other types (int, float, bool, None), return unchanged
            return data

    def _apply_failsafe_masking(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fail-safe masking when normal masking fails.

        Args:
            response: The original response data

        Returns:
            A response with all string content masked
        """
        logger.debug("Applying fail-safe masking")

        try:
            # Convert response to string and mask it entirely
            json.dumps(response, default=str)
            masked_content = "__MASKED_ERROR__"

            # Try to preserve the basic response structure
            if "result" in response:
                return {"result": masked_content}
            else:
                return {"masked_response": masked_content}

        except Exception as e:
            logger.error(f"Error in fail-safe masking: {e}")
            # Ultimate fail-safe
            return {"result": "__MASKED_ERROR__"}

    def _apply_patterns(self, text: str, patterns: List[str]) -> str:
        """Apply code-based maskers and regex patterns to mask text content.

        This method applies masking in two phases:
        1. Code-based maskers (more specific, structural awareness)
        2. Regex patterns (more general, pattern matching)

        Args:
            text: The text content to mask
            patterns: List of pattern names to apply

        Returns:
            The text with sensitive information masked
        """
        logger.debug(
            f"_apply_patterns called with {len(patterns)} patterns on text length {len(text)}"
        )

        if not text or not patterns:
            return text

        masked_text = text
        patterns_applied = 0

        # Phase 1: Apply code-based maskers (more specific)
        for pattern_name in patterns:
            if pattern_name in self.code_based_maskers:
                masker = self.code_based_maskers[pattern_name]
                try:
                    if masker.applies_to(masked_text):
                        previous_text = masked_text
                        masked_text = masker.mask(masked_text)

                        if masked_text != previous_text:
                            patterns_applied += 1
                            logger.debug(
                                f"Code-based masker '{pattern_name}' applied - masking performed"
                            )
                        else:
                            logger.debug(
                                f"Code-based masker '{pattern_name}' applied - no changes made"
                            )
                except Exception as e:
                    logger.error(
                        f"Error in code-based masker '{pattern_name}': {e}", exc_info=True
                    )
                    # Continue with other patterns rather than failing completely
                    continue

        # Phase 2: Apply regex patterns (more general)
        for pattern_name in patterns:
            # Skip patterns that are code-based only (not regex patterns)
            if (
                pattern_name in self.code_based_maskers
                and pattern_name not in self.compiled_patterns
            ):
                continue

            if pattern_name not in self.compiled_patterns:
                logger.warning(
                    f"Pattern '{pattern_name}' not found in compiled patterns - skipping"
                )
                continue

            # Get replacement text (built-in vs custom patterns)
            if pattern_name.startswith("custom_"):
                # Custom pattern - get replacement from metadata
                if pattern_name not in self.custom_pattern_metadata:
                    logger.warning(
                        f"Custom pattern '{pattern_name}' metadata not found - skipping"
                    )
                    continue
                replacement = self.custom_pattern_metadata[pattern_name]["replacement"]
            else:
                # Built-in pattern - get replacement from builtin patterns
                if pattern_name not in BUILTIN_MASKING_PATTERNS:
                    logger.warning(
                        f"Built-in pattern '{pattern_name}' not found in builtin patterns - skipping"
                    )
                    continue
                replacement = BUILTIN_MASKING_PATTERNS[pattern_name]["replacement"]

            try:
                compiled_pattern = self.compiled_patterns[pattern_name]

                # Apply the pattern with error handling
                previous_text = masked_text
                masked_text = compiled_pattern.sub(replacement, masked_text)

                if masked_text != previous_text:
                    patterns_applied += 1
                    logger.debug(f"Pattern '{pattern_name}' applied - masking performed")
                else:
                    logger.debug(f"Pattern '{pattern_name}' applied - no matches found")

            except Exception as e:
                logger.error(f"Error applying pattern '{pattern_name}': {e}")
                # Continue with other patterns rather than failing completely
                continue

        logger.debug(
            f"Pattern application complete - {patterns_applied}/{len(patterns)} patterns had matches"
        )
        return masked_text

    def _expand_pattern_groups(self, pattern_groups: List[str]) -> List[str]:
        """Expand pattern group names to individual pattern names.

        Args:
            pattern_groups: List of pattern group names to expand

        Returns:
            List of individual pattern names from all specified groups
        """
        logger.debug(f"_expand_pattern_groups called with groups: {pattern_groups}")

        expanded_patterns: List[str] = []
        for group_name in pattern_groups:
            if group_name in BUILTIN_PATTERN_GROUPS:
                group_patterns = BUILTIN_PATTERN_GROUPS[group_name]
                expanded_patterns.extend(group_patterns)
                logger.debug(f"Expanded group '{group_name}' to patterns: {group_patterns}")
            else:
                logger.warning(f"Unknown pattern group '{group_name}' - skipping")

        # Remove duplicates while preserving order
        unique_patterns: List[str] = []
        seen: set[str] = set()
        for pattern in expanded_patterns:
            if pattern not in seen:
                unique_patterns.append(pattern)
                seen.add(pattern)

        logger.debug(f"Final expanded patterns: {unique_patterns}")
        return unique_patterns

    def _get_masking_config_for_server(self, server_name: str) -> Optional[MaskingConfig]:
        """Get masking configuration for a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            The masking configuration for the server, or None if not configured
        """
        logger.debug(f"_get_masking_config_for_server called for server: {server_name}")

        if not self._get_server_masking_config:
            logger.debug("No config callback available - masking disabled")
            return None

        try:
            masking_config = self._get_server_masking_config(server_name)
            if not masking_config:
                logger.debug(f"No masking configuration found for server: {server_name}")
                return None

            logger.debug(
                f"Found masking configuration for server '{server_name}': "
                f"enabled={masking_config.enabled}"
            )
            return masking_config

        except Exception as e:
            logger.error(
                f"Error retrieving masking config for server '{server_name}': {e}"
            )
            return None

