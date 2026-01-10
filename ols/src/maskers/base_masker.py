"""Base masker interface for code-based data masking.

This module defines the abstract interface that all code-based maskers must implement.
Code-based maskers complement regex patterns by providing structural awareness and
context-sensitive masking logic for complex data formats.
"""

from abc import ABC, abstractmethod


class BaseMasker(ABC):
    """Abstract base class for code-based data maskers.

    Code-based maskers provide sophisticated masking logic that goes beyond
    simple regex pattern matching. They can parse structured data (YAML, JSON),
    understand context, and apply intelligent masking rules.

    Example:
        class MyMasker(BaseMasker):
            def name(self) -> str:
                return "my_masker"

            def applies_to(self, data: str) -> bool:
                return "custom_marker" in data

            def mask(self, data: str) -> str:
                # Apply custom masking logic
                return masked_data

    Note:
        Maskers should be defensive and fail gracefully. If parsing or masking
        fails, return the original data rather than raising an exception.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this masker.

        The name is used to reference this masker in configuration files
        and pattern groups. It should be descriptive and unique across
        all maskers.

        Returns:
            A unique string identifier (e.g., "kubernetes_secret")
        """
        pass

    @abstractmethod
    def applies_to(self, data: str) -> bool:
        """Check if this masker should process the given data.

        This method performs a lightweight check to determine if the masker
        is applicable to the provided data. It should be fast and avoid
        expensive parsing operations when possible.

        Args:
            data: The input data to check for applicability

        Returns:
            True if this masker should process the data, False otherwise

        Note:
            This method should never raise exceptions. If uncertain,
            return False to skip masking.
        """
        pass

    @abstractmethod
    def mask(self, data: str) -> str:
        """Apply masking logic to the data and return the masked result.

        This method performs the actual masking operation. It should:
        - Parse the data structure if needed
        - Identify sensitive fields or patterns
        - Replace sensitive content with masked values
        - Preserve structure and formatting where possible
        - Handle errors gracefully (return original data on failure)

        Args:
            data: The input data to mask

        Returns:
            The masked data with sensitive information replaced

        Note:
            This method should be defensive and handle malformed input.
            If masking fails, return the original data rather than
            raising an exception to maintain system reliability.
        """
        pass

