"""Masking configuration data models.

This module defines Pydantic models for validating data masking configurations
that are applied to MCP server request/response flows to protect sensitive data.
"""

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class MaskingPattern(BaseModel):
    """A single masking pattern configuration.

    Defines a regex pattern to match sensitive data and the replacement
    text to use when masking.
    """

    name: str = Field(..., description="Pattern identifier (e.g., 'security_token')")
    pattern: str = Field(..., description="Regex pattern for matching sensitive data")
    replacement: str = Field(..., description="Replacement text for matches")
    description: str = Field(..., description="Human-readable description of the pattern")
    enabled: bool = Field(True, description="Whether pattern is active")

    @field_validator("pattern")
    @classmethod
    def validate_regex_pattern(cls, v: str) -> str:
        """Validate that the pattern is a valid regex.

        Args:
            v: The regex pattern string

        Returns:
            The validated pattern string

        Raises:
            ValueError: If the pattern is not a valid regex
        """
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the pattern name is not empty.

        Args:
            v: The pattern name

        Returns:
            The validated pattern name

        Raises:
            ValueError: If the name is empty or only whitespace
        """
        if not v or not v.strip():
            raise ValueError("Pattern name cannot be empty")
        return v.strip()


class MaskingConfig(BaseModel):
    """Configuration for data masking on a specific MCP server.

    Defines which patterns and pattern groups to apply when masking
    sensitive data from MCP server responses.
    """

    enabled: bool = Field(True, description="Whether masking is enabled for this server")
    pattern_groups: List[str] = Field(
        default_factory=list, description="List of built-in pattern group names to apply"
    )
    patterns: List[str] = Field(
        default_factory=list, description="List of built-in pattern names to apply"
    )
    custom_patterns: Optional[List[MaskingPattern]] = Field(
        None, description="Server-specific custom patterns"
    )

    @field_validator("pattern_groups")
    @classmethod
    def validate_pattern_groups(cls, v: List[str]) -> List[str]:
        """Validate pattern group names are not empty.

        Args:
            v: List of pattern group names

        Returns:
            The validated list of pattern group names

        Raises:
            ValueError: If any group name is empty
        """
        validated_groups = []
        for group in v:
            if not group or not group.strip():
                raise ValueError("Pattern group name cannot be empty")
            validated_groups.append(group.strip())
        return validated_groups

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v: List[str]) -> List[str]:
        """Validate pattern names are not empty.

        Args:
            v: List of pattern names

        Returns:
            The validated list of pattern names

        Raises:
            ValueError: If any pattern name is empty
        """
        validated_patterns = []
        for pattern in v:
            if not pattern or not pattern.strip():
                raise ValueError("Pattern name cannot be empty")
            validated_patterns.append(pattern.strip())
        return validated_patterns

