"""Kubernetes Secret masker for comprehensive secret data masking.

This module provides code-based masking for Kubernetes Secret resources,
handling both YAML and JSON formats, including complex nested structures
like last-applied-configuration annotations.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

import yaml

from ols.src.maskers.base_masker import BaseMasker

logger = logging.getLogger(__name__)


class KubernetesSecretMasker(BaseMasker):
    """Code-based masker for Kubernetes Secret resources.

    This masker handles Kubernetes Secrets in multiple formats:
    - YAML format with data: sections
    - JSON format with "data" objects
    - Nested JSON in annotations (e.g., last-applied-configuration)
    - Both data and stringData fields

    It intelligently distinguishes between Secrets (which should be masked)
    and ConfigMaps (which should not be masked), even when they appear in
    the same document or nested structures.
    """

    MASKED_VALUE = "__MASKED_SECRET_DATA__"

    def name(self) -> str:
        """Return the unique identifier for this masker."""
        return "kubernetes_secret"

    def applies_to(self, data: str) -> bool:
        """Check if this masker should process the given data.

        Returns True only if the data appears to contain Kubernetes Secret resources.
        ConfigMaps are completely ignored.
        """
        if not data:
            return False

        # Check for Kubernetes Secret patterns:
        # YAML: kind: Secret
        # JSON: "kind": "Secret" or "kind":"Secret"
        yaml_pattern = r"\bkind:\s*Secret\b"
        json_pattern = r'"kind"\s*:\s*"Secret"'

        return bool(re.search(yaml_pattern, data) or re.search(json_pattern, data))

    def mask(self, data: str) -> str:
        """Apply masking logic to Kubernetes Secret resources.

        This method:
        1. Detects the format (YAML or JSON)
        2. Parses the content
        3. Identifies Secret resources (not ConfigMaps)
        4. Masks data and stringData fields
        5. Recursively processes nested structures (annotations)
        """
        if not data:
            return data

        try:
            # Try YAML format first (most common for kubectl output)
            masked = self._mask_yaml_format(data)
            if masked != data:
                return masked

            # Try pure JSON format
            masked = self._mask_json_format(data)
            if masked != data:
                return masked

            # If neither worked, return original
            logger.debug("KubernetesSecretMasker: No masking applied (not a recognized format)")
            return data

        except Exception as e:
            logger.error(f"Error in KubernetesSecretMasker.mask: {e}", exc_info=True)
            # Fail-safe: return original data
            return data

    def _mask_yaml_format(self, data: str) -> str:
        """Mask Kubernetes Secrets in YAML format.

        Handles both simple YAML and YAML containing JSON in annotations.
        """
        try:
            # Parse YAML to check what we have and identify Secret resources
            docs = list(yaml.safe_load_all(data))
            if not docs:
                return data

            # Check if we have any Secrets to mask
            secret_indices = [
                i
                for i, doc in enumerate(docs)
                if isinstance(doc, dict) and doc.get("kind") == "Secret"
            ]

            if not secret_indices:
                # No secrets to mask, but might have JSON in annotations
                # Try to mask JSON within the YAML text
                return self._mask_json_in_text(data)

            # We have secrets - use regex to find and mask data: sections in Secrets only
            # Split by document separator to process each YAML doc
            doc_texts = data.split("\n---\n")

            # Safety check: ensure parsed docs count matches split docs count
            if len(doc_texts) != len(docs):
                logger.warning(
                    f"KubernetesSecretMasker: Document count mismatch - "
                    f"parsed {len(docs)} YAML docs but split into {len(doc_texts)} text segments. "
                    f"Applying conservative fallback masking to avoid silent skips."
                )
                # Conservative fallback: mask all splits at secret indices
                masked_docs = []
                for i, doc_text in enumerate(doc_texts):
                    if i in secret_indices:
                        # This index corresponds to a Secret, mask it
                        doc_text = self._mask_yaml_secret_data_sections(doc_text)
                    masked_docs.append(doc_text)

                masked_data = "\n---\n".join(masked_docs)
                masked_data = self._mask_json_in_text(masked_data)
                return masked_data

            masked_docs = []
            for i, doc_text in enumerate(doc_texts):
                # Check if this document index corresponds to a Secret
                if i in secret_indices:
                    # This is a Secret, mask its data: and stringData: sections
                    doc_text = self._mask_yaml_secret_data_sections(doc_text)

                masked_docs.append(doc_text)

            masked_data = "\n---\n".join(masked_docs)

            # Also mask any JSON in annotations (like last-applied-configuration)
            masked_data = self._mask_json_in_text(masked_data)

            return masked_data

        except yaml.YAMLError:
            # Not valid YAML, try other formats
            return data
        except Exception as e:
            logger.error(f"Error in _mask_yaml_format: {e}", exc_info=True)
            return data

    def _mask_yaml_secret_data_sections(self, yaml_text: str) -> str:
        """Mask data: and stringData: sections in a YAML Secret.

        Args:
            yaml_text: YAML text of a single Secret resource

        Returns:
            YAML text with data sections masked
        """
        lines = yaml_text.split("\n")
        result_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for data: or stringData: at the start of a line (field name)
            data_match = re.match(r"^(data|stringData):\s*$", line)
            if data_match:
                # Found data section, mask it
                result_lines.append(line.rstrip())
                i += 1

                # Determine indent level from first non-blank line in the data section
                indent = " "  # Default to single space (standard kubectl output)
                start_i = i
                while i < len(lines):
                    next_line = lines[i]
                    # Look for first indented line to capture indent
                    if next_line and next_line[:1] in (" ", "\t"):
                        # Calculate indent from this line
                        indent_len = len(next_line) - len(next_line.lstrip())
                        indent = next_line[:indent_len]
                        break
                    elif not next_line.strip():
                        # Blank line, continue searching
                        i += 1
                    else:
                        # Non-indented line, data section ended (empty data section)
                        break

                # Reset to start and skip all indented or blank lines
                i = start_i
                while i < len(lines):
                    next_line = lines[i]
                    # Indented or blank line → still part of the data section
                    if (next_line and next_line[:1] in (" ", "\t")) or not next_line.strip():
                        i += 1
                    else:
                        # Non-indented line, data section ended
                        break

                # Add masked value with preserved indentation
                result_lines.append(f"{indent}{self.MASKED_VALUE}")
            else:
                result_lines.append(line)
                i += 1

        return "\n".join(result_lines)

    def _mask_json_format(self, data: str) -> str:
        """Mask Kubernetes Secrets in pure JSON format."""
        try:
            # Try to parse as JSON
            obj = json.loads(data)

            # Mask if it's a Secret
            if isinstance(obj, dict):
                masked_obj = self._mask_secret_object(obj)
                return json.dumps(masked_obj, separators=(",", ":"))

            return data

        except json.JSONDecodeError:
            # Not valid JSON
            return data
        except Exception as e:
            logger.error(f"Error in _mask_json_format: {e}", exc_info=True)
            return data

    def _mask_secret_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask Secret objects in dictionaries.

        This handles nested structures like annotations containing JSON.
        """
        if not isinstance(obj, dict):
            return obj

        # Check if this is a Secret resource
        is_secret = obj.get("kind") == "Secret"

        result = {}
        for key, value in obj.items():
            if is_secret and key in ("data", "stringData"):
                # Mask the entire data or stringData object
                result[key] = self.MASKED_VALUE
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = self._mask_secret_object(value)
            elif isinstance(value, str):
                # Check if this string contains JSON with Secrets
                result[key] = self._mask_json_in_text(value)
            elif isinstance(value, list):
                # Process lists
                result[key] = [
                    self._mask_secret_object(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def _mask_json_in_text(self, text: str) -> str:
        """Find and mask JSON objects containing Secrets within text.

        This handles cases like JSON in YAML annotations.
        """
        if not text or "Secret" not in text:
            return text

        # Pattern to find JSON objects (simplified)
        # Look for {"...kind":"Secret"...}
        json_pattern = r'\{[^{}]*"kind"\s*:\s*"Secret"[^{}]*\}'

        def mask_json_match(match: re.Match) -> str:
            """Mask a JSON Secret object found in text."""
            json_str = match.group(0)
            try:
                obj = json.loads(json_str)
                masked_obj = self._mask_secret_object(obj)
                return json.dumps(masked_obj, separators=(",", ":"))
            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"Failed to parse JSON in text: {e}")
                return json_str

        # Try simple pattern first
        masked = re.sub(json_pattern, mask_json_match, text)

        # Handle more complex nested JSON (with nested braces)
        # This is a more sophisticated approach for deeply nested structures
        if "data" in masked and "Secret" in masked:
            masked = self._mask_nested_json_in_text(masked)

        return masked

    def _mask_nested_json_in_text(self, text: str) -> str:
        """Handle complex nested JSON objects in text using bracket counting."""
        if "Secret" not in text:
            return text

        result = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                # Found potential JSON start
                # Extract the JSON object by counting braces
                json_str, end_idx = self._extract_json_object(text, i)
                if json_str:
                    try:
                        obj = json.loads(json_str)
                        if isinstance(obj, dict) and obj.get("kind") == "Secret":
                            # This is a Secret, mask it
                            masked_obj = self._mask_secret_object(obj)
                            result.append(json.dumps(masked_obj, separators=(",", ":")))
                            i = end_idx + 1
                            continue
                    except (json.JSONDecodeError, Exception):
                        pass

            result.append(text[i])
            i += 1

        return "".join(result)

    def _extract_json_object(self, text: str, start: int) -> tuple[Optional[str], int]:
        """Extract a complete JSON object from text starting at given position.

        Returns:
            Tuple of (json_string, end_index) or (None, start) if not valid JSON
        """
        if start >= len(text) or text[start] != "{":
            return None, start

        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        return text[start : i + 1], i

        # Incomplete JSON object
        return None, start

