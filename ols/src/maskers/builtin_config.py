"""Built-in masking patterns, pattern groups, and code-based maskers.

This module contains all built-in masking patterns and configuration
for the DataMaskingService. It serves as the single source of truth for:
- Built-in regex masking patterns
- Pattern groups for convenient configuration
- Code-based masker import paths
"""

from typing import Dict, List

# ==============================================================================
# BUILT-IN MASKING PATTERNS
# ==============================================================================

# Central registry of all built-in masking patterns for MCP server responses
# Format: "pattern_name" -> {"pattern": regex, "replacement": text, "description": text}
# Note: Kubernetes Secret masking is handled by code-based masker (see BUILTIN_CODE_MASKERS)
BUILTIN_MASKING_PATTERNS: Dict[str, Dict[str, str]] = {
    "base64_secret": {
        "pattern": r"\b([A-Za-z0-9+/]{20,}={0,2})\b",
        "replacement": "__MASKED_BASE64_VALUE__",
        "description": "Base64-encoded values in secret contexts (20+ chars)",
    },
    "base64_short": {
        "pattern": r"(?<=:\s)([A-Za-z0-9+/]{4,19}={0,2})(?=\s|$)",
        "replacement": "__MASKED_SHORT_BASE64__",
        "description": "Short base64-encoded values (4-19 chars) after colons in Kubernetes contexts",
    },
    "api_key": {
        "pattern": r'(?i)(?:api[_-]?key|apikey|key)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        "replacement": r'"api_key": "__MASKED_API_KEY__"',
        "description": "API keys in various formats",
    },
    "password": {
        "pattern": r'(?i)(?:password|pwd|pass)["\']?\s*[:=]\s*["\']?([^"\'\s\n]{6,})["\']?',
        "replacement": r'"password": "__MASKED_PASSWORD__"',
        "description": "Password fields",
    },
    "certificate": {
        "pattern": r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----",
        "replacement": "__MASKED_CERTIFICATE__",
        "description": "SSL/TLS certificates and private keys",
    },
    "certificate_authority_data": {
        "pattern": r"(?i)certificate-authority-data:\s*([A-Za-z0-9+/]{20,}={0,2})",
        "replacement": r"certificate-authority-data: __MASKED_CA_CERTIFICATE__",
        "description": "Certificate authority data in Kubernetes configs and YAML files",
    },
    "email": {
        "pattern": r"(?<!\\)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*\.[A-Za-z]{2,63}\b(?!\()",
        "replacement": "__MASKED_EMAIL__",
        "description": "Email addresses",
    },
    "token": {
        "pattern": r'(?i)(?:token|bearer|jwt)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-\.]{20,})["\']?',
        "replacement": r'"token": "__MASKED_TOKEN__"',
        "description": "Access tokens, bearer tokens, and JWTs",
    },
    "ssh_key": {
        "pattern": r"ssh-(?:rsa|dss|ed25519|ecdsa)\s+[A-Za-z0-9+/=]+",
        "replacement": "__MASKED_SSH_KEY__",
        "description": "SSH public keys (all common algorithms)",
    },
}

# ==============================================================================
# BUILT-IN PATTERN GROUPS
# ==============================================================================

# Central registry of built-in pattern groups for convenient configuration
# Format: "group_name" -> [list_of_pattern_names]
# Groups can reference both regex patterns and code-based maskers
BUILTIN_PATTERN_GROUPS: Dict[str, List[str]] = {
    "basic": ["api_key", "password"],  # Most common secrets
    "secrets": ["api_key", "password", "token"],  # Basic + tokens
    "security": [
        "api_key",
        "password",
        "token",
        "certificate",
        "certificate_authority_data",
        "email",
        "ssh_key",
    ],  # Full security focus
    "kubernetes": [
        "kubernetes_secret",
        "api_key",
        "password",
        "certificate_authority_data",
    ],  # Kubernetes-specific - uses code-based masker for Secrets (not ConfigMaps)
    "all": [
        "base64_secret",
        "base64_short",
        "api_key",
        "password",
        "certificate",
        "certificate_authority_data",
        "email",
        "token",
        "ssh_key",
    ],  # All patterns
}

# ==============================================================================
# BUILT-IN CODE-BASED MASKERS
# ==============================================================================

# Central registry of built-in code-based maskers
# Format: "masker_name" -> "import.path.ClassName"
# Code-based maskers provide structural awareness for complex masking scenarios
# where simple regex patterns are insufficient (e.g., parsing YAML/JSON structures)
BUILTIN_CODE_MASKERS: Dict[str, str] = {
    "kubernetes_secret": "ols.src.maskers.kubernetes_secret_masker.KubernetesSecretMasker",
    # Future maskers will be added here as needed
}


# ==============================================================================
# CONVENIENCE ACCESSORS
# ==============================================================================


def get_builtin_masking_patterns() -> Dict[str, Dict[str, str]]:
    """Get all built-in masking patterns."""
    return BUILTIN_MASKING_PATTERNS.copy()


def get_builtin_pattern_groups() -> Dict[str, List[str]]:
    """Get all built-in pattern groups."""
    return {k: v.copy() for k, v in BUILTIN_PATTERN_GROUPS.items()}


def get_builtin_code_maskers() -> Dict[str, str]:
    """Get all built-in code-based masker import paths."""
    return BUILTIN_CODE_MASKERS.copy()

