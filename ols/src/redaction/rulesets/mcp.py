"""Sequential MCP tool-output redaction rules."""

import re

from ols.src.redaction.engine import RedactionRule, RuleSet

MCP_RULES = RuleSet(
    name="mcp-sequential",
    flags=re.IGNORECASE | re.VERBOSE,
    sequential=True,
    rules=(
        RedactionRule(
            "credential",
            r"""
            (?P<credential_prefix>
                (?<![A-Za-z0-9_])
                (?:password|passwd|pwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret[_-]?key)
                [ \t]*=[ \t]*
            )
            (?P<credential_value>
                "(?:\\.|[^"\\])*"
                |'(?:\\.|[^'\\])*'
                |[^\s,;#]+
            )
            """,
            lambda match: f"{match.group('credential_prefix')}[REDACTED:credential]",
        ),
        RedactionRule(
            "jwt",
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])",
            "[REDACTED:jwt]",
        ),
        RedactionRule(
            "github_token",
            r"(?<![A-Za-z0-9_])ghp_[A-Za-z0-9]{36}(?![A-Za-z0-9_])",
            "[REDACTED:github_token]",
        ),
        RedactionRule(
            "aws_access_key",
            r"(?<![A-Za-z0-9_])AKIA[0-9A-Z]{16}(?![A-Za-z0-9_])",
            "[REDACTED:aws_access_key]",
        ),
        RedactionRule(
            "api_key",
            r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])",
            "[REDACTED:api_key]",
        ),
        RedactionRule(
            "oauth_token",
            r"""
            (?P<bearer_prefix>(?<![A-Za-z0-9_])Bearer[ \t]+)
            (?P<bearer_value>[A-Za-z0-9._~+/-]{16,}={0,2}(?![A-Za-z0-9_]))
            """,
            lambda match: f"{match.group('bearer_prefix')}[REDACTED:oauth_token]",
        ),
        RedactionRule(
            "email",
            r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+[@\uFF20][A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9-]|\.[A-Za-z0-9])",
            "[REDACTED:email]",
        ),
        RedactionRule(
            "ssn",
            r"""
            (?<![0-9])
            (?!(?:000|666|9[0-9]{2})[- ])
            (?:[0-6][0-9]{2}|7[0-9]{2}|8[0-9]{2})
            [- ](?!00)[0-9]{2}
            [- ](?!0000)[0-9]{4}
            (?![0-9])
            """,
            "[REDACTED:ssn]",
        ),
        RedactionRule(
            "phone",
            r"""
            (?<![0-9])(?:\+1[ .-]?)?\(?[2-9][0-9]{2}\)?[ .-][2-9][0-9]{2}[ .-][0-9]{4}(?![0-9])
            |(?<![0-9])\+[1-9][0-9]{0,2}(?:[ .-]?[0-9]){7,12}(?![0-9])
            """,
            "[REDACTED:phone]",
        ),
    ),
)
