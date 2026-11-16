"""Regression tests for the redaction engine and built-in MCP policy."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from ols.src.redaction import engine as engine_module
from ols.src.redaction.engine import RedactionRule, Redactor, RuleSet
from ols.src.redaction.rulesets.mcp import MCP_RULES

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tool_output_redaction.json"


def test_sequential_mcp_rules_match_conformance_fixtures() -> None:
    """Match sandbox policy output and counts while preserving network values."""
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    compiled = MCP_RULES.compile()

    for fixture in fixtures:
        result = Redactor.redact(fixture["input"], compiled)
        assert result.text == fixture["expected"], fixture["name"]
        assert result.redacted is fixture["redacted"], fixture["name"]
        assert result.redacted_count == fixture["redacted_count"], fixture["name"]
        assert result.counts == fixture["counts"], fixture["name"]


@pytest.mark.parametrize(
    "value",
    [
        '"correct horse battery staple"',
        "'correct horse battery staple'",
        r'"correct \"horse\" battery staple"',
        r"'correct \'horse\' battery staple'",
        '"correct, horse; battery#staple"',
        "'correct, horse; battery#staple'",
        "MySecret123!",
    ],
)
def test_credential_rule_redacts_complete_quoted_values(value: str) -> None:
    """Replace the entire credential while preserving surrounding text."""
    result = Redactor.redact(f"before password={value}; after", MCP_RULES.compile())

    assert result.text == "before password=[REDACTED:credential]; after"
    assert result.redacted is True
    assert result.redacted_count == 1
    assert result.counts == {"credential": 1}


def test_sequential_rules_compile_once_and_reuse_compiled_matchers() -> None:
    """Reuse compiled patterns across redaction calls without recompiling."""
    rules = RuleSet(
        name="test",
        rules=(
            RedactionRule("first", "alpha", "[FIRST]"),
            RedactionRule("second", "beta", "[SECOND]"),
        ),
    )
    original_compile = re.compile
    with patch.object(
        engine_module.re, "compile", wraps=original_compile
    ) as compile_regex:
        compiled = rules.compile()
        assert compile_regex.call_count == 2
        first = Redactor.redact("alpha beta", compiled)
        second = Redactor.redact("beta", compiled)
        assert compile_regex.call_count == 2

    assert first.text == "[FIRST] [SECOND]"
    assert first.counts == {"first": 1, "second": 1}
    assert second.text == "[SECOND]"
    assert second.counts == {"second": 1}


@pytest.mark.parametrize("sequential", [True, False])
def test_rules_expand_whole_match_replacement_template(sequential: bool) -> None:
    """Expand whole-match backreferences in string replacements in either mode."""
    compiled = RuleSet(
        name="replacement-template",
        rules=(RedactionRule("copy", "secret", r"<\g<0>>"),),
        sequential=sequential,
    ).compile()

    result = Redactor.redact("before secret after", compiled)

    assert result.text == "before <secret> after"
    assert result.redacted is True
    assert result.redacted_count == 1
    assert result.counts == {"copy": 1}


@pytest.mark.parametrize("sequential", [True, False])
def test_rules_preserve_regex_replacements_and_flags(sequential: bool) -> None:
    """Apply string and callable replacements with the selected matching mode."""
    compiled = RuleSet(
        name="custom",
        rules=(
            RedactionRule("email", "admin@example.com", "[EMAIL]"),
            RedactionRule("credential", r"password=\w+", lambda _: "password=[SECRET]"),
        ),
        flags=re.IGNORECASE,
        sequential=sequential,
    ).compile()

    result = Redactor.redact("ADMIN@example.com password=secret", compiled)

    assert result.text == "[EMAIL] password=[SECRET]"
    assert result.redacted is True
    assert result.redacted_count == 2
    assert result.counts == {"email": 1, "credential": 1}


@pytest.mark.parametrize("sequential", [True, False])
def test_empty_rules_preserve_text(sequential: bool) -> None:
    """Leave input untouched when no custom prompt filters are configured."""
    compiled = RuleSet(name="empty", rules=(), sequential=sequential).compile()

    result = Redactor.redact("admin@example.com", compiled)

    assert result.text == "admin@example.com"
    assert result.redacted is False
    assert result.redacted_count == 0
    assert result.counts == {}


def test_rules_reject_duplicate_names() -> None:
    """Reject ambiguous per-rule count identities."""
    with pytest.raises(ValueError, match="ruleset rule names must be unique"):
        RuleSet(
            name="duplicate",
            rules=(RedactionRule("same", "a", "x"), RedactionRule("same", "b", "y")),
        )


def test_rules_reject_non_boolean_execution_mode() -> None:
    """Reject an invalid execution mode instead of silently changing matching."""
    with pytest.raises(TypeError, match="sequential must be a bool"):
        RuleSet(name="invalid", rules=(), sequential="false")  # type: ignore[arg-type]


def test_rules_reject_invalid_regex_on_compile() -> None:
    """Fail policy compilation when a configured regex is invalid."""
    with pytest.raises(re.error):
        RuleSet(name="invalid", rules=(RedactionRule("broken", "[", "x"),)).compile()
