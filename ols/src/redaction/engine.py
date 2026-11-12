"""Generic rule-set compilation and redaction engine."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

Replacement = str | Callable[[re.Match[str]], str]
ApplyFunction = Callable[[str], tuple[str, dict[str, int]]]


@dataclass(frozen=True)
class RedactionRule:
    """One named pattern and its replacement behavior."""

    name: str
    pattern: str
    replacement: Replacement


@dataclass(frozen=True)
class CompiledRuleSet:
    """Reusable compiled policy and its selected reduction behavior."""

    name: str
    _apply: ApplyFunction = field(repr=False, compare=False)

    def apply(self, text: str) -> tuple[str, dict[str, int]]:
        """Apply this compiled policy and return text and per-rule counts."""
        return self._apply(text)


@dataclass(frozen=True)
class RuleSet:
    """A redaction policy definition that can be compiled for reuse."""

    name: str
    rules: tuple[RedactionRule, ...]
    flags: int = 0
    sequential: bool = True

    def __post_init__(self) -> None:
        """Reject invalid execution settings and ambiguous rule names."""
        if not isinstance(self.sequential, bool):
            raise TypeError("sequential must be a bool")
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("ruleset rule names must be unique")

    def compile(self) -> CompiledRuleSet:
        """Compile this policy once, selecting its declared reduction mode."""
        if self.sequential:
            matchers = tuple(
                (rule, re.compile(rule.pattern, self.flags)) for rule in self.rules
            )

            def apply(text: str) -> tuple[str, dict[str, int]]:
                current = text
                counts: dict[str, int] = {}
                for rule, matcher in matchers:
                    current, count = matcher.subn(rule.replacement, current)
                    if count:
                        counts[rule.name] = counts.get(rule.name, 0) + count
                return current, counts

        else:
            groups = tuple(
                (f"rule_{index}", rule) for index, rule in enumerate(self.rules)
            )
            alternatives = "|".join(
                f"(?P<{group_name}>{rule.pattern})" for group_name, rule in groups
            )
            matcher = re.compile(alternatives or r"(?!)", self.flags)

            def apply(text: str) -> tuple[str, dict[str, int]]:
                counts: dict[str, int] = {}

                def replace(match: re.Match[str]) -> str:
                    group_name = match.lastgroup
                    for candidate, rule in groups:
                        if candidate == group_name:
                            counts[rule.name] = counts.get(rule.name, 0) + 1
                            replacement = rule.replacement
                            return (
                                replacement(match)
                                if callable(replacement)
                                else match.expand(replacement)
                            )
                    raise RuntimeError("compiled ruleset matched without a named rule")

                return matcher.sub(replace, text), counts

        return CompiledRuleSet(name=self.name, _apply=apply)


@dataclass(frozen=True)
class RedactionResult:
    """Sanitized text and counts from one redaction operation."""

    text: str
    redacted: bool
    redacted_count: int
    counts: dict[str, int]


class Redactor:
    """Apply any compiled policy and return a generic result."""

    @staticmethod
    def redact(text: str, rules: CompiledRuleSet) -> RedactionResult:
        """Apply compiled rules and return sanitized text and per-rule counts."""
        sanitized, counts = rules.apply(text)
        total = sum(counts.values())
        return RedactionResult(
            text=sanitized,
            redacted=total > 0,
            redacted_count=total,
            counts=counts,
        )
