"""Load and validate labeled tool-result inspection cases."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

InjectionCategory = Literal[
    "instruction_override",
    "role_change",
    "prompt_extraction",
    "data_exfiltration",
    "tool_manipulation",
    "unknown",
]
ExpectedCategory = Literal[
    "none",
    "instruction_override",
    "role_change",
    "prompt_extraction",
    "data_exfiltration",
    "tool_manipulation",
    "unknown",
]


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node, deep: bool = False):
        """Construct a mapping while rejecting repeated keys."""
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    None, None, "duplicate key", key_node.start_mark
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


class InspectionCase(BaseModel):
    """One labeled model-visible tool-result evaluation case."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: str
    tool_name: str
    result_type: Literal["result", "error"]
    content: str
    expected_outcome: Literal["benign", "malicious"]
    expected_category: ExpectedCategory
    tags: list[str]

    @model_validator(mode="after")
    def validate_category(self) -> "InspectionCase":
        """Require category and outcome to agree."""
        if self.expected_outcome == "benign" and self.expected_category != "none":
            raise ValueError("benign cases must use expected_category=none")
        if self.expected_outcome == "malicious" and self.expected_category == "none":
            raise ValueError("malicious cases must use an injection category")
        if not self.tags:
            raise ValueError("tags must not be empty")
        return self


def load_cases(path: Path) -> list[InspectionCase]:
    """Load and validate an evaluation corpus without exposing case content in errors."""
    try:
        document = yaml.load(
            path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader  # noqa: S506
        )
        if not isinstance(document, list):
            raise ValueError("corpus must contain a top-level list")
        cases = [InspectionCase.model_validate(item) for item in document]
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as error:
        if (
            isinstance(error, ValueError)
            and str(error) == "corpus must contain a top-level list"
        ):
            raise
        if isinstance(error, ValidationError):
            locations = sorted(
                str(location)
                for validation_error in error.errors()
                for location in validation_error.get("loc", ())
            )
            fields = (
                ", ".join(dict.fromkeys(locations))
                or "case fields including expected_category"
            )
            raise ValueError(f"invalid corpus field: {fields}") from error
        raise ValueError(
            f"invalid tool-result inspection corpus: {type(error).__name__}"
        ) from error

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id in tool-result inspection corpus")
    return cases
