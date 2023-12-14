import pytest
from utils.json_tools import parse_json_markdown


correct_inputs = (
    ("{}", {}),
    ('{"x":42}', {"x": 42}),
    ('{"x":42, "y":""}', {"x": 42, "y": ""}),
    ('   {"x":42, "y":""}   ', {"x": 42, "y": ""}),
    ('```{"x":42, "y":""}```', {"x": 42, "y": ""}),
    ('\n{"x":42, "y":""}\n', {"x": 42, "y": ""}),
    ('\t{"x":42, "y":""}\t', {"x": 42, "y": ""}),
    ('```json{"x":42, "y":""}```', {"x": 42, "y": ""}),
)


@pytest.mark.parametrize("json,expected", correct_inputs)
def test_parse_json_markdown(json, expected):
    """Check parsing correct input."""
    assert parse_json_markdown(json) == expected


incorrect_inputs = (None, "", "foo", "\n", '"')


@pytest.mark.parametrize("json", incorrect_inputs)
def test_parse_json_markdown_negative_test_cases(json):
    """Check parsing incorrect input."""
    with pytest.raises(Exception):
        parse_json_markdown(json)
