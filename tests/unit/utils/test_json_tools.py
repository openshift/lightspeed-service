import pytest
from utils.json_tools import parse_json_markdown
from utils.json_tools import parse_and_check_json_markdown


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


incorrect_inputs = (
    None,
    "",
    "foo",
    "\n",
    '"',
    "```foo```",
    "```json{foo}```",
    "```json{```",
)


@pytest.mark.parametrize("json", incorrect_inputs)
def test_parse_json_markdown_negative_test_cases(json):
    """Check parsing incorrect input."""
    with pytest.raises(Exception):
        parse_json_markdown(json)


@pytest.mark.parametrize("json,expected", correct_inputs)
def test_parse_and_check_json_markdown(json, expected):
    """Check parsing incorrect input."""
    json_obj = parse_and_check_json_markdown(json, [])
    assert json_obj is not None
    assert parse_json_markdown(json) == expected


correct_inputs_with_keys = (
    ("{}", {}, []),
    ('{"x":42}', {"x": 42}, ["x"]),
    ('{"x":42, "y":""}', {"x": 42, "y": ""}, ["x", "y"]),
    ('{"x":42, "y":"", "z":[]}', {"x": 42, "y": "", "z": []}, ["x", "y", "z"]),
    ('   {"x":42, "y":""}   ', {"x": 42, "y": ""}, ["x", "y"]),
    ('```{"x":42, "y":""}```', {"x": 42, "y": ""}, ["x", "y"]),
    ('\n{"x":42, "y":""}\n', {"x": 42, "y": ""}, ["x", "y"]),
    ('\t{"x":42, "y":""}\t', {"x": 42, "y": ""}, ["x", "y"]),
    ('```json{"x":42, "y":""}```', {"x": 42, "y": ""}, ["x", "y"]),
)


@pytest.mark.parametrize("json,expected,keys", correct_inputs_with_keys)
def test_parse_and_check_json_markdown_check_keys(json, expected, keys):
    """Check parsing correct input and checking if resulting object contains given keys."""
    json_obj = parse_and_check_json_markdown(json, keys)
    assert json_obj is not None
    assert parse_json_markdown(json) == expected


correct_inputs_with_unexpected_keys = (
    ('{"x":42}', {"x": 42}, ["foo"]),
    ('{"x":42}', {"x": 42}, ["foo", "bar", "baz"]),
    ('{"x":42}', {"x": 42}, ["x", "foo"]),
)


@pytest.mark.parametrize("json,expected,keys", correct_inputs_with_unexpected_keys)
def test_parse_and_check_json_markdown_check_for_unexpected_keys(json, expected, keys):
    """Check parsing incorrect input."""
    with pytest.raises(ValueError):
        parse_and_check_json_markdown(json, keys)


@pytest.mark.parametrize("json", incorrect_inputs)
def test_parse_and_check_json_markdown_negative_test_cases(json):
    """Check parsing incorrect input."""
    with pytest.raises(ValueError):
        parse_and_check_json_markdown(json, [])
