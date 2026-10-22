"""Unit tests for per-provider splitting of inspection artifacts."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from defusedxml.ElementTree import parse as parse_xml

from tests.e2e.test_tool_result_inspection import DEFAULT_INSPECTION_MATRIX
from tests.scripts import split_inspection_artifacts as splitter

SUITE_ID = "tool_result_inspection"


def _write_combined_junit(path: Path, names: list[str], failures: set[str]) -> None:
    """Write a synthetic combined inspection JUnit report."""
    testsuites = ET.Element("testsuites", {"name": "pytest tests"})
    suite = ET.SubElement(
        testsuites,
        "testsuite",
        {
            "name": SUITE_ID,
            "tests": str(len(names)),
            "failures": str(len(failures)),
            "errors": "0",
            "skipped": "0",
            "time": "9.999",
        },
    )
    for name in names:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": f"{SUITE_ID}.test_tool_result_inspection",
                "name": name,
                "time": "1.500",
            },
        )
        if name in failures:
            ET.SubElement(case, "failure", {"message": "inspection failed"})
    ET.ElementTree(testsuites).write(path, encoding="utf-8", xml_declaration=True)


def _read_suite(path: Path) -> ET.Element:
    """Read the single testsuite element of a JUnit report."""
    root = parse_xml(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None
    return suite


def test_suite_ids_cover_the_full_matrix() -> None:
    """Map every matrix entry so no test case can be left unassigned."""
    assert set(splitter.SUITE_IDS) == set(DEFAULT_INSPECTION_MATRIX)


@pytest.mark.parametrize(
    ("name", "expected_suite_id"),
    [
        (
            "test_benign_tool_result_passes_inspection[openai--gpt-6-luna]",
            "openai_gpt6_tool_result_inspection",
        ),
        (
            "test_benign_tool_result_passes_inspection[azure_openai--gpt-6-luna]",
            "azure_openai_tool_result_inspection",
        ),
        (
            "test_malicious_tool_result_fails_closed_without_leaking_content"
            "[sse-azure_openai--gpt-6-luna]",
            "azure_openai_tool_result_inspection",
        ),
        (
            "test_malicious_tool_result_fails_closed_without_leaking_content"
            "[tool-error-bedrock--anthropic.claude-haiku-5-5]",
            "bedrock_anthropic_tool_result_inspection",
        ),
        (
            "test_malicious_tool_result_fails_closed_without_leaking_content"
            "[tool-error-bedrock--openai.gpt-6-luna]",
            "bedrock_openai_tool_result_inspection",
        ),
        (
            "test_malicious_tool_result_fails_closed_without_leaking_content"
            "[tool-error-watsonx--ibm_granite-4-h-small]",
            "watsonx_tool_result_inspection",
        ),
    ],
    ids=[
        "openai",
        "azure-openai",
        "azure-openai-sse",
        "bedrock-anthropic",
        "bedrock-openai",
        "watsonx",
    ],
)
def test_suite_id_for_testcase(name: str, expected_suite_id: str) -> None:
    """Assign each parametrized test name to its historical suite id."""
    assert splitter._suite_id_for_testcase(name) == expected_suite_id


def test_suite_id_for_testcase_rejects_unknown_name() -> None:
    """Report unmatchable names so the combined report is kept."""
    assert splitter._suite_id_for_testcase("test_something[weird]") is None


def test_split_junit_writes_per_provider_reports(tmp_path: Path) -> None:
    """Write one report per provider with recomputed counts and classnames."""
    names = [
        "test_benign_tool_result_passes_inspection[openai--gpt-6-luna]",
        "test_benign_tool_result_passes_inspection[azure_openai--gpt-6-luna]",
        "test_malicious_tool_result_fails_closed_without_leaking_content"
        "[json-azure_openai--gpt-6-luna]",
    ]
    failing = {
        "test_malicious_tool_result_fails_closed_without_leaking_content"
        "[json-azure_openai--gpt-6-luna]"
    }
    combined = tmp_path / "junit_e2e_tool_result_inspection.xml"
    _write_combined_junit(combined, names, failing)

    suite_ids = splitter.split_junit(combined, tmp_path, SUITE_ID)

    assert suite_ids == {
        "openai_gpt6_tool_result_inspection",
        "azure_openai_tool_result_inspection",
    }
    assert not combined.exists()

    openai_suite = _read_suite(
        tmp_path / "junit_e2e_openai_gpt6_tool_result_inspection.xml"
    )
    assert openai_suite.attrib["name"] == "openai_gpt6_tool_result_inspection"
    assert openai_suite.attrib["tests"] == "1"
    assert openai_suite.attrib["failures"] == "0"
    case = openai_suite.find("testcase")
    assert case.attrib["classname"] == (
        "openai_gpt6_tool_result_inspection.test_tool_result_inspection"
    )

    azure_suite = _read_suite(
        tmp_path / "junit_e2e_azure_openai_tool_result_inspection.xml"
    )
    assert azure_suite.attrib["tests"] == "2"
    assert azure_suite.attrib["failures"] == "1"
    assert azure_suite.attrib["time"] == "3.000"


def test_split_junit_keeps_combined_report_for_unassigned_cases(
    tmp_path: Path,
) -> None:
    """Never drop testcases that match no provider."""
    names = [
        "test_benign_tool_result_passes_inspection[openai--gpt-6-luna]",
        "test_unknown_provider_case",
    ]
    combined = tmp_path / "junit_e2e_tool_result_inspection.xml"
    _write_combined_junit(combined, names, set())

    suite_ids = splitter.split_junit(combined, tmp_path, SUITE_ID)

    assert combined.exists()
    assert suite_ids == {"openai_gpt6_tool_result_inspection"}


def test_split_podlogs_writes_provider_slices(tmp_path: Path) -> None:
    """Slice the app-server log by the per-request classifier markers."""
    log = tmp_path / SUITE_ID / "cluster" / "podlogs" / "lightspeed-app-server-1.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "startup line\n"
        "Tool-result classifier initialized for provider=openai, model=gpt-6-luna\n"
        "openai request line\n"
        "Tool-result classifier initialized for provider=watsonx, "
        "model=ibm/granite-4-h-small\n"
        "watsonx request line\n",
        encoding="utf-8",
    )

    splitter.split_podlogs(
        tmp_path / SUITE_ID / "cluster",
        tmp_path,
        {"openai_gpt6_tool_result_inspection", "watsonx_tool_result_inspection"},
    )

    openai_slice = (
        tmp_path
        / "openai_gpt6_tool_result_inspection"
        / "cluster"
        / "podlogs"
        / "lightspeed-app-server-1.log"
    )
    watsonx_slice = (
        tmp_path
        / "watsonx_tool_result_inspection"
        / "cluster"
        / "podlogs"
        / "lightspeed-app-server-1.log"
    )
    assert "openai request line" in openai_slice.read_text(encoding="utf-8")
    assert "watsonx" not in openai_slice.read_text(encoding="utf-8")
    assert "watsonx request line" in watsonx_slice.read_text(encoding="utf-8")
    assert "openai request line" not in watsonx_slice.read_text(encoding="utf-8")


def test_split_podlogs_copies_full_log_when_no_markers_match(
    tmp_path: Path,
) -> None:
    """Fall back to the full log instead of losing debugging material."""
    log = tmp_path / SUITE_ID / "cluster" / "podlogs" / "lightspeed-app-server-1.log"
    log.parent.mkdir(parents=True)
    log.write_text("only unmarked lines\n", encoding="utf-8")

    splitter.split_podlogs(
        tmp_path / SUITE_ID / "cluster",
        tmp_path,
        {"openai_gpt6_tool_result_inspection"},
    )

    copied = (
        tmp_path
        / "openai_gpt6_tool_result_inspection"
        / "cluster"
        / "podlogs"
        / "lightspeed-app-server-1.log"
    )
    assert copied.read_text(encoding="utf-8") == "only unmarked lines\n"


def test_main_without_junit_report_is_not_an_error(tmp_path: Path) -> None:
    """Exit cleanly when the suite produced no report (startup failures)."""
    assert splitter.main(str(tmp_path), SUITE_ID) == 0
