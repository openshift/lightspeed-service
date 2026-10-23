"""Split consolidated tool-result inspection artifacts per provider.

The consolidated inspection suite runs every provider against one deployment
and produces a single JUnit report plus one must-gather tree. This script
restores the per-provider artifact layout that existed before consolidation:

- one ``junit_e2e_<suite_id>.xml`` per provider/model pair, using the
  historical suite ids so test-grid history stays continuous
- one ``<suite_id>/cluster/podlogs/`` directory per provider holding that
  provider's slice of the app-server log

Cluster-state files (pods, deployments, etc.) stay in the shared
``<suite_id>/cluster`` directory because all providers share one deployment.

Run after the inspection suite completes:

    python tests/scripts/split_inspection_artifacts.py <artifact_dir> <suite_id>
"""

import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from defusedxml.ElementTree import parse as parse_xml

# (provider, model) -> suite id used by the per-provider runs before
# consolidation. Bedrock maps to two suite ids, one per model family.
SUITE_IDS: dict[tuple[str, str], str] = {
    ("openai", "gpt-6-luna"): "openai_gpt6_tool_result_inspection",
    ("azure_openai", "gpt-6-luna"): "azure_openai_tool_result_inspection",
    ("bedrock", "anthropic.claude-haiku-5-5"): (
        "bedrock_anthropic_tool_result_inspection"
    ),
    ("bedrock", "openai.gpt-6-luna"): "bedrock_openai_tool_result_inspection",
    ("google_vertex", "gemini-3.1-flash-lite"): (
        "google_vertex_tool_result_inspection"
    ),
    ("google_vertex_anthropic", "claude-sonnet-4-6"): (
        "google_vertex_anthropic_tool_result_inspection"
    ),
    ("watsonx", "ibm/granite-4-h-small"): "watsonx_tool_result_inspection",
    ("rhoai_vllm", "gpt-4.1-mini"): "rhoai_vllm_tool_result_inspection",
    ("rhelai_vllm", "gpt-4.1-mini"): "rhelai_vllm_tool_result_inspection",
}

_CLASSIFIER_MARKER = re.compile(
    r"Tool-result classifier initialized for "
    r"provider=(?P<provider>[^,\s]+), model=(?P<model>[^,\s]+)"
)


def _suite_id_for_testcase(name: str) -> str | None:
    """Return the suite id owning a testcase, or None when unassigned.

    Parametrized test names end with ``[<provider>--<model>]`` (benign) or
    ``[<endpoint>-<provider>--<model>]`` (malicious). The character before
    the provider must be ``[`` or ``-`` so ``openai`` does not also match
    ``azure_openai``.
    """
    for (provider, model), suite_id in SUITE_IDS.items():
        needle = f"{provider}--{model.replace('/', '_')}]"
        index = name.find(needle)
        while index != -1:
            preceding = name[index - 1] if index > 0 else ""
            if preceding in ("", "-", "["):
                return suite_id
            index = name.find(needle, index + 1)
    return None


def _write_provider_junit(
    root: ET.Element,
    suite: ET.Element,
    cases: list[ET.Element],
    suite_id: str,
    classname_prefix: str,
    destination: Path,
) -> None:
    """Write one provider's testcases as a standalone JUnit report."""
    provider_suite = ET.Element("testsuite", attrib=dict(suite.attrib))
    provider_suite.set("name", suite_id)
    for case in cases:
        classname = case.attrib.get("classname", "")
        if classname.startswith(classname_prefix):
            case.set("classname", f"{suite_id}.{classname[len(classname_prefix) :]}")
        provider_suite.append(case)
    provider_suite.set("tests", str(len(cases)))
    provider_suite.set(
        "failures", str(sum(len(case.findall("failure")) for case in cases))
    )
    provider_suite.set("errors", str(sum(len(case.findall("error")) for case in cases)))
    provider_suite.set(
        "skipped", str(sum(len(case.findall("skipped")) for case in cases))
    )
    provider_suite.set(
        "time", f"{sum(float(case.attrib.get('time', 0)) for case in cases):.3f}"
    )

    if root.tag == "testsuites":
        new_root = ET.Element("testsuites", attrib=dict(root.attrib))
        new_root.append(provider_suite)
    else:
        new_root = provider_suite
    ET.ElementTree(new_root).write(destination, encoding="utf-8", xml_declaration=True)


def split_junit(junit_path: Path, artifact_dir: Path, suite_id: str) -> set[str]:
    """Write one JUnit file per provider and drop the combined report.

    The combined report is removed only when every testcase was assigned to
    a provider, so unassigned cases are never lost.

    Returns:
        The set of suite ids that received at least one testcase.
    """
    tree = parse_xml(junit_path)
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"No testsuite element in {junit_path}")

    classname_prefix = f"{suite_id}."
    grouped: dict[str, list[ET.Element]] = {}
    unassigned = 0
    for case in suite.findall("testcase"):
        provider_suite_id = _suite_id_for_testcase(case.attrib.get("name", ""))
        if provider_suite_id is None:
            unassigned += 1
            continue
        grouped.setdefault(provider_suite_id, []).append(case)

    for provider_suite_id, cases in grouped.items():
        _write_provider_junit(
            root,
            suite,
            cases,
            provider_suite_id,
            classname_prefix,
            artifact_dir / f"junit_e2e_{provider_suite_id}.xml",
        )

    if unassigned:
        print(
            f"WARNING: {unassigned} testcase(s) matched no provider; "
            f"keeping the combined report {junit_path.name}"
        )
    else:
        junit_path.unlink()
        print(
            f"Split {sum(len(c) for c in grouped.values())} testcases into "
            f"{len(grouped)} per-provider JUnit reports"
        )
    return set(grouped)


def _slice_log_by_provider(text: str) -> dict[str, str]:
    """Slice an app-server log into per-suite chunks by request markers.

    Each query logs a ``Tool-result classifier initialized for provider=...``
    line when its DocsSummarizer is constructed; requests run sequentially,
    so every line belongs to the request of the most recent marker.
    """
    slices: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines(keepends=True):
        marker = _CLASSIFIER_MARKER.search(line)
        if marker is not None:
            key = (marker.group("provider"), marker.group("model"))
            current = SUITE_IDS.get(key)
            if current is None:
                print(f"WARNING: no suite id for log marker {key!r}")
        if current is not None:
            slices.setdefault(current, []).append(line)
    return {suite_id: "".join(lines) for suite_id, lines in slices.items()}


def split_podlogs(cluster_dir: Path, artifact_dir: Path, suite_ids: set[str]) -> None:
    """Write each provider's app-server log slice under its suite directory.

    Falls back to copying the full log when a provider's slice would be
    empty, so debugging material is never silently lost.
    """
    podlogs_dir = cluster_dir / "podlogs"
    if not podlogs_dir.is_dir():
        print(f"No podlogs directory at {podlogs_dir}; skipping log slicing")
        return
    app_server_logs = sorted(podlogs_dir.glob("lightspeed-app-server-*.log"))
    if not app_server_logs:
        print("No app-server podlog found; skipping log slicing")
        return

    for suite_id in sorted(suite_ids):
        destination_dir = artifact_dir / suite_id / "cluster" / "podlogs"
        destination_dir.mkdir(parents=True, exist_ok=True)
        for log_path in app_server_logs:
            destination = destination_dir / log_path.name
            slices = _slice_log_by_provider(
                log_path.read_text(encoding="utf-8", errors="replace")
            )
            slice_text = slices.get(suite_id, "")
            if slice_text:
                with destination.open("a", encoding="utf-8") as slice_file:
                    slice_file.write(slice_text)
            elif not destination.exists():
                print(
                    f"WARNING: no log lines matched {suite_id}; "
                    f"copying the full log {log_path.name}"
                )
                shutil.copyfile(log_path, destination)


def main(artifact_dir: str, suite_id: str) -> int:
    """Split the inspection suite artifacts in artifact_dir per provider."""
    artifacts = Path(artifact_dir)
    junit_path = artifacts / f"junit_e2e_{suite_id}.xml"
    if not junit_path.is_file():
        print(f"No JUnit report at {junit_path}; nothing to split")
        return 0
    suite_ids = split_junit(junit_path, artifacts, suite_id)
    split_podlogs(artifacts / suite_id / "cluster", artifacts, suite_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
