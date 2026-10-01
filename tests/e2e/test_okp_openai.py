"""Live OKP documentation retrieval checks with the OpenAI provider."""

import json
import os
from urllib.parse import urlparse

import pytest
import requests
import yaml

from ols import constants
from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e import test_api
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.decorators import retry


@pytest.fixture(scope="module", autouse=True)
def okp_ready() -> None:
    """Require the operator-managed OKP service and OLS retrieval configuration."""
    cr = json.loads(
        cluster_utils.run_oc(["get", "olsconfig", "cluster", "-o", "json"]).stdout
    )
    assert not cr["spec"]["ols"].get("byokRAGOnly", False)
    cluster_utils.run_oc(
        ["rollout", "status", "deployment/lightspeed-rhokp", "--timeout=90s"]
    )
    configmap = json.loads(
        cluster_utils.run_oc(["get", "configmap", "olsconfig", "-o", "json"]).stdout
    )
    service_config = yaml.safe_load(configmap["data"][DEFAULT_CONFIGURATION_FILE])
    assert service_config["ols_config"].get("solr_hybrid")


@pytest.mark.rag
@pytest.mark.tool_calling
@pytest.mark.parametrize("endpoint", ("/v1/query", "/v1/streaming_query"))
@pytest.mark.parametrize("check_version", (False, True), ids=("retrieval", "version"))
@retry(max_attempts=3, wait_between_runs=10)
def test_openai_okp_documentation_retrieval(endpoint: str, check_version: bool) -> None:
    """Verify live OKP tool calls, citations, and OCP doc version for both endpoints."""
    if os.getenv("PROVIDER", "openai") != "openai":
        pytest.skip("OKP retrieval smoke test requires the OpenAI provider")

    query = (
        "Search OpenShift documentation for OpenShift Virtualization live migration."
    )
    if check_version:
        major, minor = cluster_utils.get_cluster_version()
        query = (
            f"Search OpenShift Container Platform {major}.{minor} documentation "
            "for OpenShift Virtualization live migration."
        )

    payload = {"query": query}
    if endpoint == "/v1/streaming_query":
        payload["media_type"] = constants.MEDIA_TYPE_JSON
    response = pytest.client.post(  # type: ignore[attr-defined]
        endpoint, json=payload, timeout=test_api.LLM_REST_API_TIMEOUT
    )
    assert response.status_code == requests.codes.ok

    if endpoint == "/v1/streaming_query":
        assert response.headers["content-type"].startswith(constants.MEDIA_TYPE_JSON)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1]["event"] == "end"
        tool_names = [
            event["data"]["name"] for event in events if event["event"] == "tool_call"
        ]
        references = events[-1]["data"]["referenced_documents"]
    else:
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        tool_names = [call["name"] for call in body["tool_calls"]]
        references = body["referenced_documents"]

    assert "search_openshift_documentation" in tool_names
    assert references
    urls = [doc["doc_url"] for doc in references]
    assert all(doc["doc_title"] for doc in references)
    assert all(
        urlparse(url).scheme == "https"
        and urlparse(url).hostname in ("docs.redhat.com", "access.redhat.com")
        for url in urls
    )
    assert len(urls) == len(set(urls))
    assert any(
        "openshift_container_platform" in url.lower()
        and "virtualization" in url.lower()
        for url in urls
    )
    if check_version:
        assert any(f"{major}.{minor}" in url for url in urls)
