"""Verify the full-dataset LSEval CI provider matrix."""

import os
import subprocess
from pathlib import Path

import yaml

from tests.e2e.evaluation.test_lseval_periodic import (
    _LSEVAL_PERIODIC_PROVIDERS,
    _PROVIDER_CONFIGS,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_periodic_runs_all_six_providers_after_failure(tmp_path: Path) -> None:
    """Run every provider sequentially and report any suite failure."""
    entrypoint = (SCRIPTS / "test-lseval-periodic.sh").read_text()
    runner = entrypoint.split("function run_suites() {", 1)[1].split(
        "# lightspeed-eval writes", 1
    )[0]
    calls = tmp_path / "calls"
    script = f"""
    function run_suites() {{{runner}
    run_suite() {{ printf '%s|%s|%s\\n' "$1" "$3" "$5" >> '{calls}'; [[ "$3" != watsonx ]]; }}
    cleanup_ols_operator() {{ :; }}
    run_suites
    """
    env = {
        **os.environ,
        "OPENAI_PROVIDER_KEY_PATH": "openai",
        "WATSONX_PROVIDER_KEY_PATH": "watsonx",
        "AZUREOPENAI_PROVIDER_KEY_PATH": "azure",
        "VERTEX_PROVIDER_KEY_PATH": "vertex",
        "OLS_IMAGE": "image",
        "RHOAI_PROVISION": "false",
    }
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", script], env=env, check=False
    )
    assert result.returncode != 0
    assert calls.read_text().splitlines() == [
        "lseval_periodic_openai|openai|gpt-6-luna",
        "lseval_periodic_watsonx|watsonx|ibm/granite-4-h-small",
        "lseval_periodic_azure_openai|azure_openai|gpt-6-luna",
        "lseval_periodic_vertex_gemini|vertex_gemini|gemini-3.1-flash-lite",
        "lseval_periodic_vertex_claude|vertex_claude|claude-opus-4-6",
        "lseval_periodic_bedrock_openai|bedrock_openai|openai.gpt-6-luna",
    ]
    assert all(
        provider in _LSEVAL_PERIODIC_PROVIDERS
        for provider in (
            "openai",
            "watsonx",
            "azure_openai",
            "vertex_gemini",
            "vertex_claude",
            "bedrock_openai",
        )
    )
    assert all(
        _PROVIDER_CONFIGS[provider].is_file() for provider in _LSEVAL_PERIODIC_PROVIDERS
    )


def test_bedrock_openai_periodic_model_matches_operator_config() -> None:
    """The runner, OLSConfig and eval target must all name the same model."""
    project_root = SCRIPTS.parents[1]
    olsconfig_path = (
        project_root
        / "tests/config/operator_install/olsconfig.crd.bedrock_openai_lseval.yaml"
    )
    eval_config = yaml.safe_load(_PROVIDER_CONFIGS["bedrock_openai"].read_text())
    olsconfig = yaml.safe_load(olsconfig_path.read_text())["spec"]
    provider = olsconfig["llm"]["providers"][0]
    assert provider["type"] == "bedrock"
    assert provider["models"] == [{"name": "openai.gpt-6-luna"}]
    assert (
        olsconfig["ols"]["defaultModel"]
        == eval_config["api"]["model"]
        == "openai.gpt-6-luna"
    )
    assert eval_config["api"]["provider"] == "bedrock"
