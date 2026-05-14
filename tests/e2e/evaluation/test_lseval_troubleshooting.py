"""Troubleshooting eval tests using OLS troubleshooting mode and cluster scenarios.

The pytest implementation is commented out for now; only a skip placeholder runs.
See the commented block at the bottom of this file to re-enable.

When enabled, two suites are provided:

scenario_evals
    Inject a specific broken cluster state via setup/cleanup scripts, then ask
    OLS to diagnose it.  Expected answers are known in advance and scored with
    custom:answer_correctness (and geval/deepeval metrics for multi-turn cases).

mcp_evals
    Open-ended live-cluster evals that require a pre-broken cluster with MCP
    tools available (obs-mcp + openshift-mcp-server).  No setup scripts are
    used; the cluster must be prepared externally.  Responses are scored with
    geval:generic_troubleshooting_experience.
"""

import pytest

# ---------------------------------------------------------------------------
# Troubleshooting LSEval disabled for now — implementation commented out below.
# Re-enable by removing the leading '#' from each line in this block (and restoring imports above).
# ---------------------------------------------------------------------------
# import json
# import os
# import shutil
# import subprocess
# import tempfile
# from pathlib import Path
#
# import yaml
#
# MAX_EVAL_ERROR_RATE_PCT = 10.0
#
# PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# EVAL_DIR = PROJECT_ROOT / "eval"
# TROUBLESHOOTING_EVAL_DIR = EVAL_DIR / "troubleshooting"
# LSEVAL_BIN = PROJECT_ROOT / ".venv" / "bin" / "lightspeed-eval"
# SYSTEM_CONFIG = TROUBLESHOOTING_EVAL_DIR / "system.yaml"
# SCENARIO_EVAL_DATA = TROUBLESHOOTING_EVAL_DIR / "scenario_evals.yaml"
# MCP_EVAL_DATA = TROUBLESHOOTING_EVAL_DIR / "mcp_evals.yaml"
#
#
# def _ensure_lseval_installed() -> None:
#     """Install the lightspeed-evaluation package via uv if absent."""
#     if LSEVAL_BIN.exists():
#         return

#     uv_path = shutil.which("uv")
#     if not uv_path:
#         raise FileNotFoundError("uv command not found in PATH")

#     subprocess.run(
#         [uv_path, "sync", "--extra", "lseval"],
#         check=True,
#         cwd=str(PROJECT_ROOT),
#     )


# def _resolve_ols_url() -> str:
#     """Return the OLS base URL, preferring the live pytest client over env var."""
#     client = getattr(pytest, "ols_url", None)
#     if client:
#         return client.rstrip("/")
#     return os.getenv("OLS_URL", "http://localhost:8080").rstrip("/")


# def _get_ols_token() -> str:
#     """Extract the bearer token from the pytest HTTP client or environment."""
#     client = getattr(pytest, "client", None)
#     if client is not None:
#         auth_header: str = client.headers.get("Authorization", "")
#         token = auth_header.removeprefix("Bearer ").strip()
#         if token:
#             return token
#     return os.getenv("API_KEY", "")


# def _run_troubleshooting_lseval(
#     eval_data: Path,
#     out_dir: Path,
#     tags: list[str] | None = None,
# ) -> None:
#     """Run lightspeed-eval for troubleshooting scenarios and assert artefacts are produced.

#     Args:
#         eval_data: Path to the evaluation dataset YAML.
#   b      out_dir: Directory where evaluation artefacts are written.
#         tags: Optional list of tags to filter which scenarios are run.
#     """
#     _ensure_lseval_installed()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     ols_url = _resolve_ols_url()

#     with open(SYSTEM_CONFIG, encoding="utf-8") as fh:
#         config = yaml.safe_load(fh)

#     config["api"]["api_base"] = ols_url

#     if model_override := os.getenv("LSEVAL_OLS_MODEL"):
#         config["api"]["model"] = model_override

#     with tempfile.NamedTemporaryFile(
#         mode="w", suffix=".yaml", delete=False, dir=str(TROUBLESHOOTING_EVAL_DIR)
#     ) as tmp:
#         yaml.dump(config, tmp)
#         tmp_config_path = tmp.name

#     env = os.environ.copy()

#     token = _get_ols_token()
#     if token:
#         env["API_KEY"] = token

#     cmd = [
#         str(LSEVAL_BIN),
#         "--system-config",
#         tmp_config_path,
#         "--eval-data",
#         str(eval_data),
#         "--output-dir",
#         str(out_dir),
#     ]
#     if tags:
#         cmd += ["--tags", *tags]

#     try:
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             env=env,
#             cwd=str(TROUBLESHOOTING_EVAL_DIR),
#             check=False,
#         )
#     finally:
#         os.unlink(tmp_config_path)

#     print("--- lightspeed-eval stdout ---")
#     print(result.stdout)
#     if result.stderr:
#         print("--- lightspeed-eval stderr ---")
#         print(result.stderr)

#     assert result.returncode == 0, (
#         f"lightspeed-eval exited with code {result.returncode}.\n"
#         f"stderr:\n{result.stderr}"
#     )

#     csv_files = list(out_dir.glob("*_detailed.csv"))
#     assert csv_files, f"No detailed CSV artefacts found in {out_dir}"

#     json_files = list(out_dir.glob("*_summary.json"))
#     assert json_files, f"No summary JSON artefacts found in {out_dir}"

#     with open(json_files[0], encoding="utf-8") as fh:
#         overall = json.load(fh)["summary_stats"]["overall"]
#     assert overall["error_rate"] <= MAX_EVAL_ERROR_RATE_PCT, (
#         f"{overall['ERROR']}/{overall['TOTAL']} evaluations errored "
#         f"(error_rate={overall['error_rate']:.1f}% > threshold {MAX_EVAL_ERROR_RATE_PCT}%)."
#     )


# @pytest.mark.lseval
# def test_lseval_troubleshooting_scenarios(request: pytest.FixtureRequest) -> None:
#     """Run scenario-based troubleshooting evals against the cluster.

#     Each scenario injects a specific broken cluster state via a setup script,
#     asks OLS to diagnose it, then cleans up.  Expected answers are known in
#     advance and scored with custom:answer_correctness.
#     """
#     out_dir_base = request.config.option.eval_out_dir or str(
#         EVAL_DIR / "results-lseval-troubleshooting"
#     )
#     _run_troubleshooting_lseval(
#         SCENARIO_EVAL_DATA,
#         Path(out_dir_base) / "troubleshooting" / "scenarios",
#     )


# @pytest.mark.lseval
# def test_lseval_troubleshooting_mcp(request: pytest.FixtureRequest) -> None:
#     """Run MCP-based live troubleshooting evals against the cluster.

#     Requires a pre-broken cluster with MCP tools available
#     (obs-mcp + openshift-mcp-server).  No setup scripts are used; the cluster
#     must be prepared externally before running this test.  Responses are scored
#     with geval:generic_troubleshooting_experience.
#     """
#     out_dir_base = request.config.option.eval_out_dir or str(
#         EVAL_DIR / "results-lseval-troubleshooting"
#     )
#     _run_troubleshooting_lseval(
#         MCP_EVAL_DATA,
#         Path(out_dir_base) / "troubleshooting" / "mcp",
#     )


@pytest.mark.lseval
def test_lseval_troubleshooting_suite_disabled() -> None:
    """Placeholder while troubleshooting LSEval implementation is commented out."""
    pytest.skip(
        "Troubleshooting LSEval disabled; uncomment the block in this file to re-enable."
    )
