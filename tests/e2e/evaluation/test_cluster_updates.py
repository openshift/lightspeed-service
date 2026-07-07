"""Cluster-updates evaluation tests using OpenAI GPT-4o-mini and gpt-4o-mini judge."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
LSEVAL_BIN = PROJECT_ROOT / ".venv" / "bin" / "lightspeed-eval"
SYSTEM_CONFIG = EVAL_DIR / "system_cluster_updates.yaml"
EVAL_DATA = EVAL_DIR / "eval_data_cluster_updates.yaml"


def _ensure_lseval_installed() -> None:
    """Install the lightspeed-evaluation package via uv if absent.

    Uses the version pinned in pyproject.toml under [project.optional-dependencies].lseval,
    ensuring a specific (tested) release rather than HEAD.
    """
    if LSEVAL_BIN.exists():
        return

    uv_path = shutil.which("uv")
    if not uv_path:
        raise FileNotFoundError("uv command not found in PATH")

    subprocess.run(  # noqa: S603
        [uv_path, "sync", "--extra", "lseval"],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def _resolve_ols_url() -> str:
    """Return the OLS base URL, preferring the live pytest client over env var."""
    client = getattr(pytest, "ols_url", None)
    if client:
        return client.rstrip("/")
    return os.getenv("OLS_URL", "http://localhost:8080").rstrip("/")


def _get_ols_token() -> str:
    """Extract the bearer token from the pytest HTTP client or environment."""
    client = getattr(pytest, "client", None)
    if client is not None:
        auth_header: str = client.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if token:
            return token
    return os.getenv("API_KEY", "")


def _run_lseval(eval_data: Path, out_dir: Path) -> None:
    """Run lightspeed-eval with the given data file and assert artefacts are produced."""
    _ensure_lseval_installed()

    # Ensure output directory is writable
    out_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {out_dir}")

    ols_url = _resolve_ols_url()

    with open(SYSTEM_CONFIG, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    config["api"]["api_base"] = ols_url

    # Use try/finally to ensure cleanup even on exceptions
    tmp_config_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(EVAL_DIR)
        ) as tmp:
            yaml.dump(config, tmp)
            tmp_config_path = tmp.name

        env = os.environ.copy()

        # Validate OPENAI_API_KEY is present for the judge LLM
        if "OPENAI_API_KEY" not in env:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required for cluster-updates evaluation. "
                "Ensure test-cluster-updates.sh exports it before running this test."
            )

        token = _get_ols_token()
        if token:
            env["API_KEY"] = token

        result = subprocess.run(  # noqa: S603
            [
                str(LSEVAL_BIN),
                "--system-config",
                tmp_config_path,
                "--eval-data",
                str(eval_data),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    finally:
        # Always clean up the temp file
        if tmp_config_path and os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)

    print("--- lightspeed-eval stdout ---")
    print(result.stdout)
    if result.stderr:
        print("--- lightspeed-eval stderr ---")
        print(result.stderr)

    assert result.returncode == 0, (
        f"lightspeed-eval exited with code {result.returncode}.\n"
        f"stderr:\n{result.stderr}"
    )

    csv_files = list(out_dir.glob("*_detailed.csv"))
    assert csv_files, f"No detailed CSV artefacts found in {out_dir}"

    json_files = list(out_dir.glob("*_summary.json"))
    assert json_files, f"No summary JSON artefacts found in {out_dir}"

    with open(json_files[0], encoding="utf-8") as fh:
        overall = json.load(fh)["summary_stats"]["overall"]
    assert overall["ERROR"] == 0, (
        f"{overall['ERROR']}/{overall['TOTAL']} evaluations errored "
        f"(error_rate={overall['error_rate']:.1f}%)."
    )


@pytest.mark.cluster_updates
def test_cluster_updates(request: pytest.FixtureRequest) -> None:
    """Run cluster-updates eval suite (18 conversations, 35 evaluations).

    Uses GPT-4o-mini and gpt-4o-mini judge.
    """
    out_dir_base = request.config.option.eval_out_dir or str(
        EVAL_DIR / "results-cluster-updates"
    )
    _run_lseval(EVAL_DATA, Path(out_dir_base) / "cluster-updates")
