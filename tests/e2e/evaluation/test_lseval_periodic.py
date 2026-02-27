"""LSEval periodic evaluation test using WatsonX Granite and OpenAI judge."""

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
SYSTEM_CONFIG = EVAL_DIR / "system_watsonx_granite.yaml"
EVAL_DATA = EVAL_DIR / "eval_data_short.yaml"


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


@pytest.mark.lseval
def test_lseval_periodic(request: pytest.FixtureRequest) -> None:
    """Run LSEval using WatsonX Granite as the OLS backend and OpenAI GPT as judge.

    Deploys a temporary system config patched with the live OLS URL, executes
    lightspeed-eval against eval_data_short.yaml, then asserts that the result
    artefacts (CSV and JSON summary) were produced.
    """
    _ensure_lseval_installed()

    out_dir_base = request.config.option.eval_out_dir or str(
        EVAL_DIR / "results-lseval-periodic"
    )
    out_dir = Path(out_dir_base) / "lseval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ols_url = _resolve_ols_url()

    with open(SYSTEM_CONFIG, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    config["api"]["api_base"] = ols_url

    token = _get_ols_token()
    if token:
        config.setdefault("api", {})["api_key"] = token

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(EVAL_DIR)
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name

    env = os.environ.copy()

    try:
        result = subprocess.run(  # noqa: S603
            [
                str(LSEVAL_BIN),
                "--system-config",
                tmp_config_path,
                "--eval-data",
                str(EVAL_DATA),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    finally:
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


def _get_ols_token() -> str:
    """Extract the bearer token from the pytest HTTP client if available."""
    client = getattr(pytest, "client", None)
    if client is None:
        return ""
    auth_header: str = client.headers.get("Authorization", "")
    return auth_header.removeprefix("Bearer ").strip()
