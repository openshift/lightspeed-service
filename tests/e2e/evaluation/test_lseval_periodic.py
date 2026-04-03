r"""LSEval evaluation tests across all supported OLS providers.

Each provider uses OpenAI GPT-4.1-mini as the judge LLM.  The model
under evaluation is whatever is configured in the per-provider system
config (eval/system_<provider>_lseval.yaml).

Supported providers: openai, watsonx, azure_openai, rhoai_vllm, rhelai_vllm.

Local usage
-----------
1. Start OLS locally with the provider you want to evaluate::

       OLS_CONFIG_FILE=olsconfig-openai.yaml make run   # example: openai

2. Export the judge LLM key and, for vLLM-based providers, the model name::

       export OPENAI_API_KEY=<your-key>
       export LSEVAL_OLS_MODEL=ibm/granite-3.1-8b-instruct  # rhoai_vllm / rhelai_vllm only

3. Run the eval for a single provider::

       pytest tests/e2e/evaluation/test_lseval_periodic.py \\
           -m lseval --lseval_provider openai \\
           --eval_out_dir /tmp/my-eval-results

   Or run all providers (OLS must be configured with all of them)::

       pytest tests/e2e/evaluation/test_lseval_periodic.py -m lseval
"""

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
EVAL_DATA_FULL = EVAL_DIR / "eval_data.yaml"

_PROVIDER_CONFIGS: dict[str, Path] = {
    "openai": EVAL_DIR / "system_openai_lseval.yaml",
    "watsonx": EVAL_DIR / "system_watsonx_lseval.yaml",
    "azure_openai": EVAL_DIR / "system_azure_openai_lseval.yaml",
    "rhoai_vllm": EVAL_DIR / "system_rhoai_vllm_lseval.yaml",
    "rhelai_vllm": EVAL_DIR / "system_rhelai_vllm_lseval.yaml",
}


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
    """Extract the bearer token from the pytest HTTP client if available."""
    client = getattr(pytest, "client", None)
    if client is None:
        return ""
    auth_header: str = client.headers.get("Authorization", "")
    return auth_header.removeprefix("Bearer ").strip()


def _run_lseval(eval_data: Path, out_dir: Path, system_config: Path) -> None:
    """Run lightspeed-eval with the given data file and assert artefacts are produced.

    Args:
        eval_data: Path to the evaluation dataset YAML.
        out_dir: Directory where evaluation artefacts are written.
        system_config: Provider-specific lseval system config YAML.
    """
    _ensure_lseval_installed()
    out_dir.mkdir(parents=True, exist_ok=True)

    ols_url = _resolve_ols_url()

    with open(system_config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    config["api"]["api_base"] = ols_url

    if model_override := os.getenv("LSEVAL_OLS_MODEL"):
        config["api"]["model"] = model_override

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(EVAL_DIR)
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name

    env = os.environ.copy()

    token = _get_ols_token()
    if token:
        env["API_KEY"] = token

    try:
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


@pytest.mark.lseval
@pytest.mark.parametrize("provider", list(_PROVIDER_CONFIGS.keys()))
def test_lseval_periodic(request: pytest.FixtureRequest, provider: str) -> None:
    """Run LSEval full dataset with GPT-4.1-mini as judge for the given OLS provider.

    Args:
        request: Pytest fixture request object.
        provider: OLS provider under evaluation (parametrized).
    """
    selected = request.config.option.lseval_provider
    if selected and selected != provider:
        pytest.skip(f"--lseval_provider={selected!r}: skipping provider {provider!r}")

    out_dir_base = request.config.option.eval_out_dir or str(
        EVAL_DIR / "results-lseval-periodic"
    )
    _run_lseval(
        EVAL_DATA_FULL,
        Path(out_dir_base) / "lseval" / provider,
        _PROVIDER_CONFIGS[provider],
    )
