r"""LSEval presubmit tests — short 10-question dataset across all providers.

Runs ``eval/eval_data_short.yaml`` (10 questions) against the full provider matrix.
The periodic suite uses the full 797-question dataset; this suite is a lightweight
smoke test intended for presubmit CI.

Each provider uses the judge LLM defined in ``eval/system_<provider>_lseval.yaml``
(OpenAI ``gpt-5-mini`` for scoring).

Local usage
-----------
1. Start OLS locally with the provider you want to evaluate::

       OLS_CONFIG_FILE=olsconfig-openai.yaml make run

2. Export the judge LLM key::

       export OPENAI_API_KEY=<your-key>

3. Run the eval for a single provider::

       pytest tests/e2e/evaluation/test_lseval_presubmit.py \
           -m lseval --lseval_provider openai \
           --eval_out_dir /tmp/my-eval-results
"""

from pathlib import Path

import pytest

from tests.e2e.evaluation.test_lseval_periodic import (
    EVAL_DIR,
    _run_lseval,
    _skip_reason_for_provider,
)

_LSEVAL_PRESUBMIT_PROVIDERS = (
    "openai",
    "watsonx",
    "azure_openai",
    "rhoai_vllm",
    "vertex_gemini",
    "vertex_claude",
    "bedrock_deepseek",
)

_PROVIDER_CONFIGS: dict[str, Path] = {
    "openai": EVAL_DIR / "system_openai_lseval.yaml",
    "watsonx": EVAL_DIR / "system_watsonx_lseval.yaml",
    "azure_openai": EVAL_DIR / "system_azure_openai_lseval.yaml",
    "rhoai_vllm": EVAL_DIR / "system_rhoai_vllm_lseval.yaml",
    "vertex_gemini": EVAL_DIR / "system_vertex_gemini_lseval.yaml",
    "vertex_claude": EVAL_DIR / "system_vertex_claude_lseval.yaml",
    "bedrock_deepseek": EVAL_DIR / "system_bedrock_deepseek_lseval.yaml",
}

EVAL_DATA_SHORT = EVAL_DIR / "eval_data_short.yaml"


@pytest.mark.lseval
@pytest.mark.parametrize("provider", _LSEVAL_PRESUBMIT_PROVIDERS)
def test_lseval_presubmit(request: pytest.FixtureRequest, provider: str) -> None:
    """Run LSEval short dataset for the given OLS provider (presubmit smoke test).

    Args:
        request: Pytest fixture request object.
        provider: OLS provider under evaluation (parametrized).
    """
    if reason := _skip_reason_for_provider(request, provider):
        pytest.skip(reason)

    out_dir_base = request.config.option.eval_out_dir or str(
        EVAL_DIR / "results-lseval-presubmit"
    )
    _run_lseval(
        EVAL_DATA_SHORT,
        Path(out_dir_base) / "lseval" / provider,
        _PROVIDER_CONFIGS[provider],
    )
