# Tool-result inspection evaluation

This directory contains labeled cases for the Classic tool-result inspection path.

## Corpus schema

Each YAML list item contains:

- `case_id`: unique stable identifier.
- `tool_name`: tool name supplied to the service.
- `result_type`: `result` or `error`.
- `content`: untrusted tool output or tool-generated error text.
- `expected_outcome`: `benign` or `malicious`.
- `expected_category`: `none` for benign cases, or one of `instruction_override`,
  `role_change`, `prompt_extraction`, `data_exfiltration`, `tool_manipulation`, or
  `unknown` for malicious cases.
- `tags`: one or more labels used for report grouping.

`corpus_smoke.yaml` is a small development dataset. `corpus_full.yaml` provides
broader category, OpenShift output, error, multilingual, JSON, and chunk-boundary
coverage.

## Run the evaluation

Start a Classic OLS deployment, then run the smoke corpus:

```bash
API_KEY="$OLS_API_KEY" \
uv run python -m eval.tool_result_inspection.runner \
  --base-url http://localhost:8080 \
  --dataset eval/tool_result_inspection/corpus_smoke.yaml \
  --output-dir /tmp/ols-tool-result-inspection \
  --provider openai \
  --model gpt-5-mini
```

Use `corpus_full.yaml` for the complete opt-in evaluation. The runner reads
`API_KEY` from the environment and writes only redacted case results and
aggregates. It does not write tool-result content, classifier prompts, model
reasoning, or credentials to output files.

The deployed evaluation fixture must map each case ID to the corresponding
corpus content in the model-visible tool result. The runner sends only the case
ID and tool name in the user query; it does not send corpus content as user
input. This keeps the cases on the tool-result path that the classifier is
intended to inspect.

Real-model evaluation can incur provider costs. Keep result directories outside
the repository and review them before sharing.
