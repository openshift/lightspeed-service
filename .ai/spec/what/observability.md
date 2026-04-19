# Observability

The service exposes Prometheus metrics, records conversation transcripts and user feedback to persistent storage, logs HTTP request/response details with header redaction, and optionally integrates with Pyroscope for continuous profiling.

## Behavioral Rules

### Prometheus Metrics

1. The service exposes the following Prometheus metrics, all prefixed with `ols_`:

   | Metric name | Type | Labels | Description |
   |---|---|---|---|
   | `ols_rest_api_calls_total` | Counter | `path`, `status_code` | Total REST API calls. Excludes `/metrics` endpoint. |
   | `ols_response_duration_seconds` | Histogram | `path` | Duration of request handling. Includes all registered routes. |
   | `ols_llm_calls_total` | Counter | `provider`, `model` | Total LLM invocations. |
   | `ols_llm_calls_failures_total` | Counter | _(none)_ | Total failed LLM calls across all providers. |
   | `ols_llm_token_sent_total` | Counter | `provider`, `model` | Cumulative input tokens sent to LLMs. |
   | `ols_llm_token_received_total` | Counter | `provider`, `model` | Cumulative output tokens received from LLMs. |
   | `ols_llm_reasoning_token_total` | Counter | `provider`, `model` | Cumulative reasoning summary tokens received from LLMs. |
   | `ols_provider_model_configuration` | Gauge | `provider`, `model` | Configured provider/model combinations. Value `1` for the default, `0` for others. |

2. The `_created` timestamp metadata on all counters must be suppressed (via `disable_created_metrics()`).

3. The `ols_rest_api_calls_total` counter must NOT be incremented for requests to the `/metrics` path.

4. The `ols_response_duration_seconds` histogram must be recorded for all registered application routes, including `/metrics`.

5. The `ols_provider_model_configuration` gauge must be populated at startup from the configuration, before the first Prometheus scrape can occur. The default provider/model pair gets value `1`; all other configured pairs get `0`. If multiple provider entries resolve to the same `(provider_type, model_name)` key, a pair already set to `1` must not be overwritten with `0`.

### Metrics Endpoint

6. The metrics endpoint must be served at `GET /metrics`.

7. The response must use Prometheus text exposition format (`text/plain; version=0.0.4; charset=utf-8`).

8. Access to the metrics endpoint requires a separate permission scope, authorized via the virtual path `/ols-metrics-access`. A user authorized for the main API (`/ols-access`) is not automatically authorized to read metrics.

### Token Counting

9. For each LLM interaction, the service must count input tokens, output tokens, and reasoning tokens using a tokenizer (tiktoken `cl100k_base`).

10. Input tokens are counted when the LLM call starts (from the prompt strings). Output tokens are counted as each text token is yielded. Reasoning tokens are counted separately from output tokens when the LLM emits structured content blocks with `type: "reasoning"` containing `summary` sub-blocks.

11. The number of LLM calls within a single user request must be tracked (relevant for agentic tool-calling loops that invoke the LLM multiple times).

12. Upon completion of an LLM interaction, token counts must be accumulated into the Prometheus counters `ols_llm_token_sent_total`, `ols_llm_token_received_total`, `ols_llm_reasoning_token_total`, and `ols_llm_calls_total`, all labeled by `provider` and `model`.

13. Token counts must also be available per-request for quota enforcement and optionally recorded in the usage history database.

### Transcript Recording

14. The service must optionally record conversation transcripts as JSON files in a configurable directory.

15. Each transcript file must contain: metadata (provider, model, user ID, conversation ID, query mode, ISO 8601 timestamp), the redacted user query, the LLM response, RAG chunks (as dicts), a truncation flag, merged tool calls and results, and attachments.

16. Transcripts are organized under `{transcripts_storage}/{user_id}/{conversation_id}/{suid}.json`.

17. Transcript recording is independently enabled or disabled. When disabled, no files are written and a debug log message is emitted. [PLANNED: OLS-1805 -- enhance transcripts with per-request token usage data]

### User Feedback

18. The service must accept user feedback via `POST /v1/feedback`. The request must include `conversation_id`, `user_question`, `llm_response`, and at least one of `sentiment` (integer, must be `-1` or `1`) or `user_feedback` (free-text string).

19. Feedback is stored as individual JSON files in the configured feedback storage directory, each named `{suid}.json` and containing `user_id`, `timestamp`, and all feedback fields.

20. Feedback collection is independently enabled or disabled. When disabled, the `POST` endpoint returns HTTP 403.

21. The feedback status endpoint `GET /v1/feedback/status` must report whether feedback collection is enabled. This endpoint must NOT require authentication.

### Request/Response Logging

22. The service must log HTTP request and response details at DEBUG level, including client host/port, headers, and body content.

23. The following request headers must be redacted (replaced with `XXXXX`) in log output: `authorization`, `proxy-authorization`, `cookie`.

24. The following response headers must be redacted in log output: `www-authenticate`, `proxy-authenticate`, `set-cookie`.

25. When `suppress_metrics_in_log` is enabled and the request path is `/metrics`, the request/response logging middleware must skip logging entirely.

### Continuous Profiling

26. The service must optionally integrate with Pyroscope for continuous CPU profiling. When configured, it registers with the Pyroscope server using application name `lightspeed-service`, with `oncpu=True` and `gil_only=True`.

27. Pyroscope integration is activated only when a URL is provided in `dev_config.pyroscope_url` and the server is reachable. When not configured, no profiling code is loaded and no overhead is incurred.

## Configuration Surface

| Config field path | Type | Default | Purpose |
|---|---|---|---|
| `ols_config.logging_config.app_log_level` | string (log level name) | `info` | Log level for `ols.*` loggers |
| `ols_config.logging_config.lib_log_level` | string (log level name) | `warning` | Log level for root/third-party loggers |
| `ols_config.logging_config.uvicorn_log_level` | string (log level name) | `warning` | Log level for Uvicorn HTTP server |
| `ols_config.logging_config.suppress_metrics_in_log` | bool | `false` | Suppress `/metrics` requests from debug logs |
| `ols_config.user_data_collection.feedback_disabled` | bool | `true` | Disable user feedback collection |
| `ols_config.user_data_collection.feedback_storage` | string (path) | _(none)_ | Directory for feedback JSON files (required when enabled) |
| `ols_config.user_data_collection.transcripts_disabled` | bool | `true` | Disable transcript recording |
| `ols_config.user_data_collection.transcripts_storage` | string (path) | _(none)_ | Directory for transcript JSON files (required when enabled) |
| `dev_config.pyroscope_url` | string (URL) | _(none)_ | Pyroscope server URL; omit to disable profiling |

## Constraints

1. All metric names use the `ols_` prefix. No other prefix is permitted.
2. The six redacted header names (`authorization`, `proxy-authorization`, `cookie`, `www-authenticate`, `proxy-authenticate`, `set-cookie`) are defined as frozen sets in `constants.py` and must not be logged in plaintext under any log level.
3. Enabling feedback or transcripts without providing the corresponding `*_storage` path is a validation error at startup.
4. Sentiment values are restricted to exactly `-1` or `1`; any other integer is rejected with a validation error.
5. The feedback `POST` endpoint requires authentication (via `/ols-access`); the feedback `GET /status` endpoint does not.
6. The metrics endpoint permission scope (`/ols-metrics-access`) is independent of the main API scope (`/ols-access`).

## Planned Changes

- [PLANNED: OLS-1279] The `ols_response_duration_seconds` histogram currently measures total request handling time, not isolated LLM call duration. A dedicated LLM-only duration metric is needed.
- [PLANNED: OLS-1805] Transcripts should be enhanced to include per-request token usage (input, output, reasoning token counts).
