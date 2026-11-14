# Classic Chat Tool-Output Redaction

OLS-4878 implements the Classic chat execution-loop boundary for tool-output redaction. MCP Apps are explicitly out of scope.

## Scope: chat tool results, not MCP Apps

Redaction applies to results handled by the chat tool executor, including offload read/search results. It does **not** apply to MCP Apps resource fetches (`POST /v1/mcp-apps/resources`) or direct app tool calls (`POST /v1/mcp-apps/tools/call`). These endpoints bypass the chat execution loop and do not acquire its redaction or replacement metadata merely because `ols_config.tool_output_redaction` is enabled. Scope depends on the execution path, not the MCP server or tool name.

An MCP App server can fetch data internally and embed it in an HTML/JS/CSS resource, or provide data separately through app tool calls. OLS cannot assume all sensitive data is available as a separate plain-text tool result. Regex substitution over application source can corrupt markup, scripts, attributes, and embedded data; changing structured app data can also break application behavior. The chat text-redaction policy is therefore not a general, semantics-preserving sanitizer for MCP Apps.

This exclusion is not a streaming limitation: the current proxy awaits complete MCP results and returns JSON responses. Enabling chat tool-output redaction does not guarantee that app resources, app tool data, or rendered iframe content are redacted. MCP Apps protection would require a separately agreed app-aware data/rendering contract and is not a remaining implementation step in OLS-4878. See `what/mcp-apps.md`.

## Policies and configuration

1. Administrator-defined `ols_config.query_filters` apply only to prompts and attachments. Preserve configured order, replacements, and duplicate names. Absent/empty filters leave inputs unchanged. `ols/src/redaction/query_filters.py` adapts these filters to the shared engine without logging queries, attachments, or filter configuration.
2. `ols_config.tool_output_redaction` is a strict boolean, defaults to true, and preserves explicit false. The service compiles the fixed built-in sequential `MCP_RULES` policy from `ols/src/redaction/rulesets/mcp.py` once at module load. `AppConfig.tool_output_redaction_rules` returns those rules or `None` according to the current enablement flag; no reload-time matcher state is needed. MCP defaults never become query filters. Custom rules and policy revisioning are outside this implementation.
3. Preserve standalone IPv4, IPv6, and CIDRs under the built-in policy. Pattern matching is best-effort redaction, not complete PII detection, certified DLP, or pathological-regex protection.

## Execution boundary

4. Redact after retries and existing per-tool size guards, before building the result event. Cover successful results and bounded final tool-generated errors. Inspection is independent: when enabled it sees sanitized input; disabling inspection does not disable redaction. Service-generated approval rejections bypass this boundary.
5. Offloading writes original tool output to existing request-scoped disk storage without running the matcher over the huge payload. After successful offloading, omit `structured_content` from the returned result so it cannot bypass offloading as an attached full-size projection. Only text is stored; structured-only fields are not preserved. Small results and failed/skipped offloads retain their structured content and follow the normal sanitization path. Redact the initial bounded message and subsequent bounded read/search results before delivery. Existing storage access controls and cleanup remain in force; raw sensitive content can remain on disk until cleanup.
6. Sanitize text, structured content, and referenced documents. Convert Pydantic metadata to sanitized plain dictionaries rather than reconstructed models, so computed fields and custom serializers cannot restore original values during response serialization. Preserve the existing `RagChunk` dataclass representation. Bound structured/reference traversal to the per-tool character budget before matching each string. An oversized or unsupported metadata value, invalid sanitized mapping, or matcher failure fails closed with a controlled payload-free error; never forward the raw result as a fallback.
7. Sanitized results are the source for inspection, model context, references, tool-result SSE, and source audit/tool spans. No raw result content is newly logged or captured in spans. The later aggregate token budget and `<tool_data>` wrapper still affect model-facing copies only. Audit/SSE can use the complete sanitized result from the earlier per-tool size boundary.
8. Preserve whole-round inspection gating: do not emit any tool-result SSE or UI redaction metadata until all results in the concurrent round pass inspection. A passing sibling can still receive its sanitized per-result audit event even if another result fails.

## Explicit UI and telemetry metadata

9. Only results with replacements carry `redacted: true`, `redacted_count`, and `redaction_counts` in `ToolMessage.additional_kwargs` and the tool-result SSE data. Unchanged and opted-out results omit all three fields. Preserve the existing tool-call correlation ID. Counts are summed directly from the engine across returned text and metadata, with no deduplication: a value replaced in two fields counts as two replacements. The console rendering change is separate.
10. After redaction, log only replacement totals and rule/category counts. Changed tool spans use product attributes `lightspeed.tool_output.redacted`, `lightspeed.tool_output.redacted_count`, and JSON-encoded `lightspeed.tool_output.redaction_counts`. Runtime failures log a controlled message without payload or engine exception text. Disabling redaction intentionally allows raw bounded results through this boundary, including audit capture.

## Verification

Use the copied shared fixture and engine tests to verify sequential rules, placeholders, counts, and preserved network addresses. Boundary tests cover success/error text, opt-out, no-match metadata omission, structured projections, fail-closed behavior, bounded matching, and offload read/search rather than write-time redaction. Execution-loop tests verify sanitized inspection/model/SSE/audit/log content and metadata with inspection enabled and disabled. Existing concurrent-round rejection tests remain required.
