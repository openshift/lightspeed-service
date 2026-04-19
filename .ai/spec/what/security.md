# Security

The service must protect customer data, enforce transport-layer encryption, redact sensitive information before it leaves the cluster, and operate within hardened, FIPS-capable container environments.

## Behavioral Rules

### TLS for service endpoints

1. All service endpoints (API, metrics) must support TLS encryption using administrator-supplied certificates and private keys.
2. The service must support three TLS security profiles aligned with OpenShift cluster-level configuration: `IntermediateType` (minimum TLS 1.2), `ModernType` (minimum TLS 1.3), and `Custom` (administrator-specified ciphers and minimum version).
3. The service must reject the `OldType` profile at configuration time. The absolute minimum permitted TLS version is 1.2 (`VersionTLS12`); any version below this must be rejected regardless of profile.
4. For non-Custom profiles, only ciphers defined in that profile's allowed list may be used. If a cipher not in the profile's list appears in configuration, the service must reject it at load time.
5. When no TLS security profile is configured, the service must use default cipher suites that include the `IntermediateType` ciphers plus FIPS-compliant CBC ciphers (`ECDHE-ECDSA-AES128-SHA256`, `ECDHE-RSA-AES128-SHA256`, `ECDHE-ECDSA-AES256-SHA384`, `ECDHE-RSA-AES256-SHA384`, `DHE-RSA-AES128-SHA256`, `DHE-RSA-AES256-SHA256`) for compatibility with HAProxy reencrypt routes.
6. The service must support certificate rotation: when certificates change on disk, the service must pick up the new certificates on restart or configuration reload.
7. The private key password must be read from a file referenced by `ols_config.tls_config.tls_key_password_path`, never stored as a plaintext value in configuration.
8. [PLANNED: OLS-2866] The service's TLS endpoint configuration must be verifiable by automated TLS scanning tools.

### FIPS compliance

9. The service must carry a "Designed for FIPS" designation and use FIPS-validated cryptographic modules when the underlying platform enforces FIPS mode.
10. Container images must be built on FIPS-compatible base images.
11. The build and CI pipeline must verify FIPS compliance and reject non-compliant builds.
12. FIPS mode is not required by default but must be fully functional when enforced by the platform.

### PII redaction

13. The service must support an ordered list of configurable regex-based query filters that redact personally identifiable information from user queries before they are sent to the LLM provider.
14. Each filter consists of a name, a compiled regex pattern, and a replacement string. Filters are applied sequentially in the order defined.
15. The meaning of the question must be preserved while sensitive data (IP addresses, email addresses, names, account numbers, etc.) is replaced with safe placeholders.
16. The service does not impose a fixed set of redaction rules; administrators define which patterns to apply.
17. Filter patterns must be validated as compilable regular expressions during configuration loading. Invalid patterns must cause a configuration error.

### Data sensitivity

18. User queries, conversation history, and feedback must be processed and stored within the customer's cluster.
19. The only external communication the service initiates is to the configured LLM provider endpoint. Queries sent to the LLM provider must pass through the configured redaction filters first.
20. Transcript and feedback data must be stored on the cluster's local persistent storage and must not be transmitted externally by the service.

### Security headers

21. The service must set `X-Content-Type-Options: nosniff` on all HTTP responses except health-check and metrics endpoints (`/readiness`, `/liveness`, `/metrics`).
22. When TLS is enabled, the service must set `Strict-Transport-Security: max-age=31536000; includeSubDomains` on all non-health-check, non-metrics responses.

### Credential handling

23. API keys, passwords, and other secrets must never appear as plaintext values in the configuration file. All credentials must be specified as file paths; the service reads the secret value from the referenced file at startup.
24. Credential files may be individual files or directories containing named secret files (e.g., `apitoken`, `client_id`, `tenant_id`, `client_secret`).
25. Sensitive HTTP request headers (`authorization`, `proxy-authorization`, `cookie`) and response headers (`www-authenticate`, `proxy-authenticate`, `set-cookie`) must be redacted in debug-level request/response logs.

### TLS for provider connections

26. The service must support custom CA certificates for providers using private or enterprise certificate authorities, configured via `ols_config.extra_ca`.
27. The service must aggregate extra CA certificates into a merged certificate store (combining system-trusted certificates from certifi with the extra CAs) and use that store when connecting to providers and MCP servers.
28. TLS security profiles (cipher suites, minimum TLS version) must be configurable per LLM provider, independent of the service's own endpoint TLS settings.
29. HTTPS proxy support must be available with configurable proxy URL, proxy CA certificate, and no-proxy host list. Proxy configuration must also be derivable from `https_proxy`/`HTTPS_PROXY` and `no_proxy` environment variables.

### Tool approval security

30. Pending tool approval state must be held in-memory only. Approval state must not survive a process restart.
31. Each approval request must have a configurable timeout. If no decision is received within the timeout, the outcome must be `timeout` (fail-closed). The approval request must be cleaned up from memory after resolution or timeout.
32. Approval flow is only active for streaming requests. Non-streaming requests must never enter the approval workflow.
33. The approval strategy must be one of: `never` (no approval required), `always` (all tool calls require approval), or `tool_annotations` (approval required unless the tool declares `readOnlyHint: true`).

### MCP credential handling

34. MCP server authorization headers support three resolution modes: (a) a file path, whose contents are read at config load time; (b) the `kubernetes` placeholder, which is replaced at request time with a `Bearer <token>` derived from the authenticated user's Kubernetes token; (c) the `client` placeholder, which is replaced at request time with a header value supplied by the calling client.
35. The `kubernetes` placeholder must only be permitted when the authentication module is `k8s` or `noop-with-token`. If used with any other authentication module, the MCP server must be excluded at config validation time.
36. MCP connections must use the aggregated certificate store when a certificate directory is configured, ensuring custom CA certificates apply to MCP server communication.
37. [PLANNED: OLS-2717] MCP servers (including openshift-mcp-server) must support TLS-encrypted connections.

### Disconnected / air-gapped environments

38. The service must not require any outbound internet connectivity beyond the configured LLM provider endpoint (which may be on-premises).
39. All dependencies (container images, RAG indexes, embedding models) must be distributable via disconnected-compatible mechanisms (mirrored registries, local storage).

### Network policies

40. The deployment must support network policies that restrict traffic to only the necessary communication paths, protecting against unintended data leaks and lateral movement.

### Read-only root filesystem

41. Containers must run with a read-only root filesystem to minimize the attack surface and comply with security hardening requirements.

## Configuration Surface

| Field path | Purpose |
|---|---|
| `ols_config.tls_config.tls_certificate_path` | Path to the TLS certificate file for service endpoints |
| `ols_config.tls_config.tls_key_path` | Path to the TLS private key file |
| `ols_config.tls_config.tls_key_password_path` | Path to a file containing the private key password |
| `ols_config.tls_security_profile.type` | TLS profile: `IntermediateType`, `ModernType`, or `Custom` |
| `ols_config.tls_security_profile.minTLSVersion` | Minimum TLS version (e.g., `VersionTLS12`, `VersionTLS13`) |
| `ols_config.tls_security_profile.ciphers` | List of allowed cipher suite names (Custom profile or subset of profile ciphers) |
| `ols_config.extra_ca` | List of extra CA certificate file paths to trust |
| `ols_config.certificate_directory` | Directory where the merged certificate store is written |
| `ols_config.query_filters[].name` | Human-readable name for a redaction filter |
| `ols_config.query_filters[].pattern` | Regex pattern to match sensitive data |
| `ols_config.query_filters[].replace_with` | Replacement string for matched data |
| `llm_providers[].proxy_config.proxy_url` | HTTPS proxy URL for provider connections (falls back to `https_proxy`/`HTTPS_PROXY` env var) |
| `llm_providers[].proxy_config.proxy_ca_cert_path` | CA certificate for the HTTPS proxy |
| `llm_providers[].proxy_config.no_proxy_hosts` | List of hosts to bypass the proxy (falls back to `no_proxy` env var) |
| `llm_providers[].credentials_path` | Path to directory or file containing provider credentials |
| `mcp_servers.servers[].headers` | Map of header names to file paths, `kubernetes`, or `client` |
| `ols_config.tools_approval.approval_type` | Approval strategy: `never`, `always`, or `tool_annotations` |
| `ols_config.tools_approval.approval_timeout` | Seconds to wait for an approval decision before timing out (default 600, minimum 1) |
| `dev_config.disable_tls` | Disable TLS for development (suppresses HSTS header, bypasses certificate requirements) |

## Constraints

- TLS 1.0 and TLS 1.1 are unconditionally prohibited. The `OldType` profile is rejected at config load.
- The minimum TLS version floor is `VersionTLS12`. Any configured `minTLSVersion` below this value must cause a configuration error.
- Cipher validation for non-Custom profiles is strict: only ciphers in the profile's defined set are accepted.
- Query filter patterns must be valid Python regular expressions. Invalid patterns must cause a configuration error at startup, not a runtime failure.
- Credential values never appear in the configuration YAML. Only file system paths are accepted.
- The redacted header sets for request logging (`authorization`, `proxy-authorization`, `cookie`) and response logging (`www-authenticate`, `proxy-authenticate`, `set-cookie`) are fixed and not configurable.
- Tool approval state is ephemeral (in-memory, per-process). A restart clears all pending approvals.
- The approval timeout prevents indefinite waits: if no decision arrives, the tool call is not executed (fail-closed).
- The `kubernetes` MCP header placeholder is valid only with `k8s` or `noop-with-token` authentication modules. Other modules cause the server to be excluded.
- The merged certificate store is written to `certificate_directory` at startup. If `certificate_directory` is not set, extra CA certificates are not aggregated.
- Containers must operate with a read-only root filesystem; writable paths are limited to explicitly mounted volumes.

## Planned Changes

- [PLANNED: OLS-2866] Automated TLS scanning verification of OLS service endpoints using tls-scanner.
- [PLANNED: OLS-2717] Enable TLS on openshift-mcp-server connections.
