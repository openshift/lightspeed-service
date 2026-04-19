# Authentication & Authorization

Control which users can access the OpenShift LightSpeed service and what they are permitted to do, integrating with Kubernetes RBAC in production and providing bypass modes for development.

## Behavioral Rules

1. The system must support exactly three authentication modules: `k8s`, `noop`, and `noop-with-token`.
2. The authentication module must be selected once at startup from configuration and must not change for the lifetime of the process.
3. If no authentication module is specified in configuration, the system must default to `k8s`.
4. If an unsupported module value is specified, the system must reject the configuration at startup.

### k8s Module (Production)

5. The system must extract a bearer token from the HTTP `Authorization` header on every request.
6. If the `Authorization` header is absent, the system must reject the request with HTTP 401.
7. If the header is present but does not contain a valid bearer token, the system must reject the request with HTTP 401.
8. The system must validate the bearer token by submitting a Kubernetes TokenReview to the cluster API.
9. If the TokenReview indicates the token is not authenticated, the system must reject the request with HTTP 403.
10. After successful authentication, the system must check the user's permissions by submitting a Kubernetes SubjectAccessReview against a synthetic non-resource path using the `get` verb.
11. The SubjectAccessReview must include the user's username and group memberships from the TokenReview result.
12. If the SubjectAccessReview denies access, the system must reject the request with HTTP 403.
13. On successful authentication and authorization, the system must return the user's UID, username, bearer token, and a flag indicating that user ID override is not permitted.
14. When the authenticated username is `kube:admin`, the system must replace the user ID with the cluster ID retrieved from the OpenShift ClusterVersion resource.
15. The k8s module must never allow callers to override the authenticated user ID.

### noop Module (Development)

16. The system must skip all authentication and authorization checks.
17. The system must accept an optional `user_id` query parameter; when absent, it must use the default nil UUID identity.
18. The system must allow callers to override the user ID (the skip-user-ID-check flag must be true).
19. The system must never provide a user token (the returned token must be empty).

### noop-with-token Module (Development with Token Forwarding)

20. The system must not validate the bearer token against Kubernetes.
21. The system must extract the bearer token from the `Authorization` header and return it for downstream use (e.g., MCP server requests that require a Kubernetes token).
22. If the `Authorization` header is absent, the system must reject the request with HTTP 400.
23. If the header is present but contains no valid bearer token, the system must reject the request with HTTP 400.
24. The system must accept an optional `user_id` query parameter; when absent, it must use the default nil UUID identity.
25. The system must allow callers to override the user ID (the skip-user-ID-check flag must be true).

### Dev Auth Override

26. When the dev-config auth-disable flag is set, the k8s and noop-with-token modules must bypass all checks and return the default identity with no token, regardless of request headers.
27. Bypassed auth checks must log a warning unless warning suppression is enabled in the logging configuration.

### Permission Scopes

28. The system must define two permission scopes, each mapped to a synthetic non-resource path for SubjectAccessReview evaluation:
    - **ols-access**: guards all query, conversation, feedback, tool-approval, MCP, and authorization-check endpoints.
    - **metrics-access**: guards the Prometheus metrics endpoint.
29. Each endpoint must declare exactly one permission scope at module load time.
30. The synthetic paths are not real HTTP routes; they exist solely for Kubernetes RBAC policy evaluation.

### User Identity Contract

31. Every authentication module must return a four-element tuple: (user_id, username, skip_user_id_check, user_token).
32. The user_id must be in UUID format.
33. The default identity must use the nil UUID (`00000000-0000-0000-0000-000000000000`) and a fixed default username.
34. [PLANNED: OLS-1393] The system must read the authenticated user's roles from the cluster and make them available to downstream processing.
35. [PLANNED: OLS-1394] The system must filter content returned to the user based on their roles.

### Cluster Information

36. The system must retrieve the cluster ID from the OpenShift ClusterVersion custom resource (`spec.clusterID`).
37. When not running inside a Kubernetes cluster, the cluster ID must default to `"local"`.
38. The cluster ID must be cached for the lifetime of the process after first retrieval.
39. The system must retrieve the cluster version from the ClusterVersion resource (`status.desired.version`).
40. If the cluster version cannot be retrieved, the system must fall back to `"unknown"` rather than failing.
41. The cluster version must be cached for the lifetime of the process (OLS pods restart during cluster upgrades, bounding staleness).
42. When not running inside a Kubernetes cluster, the cluster version must always be `"unknown"`.

### Kubernetes Client Lifecycle

43. The Kubernetes API client must be initialized at most once per process (singleton).
44. The singleton must support three configuration sources in priority order: explicit cluster API URL and auth token from config, in-cluster service account credentials, and local kubeconfig file.
45. The Kubernetes API host may be overridden via configuration.
46. TLS verification for the Kubernetes API may be disabled via configuration.
47. A custom CA certificate path for the Kubernetes API may be provided via configuration.

## Configuration Surface

| Field path | Type | Default | Description |
|---|---|---|---|
| `ols_config.authentication_config.module` | string | `"k8s"` | Auth module: `k8s`, `noop`, or `noop-with-token` |
| `ols_config.authentication_config.skip_tls_verification` | bool | `false` | Disable TLS verification for Kubernetes API calls |
| `ols_config.authentication_config.k8s_cluster_api` | URL | (from kubeconfig) | Override Kubernetes API server URL |
| `ols_config.authentication_config.k8s_ca_cert_path` | file path | (from kubeconfig) | Custom CA certificate for Kubernetes API TLS |
| `dev_config.disable_auth` | bool | `false` | Bypass all auth checks (dev only) |
| `dev_config.k8s_auth_token` | string | (none) | Override bearer token for Kubernetes API client initialization |
| `ols_config.logging_config.suppress_auth_checks_warning_in_log` | bool | `false` | Suppress repeated auth-bypass warnings in dev mode |

## Constraints

1. The authentication module selection is immutable after process startup.
2. The Kubernetes client singleton is initialized once and shared for the lifetime of the process; it cannot be reconfigured at runtime.
3. The `noop` and `noop-with-token` modules are intended for development only and must not be used in production deployments.
4. The `noop-with-token` module uses HTTP 400 (not 401) for missing or invalid tokens because the failure is a malformed request, not an authentication rejection.
5. The cluster ID is required for `kube:admin` identity mapping; if it cannot be retrieved in-cluster, the system must raise an error.
6. The cluster version is non-critical (used for prompt enrichment); retrieval failure must never block request processing.
7. The SubjectAccessReview uses the `get` verb against a non-resource path; no actual Kubernetes resource is accessed.

## Planned Changes

- [OLS-1393] Read user roles from the cluster during authentication and attach them to the request context.
- [OLS-1394] Use the retrieved user roles to filter content returned by the service.
