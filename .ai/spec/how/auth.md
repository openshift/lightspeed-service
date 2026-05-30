# Authentication -- Architecture

The auth system uses a strategy pattern to inject authentication/authorization logic into FastAPI endpoints. A factory selects one of three implementations based on configuration, and the selected strategy is bound at module import time via FastAPI's `Depends()`.

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `src/auth/auth.py` | `get_auth_dependency()`, `use_k8s_auth()` | Factory: selects auth implementation by config module name |
| `src/auth/auth_dependency_interface.py` | `AuthDependencyInterface` (ABC) | Contract: `async __call__(request) -> (uid, username, skip_check, token)` |
| `src/auth/k8s.py` | `K8sClientSingleton`, `AuthDependency`, `get_user_info()` | Production auth: Kubernetes TokenReview + SubjectAccessReview |
| `src/auth/noop.py` | `AuthDependency` | Dev auth: returns defaults, no validation |
| `src/auth/noop_with_token.py` | `AuthDependency` | Test auth: extracts bearer token without validating it |

## Data Flow

### Factory resolution (startup)

```
olsconfig.yaml: authentication_config.module = "k8s" | "noop" | "noop-with-token"
  -> get_auth_dependency(ols_config, virtual_path="/ols-access")
       match module:
         "k8s"             -> k8s.AuthDependency(virtual_path)
         "noop"            -> noop.AuthDependency(virtual_path)
         "noop-with-token" -> noop_with_token.AuthDependency(virtual_path)
```

The result is stored at module level in endpoint files (e.g., `app/endpoints/ols.py`). This means the auth strategy is fixed at import time and cannot change without restarting the process.

### K8s authentication flow (per request)

```
HTTP request with Authorization: Bearer <token>
  -> _extract_bearer_token(header) -> token string
  -> get_user_info(token)
       kubernetes.AuthenticationV1Api.create_token_review(V1TokenReview(token))
       -> if authenticated: return V1TokenReviewStatus (uid, username, groups)
       -> if not authenticated: raise HTTPException(403)
  -> SubjectAccessReview
       kubernetes.AuthorizationV1Api.create_subject_access_review(
         V1SubjectAccessReview(user, groups, non_resource_attributes={
           path: virtual_path,  # "/ols-access" or "/ols-metrics-access"
           verb: "get"
         })
       )
       -> if allowed: return (uid, username, False, token)
       -> if denied: raise HTTPException(403)
```

Special case: if username is `"kube:admin"`, the UID is replaced with the cluster ID to prevent cross-cluster privilege escalation.

### Return tuple semantics

All implementations return `tuple[str, str, bool, str]`:

| Field | K8s (enabled) | K8s (disabled) | noop | noop-with-token |
|---|---|---|---|---|
| UID | From TokenReview | DEFAULT_USER_UID | Query param or DEFAULT | Query param or DEFAULT |
| Username | From TokenReview | DEFAULT_USER_NAME | DEFAULT_USER_NAME | DEFAULT_USER_NAME |
| skip_userid_check | `False` | `False` | `True` | `True` |
| Token | Bearer token | `""` | `""` | Bearer token (unvalidated) |

## Key Abstractions

### K8sClientSingleton

Process-global singleton that initializes and caches Kubernetes API clients. Initialization order:
1. Check for token override in dev config (`k8s_auth_token`)
2. Try in-cluster config (`load_incluster_config()`)
3. Fall back to local kubeconfig (`load_kube_config()`)

Cached instances: `AuthenticationV1Api` (TokenReview), `AuthorizationV1Api` (SubjectAccessReview), `CustomObjectsApi` (cluster version queries). Also caches `cluster_id` and `cluster_version` after first retrieval.

### Virtual path pattern

Authorization is checked against a virtual (non-existent) HTTP path:
- `/ols-access` for all business endpoints
- `/ols-metrics-access` for the metrics endpoint

This maps to Kubernetes RBAC without requiring real URL-based authorization rules.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `app/endpoints/ols.py` | `src/auth/auth.py` | `auth_dependency = get_auth_dependency(...)` at module level |
| `app/metrics/metrics.py` | `src/auth/auth.py` | Same pattern, virtual path `/ols-metrics-access` |
| All endpoint functions | Auth implementation | `auth: Any = Depends(auth_dependency)` in function signature |
| `k8s.AuthDependency` | `K8sClientSingleton` | Singleton access to Kubernetes API clients |

## Implementation Notes

### Auth dependency is resolved at module level

`auth_dependency = get_auth_dependency(config.ols_config, virtual_path=...)` executes when the endpoint module is first imported. The auth module selection is fixed for the process lifetime. Changing the auth module requires a restart.

### Noop modes allow user_id injection

Both `noop` and `noop-with-token` accept a `?user_id=...` query parameter to override the default user ID. This is for testing multi-user scenarios without real authentication. The `skip_userid_check=True` flag signals downstream code to accept the injected user ID.

### Error handling by HTTP status

- `401 Unauthorized` -- missing Authorization header or malformed bearer token (K8s only)
- `403 Forbidden` -- expired/invalid token, or user lacks permission for virtual path (K8s only)
- `400 Bad Request` -- missing Authorization header (noop-with-token only)
- `500 Internal Server Error` -- unexpected Kubernetes API exceptions during TokenReview
