# Token Quota Management

The quota subsystem enforces token consumption limits on LLM requests, preventing uncontrolled cost by gating access before each LLM call and tracking usage afterward.

## Behavioral Rules

1. The service must support two limiter types: **per-user** (keyed by authenticated user ID) and **cluster-wide** (a single global counter shared by all users). Either type may be configured alone or both may be active simultaneously.

2. When both limiter types are active, a request must pass **both** checks to proceed. Exhaustion of either limiter is sufficient to reject the request.

3. **Pre-flight check**: Before every LLM invocation, the service must verify that all configured limiters report a positive available balance for the subject. If any limiter reports zero or negative tokens, the request must be rejected immediately and no LLM resources may be consumed.

4. **Post-response consumption**: After the LLM produces a response, the service must debit the sum of input tokens and output tokens from every configured limiter. The available balance may go negative due to concurrent requests; the pre-flight check prevents this under normal load.

5. **Auto-initialization**: When a subject (user or cluster) is queried for the first time and no quota record exists, the system must automatically create a record with the available balance set to the configured initial quota value. No manual provisioning is required.

6. **Revoke (reset)**: It must be possible to reset a subject's available balance to the configured initial quota value. This is the mechanism used by the scheduler to restore quotas on a recurring basis.

7. **Increase (top-up)**: It must be possible to add a configured increment of tokens to a subject's current available balance without resetting it. This is the mechanism used by the scheduler for incremental replenishment.

8. **Scheduler**: A background process must run at the configured interval and, for each configured limiter, apply the increase and/or reset operations to all matching subjects. The scheduler must guard against double-application: if a subject's quota was already updated within the current period, the scheduler must skip that subject until the next period elapses.

9. **Quota visibility in API responses**: After a successful query, the response must include the remaining available quota for each configured limiter (keyed by limiter class name), so that clients can display remaining balance and warn when quota is low.

10. **Quota exceeded error**: When quota is exhausted, the error must identify: (a) whether the limit is per-user or cluster-wide, (b) the subject identifier for per-user limits, and (c) the current available balance. The error must carry the subject ID and balance as structured attributes, not only in the message string.

11. **Token usage history**: When enabled, the service must record cumulative input and output token counts per user, per provider, and per model. This history is stored independently from quota enforcement state and supports analytics. The record is upserted on each consumption -- if a record exists for the (user, provider, model) tuple, the counts are incremented; otherwise a new record is created.

12. **PostgreSQL database errors** during quota checks must result in an HTTP 500 response with a cause describing the database communication failure. The request must not proceed to LLM invocation.

13. When no quota limiters are configured, the service must skip all quota checks and consumption steps, allowing unrestricted access.

## Configuration Surface

- `ols_config.quota_handlers.storage` -- PostgreSQL connection details (host, port, user, password, dbname, SSL mode, GSS encryption mode) for quota state persistence.
- `ols_config.quota_handlers.scheduler.period` -- Interval in seconds at which the background scheduler runs.
- `ols_config.quota_handlers.limiters` -- List of limiter definitions, each containing:
  - `name` -- Identifier for the limiter.
  - `type` -- Either `user_limiter` (per-user) or `cluster_limiter` (cluster-wide).
  - `initial_quota` -- Starting token balance assigned on first access and used as the reset target.
  - `quota_increase` -- Number of tokens added per scheduler cycle (incremental top-up).
  - `period` -- PostgreSQL interval expression controlling the double-application guard (e.g., `"1 day"`).
- `ols_config.quota_handlers.enable_token_history` -- Boolean toggling token usage history recording.

## Constraints

1. All quota state must be stored in PostgreSQL. In-memory-only quota storage is not supported because state must survive restarts and remain consistent across multiple service replicas.

2. The quota check must occur **before** the LLM call; token consumption must occur **after** the LLM response is received. This ordering is invariant.

3. The scheduler runs as a daemon thread. It must not block service startup or request processing.

4. The cluster-wide limiter uses an empty string as the subject identifier internally. All requests share the same row regardless of user identity.

5. Token usage history storage is independent from quota enforcement storage. Enabling or disabling history does not affect quota limit enforcement.

6. Each limiter definition must include a `name`. The configuration must reject limiter entries that omit the name.

7. Quota handler configuration must require both `storage` and `scheduler` sections to be present. Omission of either is a configuration error.

## Planned Changes

- [PLANNED: OLS-1470] When cluster awareness is enabled by default, a default quota must be automatically set so that the system operates with quota enforcement out of the box.
- [PLANNED: OLS-2823] Per-user LLM provider API keys, enabling individual token quota management tied to each user's own provider credentials rather than a shared service credential.
