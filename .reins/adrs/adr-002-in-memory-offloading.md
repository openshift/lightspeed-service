# ADR-002: In-Memory Storage for Offloaded Content

Date: 2026-04-07
Story: OLS-2277
Amends: ADR-001 (Option 4 storage mechanism only)
Status: rejected — disk storage retained; the two-tool retrieval design (ADR-001 updated 2026-04-07) benefits from file-based random access where content stays on disk between search and read operations

## Context

ADR-001 selected disk-based storage for offloaded tool outputs:
write the full content to a temporary file on disk, provide a
grep-style retrieval tool, delete files when the request completes.

During implementation planning, we evaluated whether disk storage
is necessary given the actual deployment model:

- Offloaded content is **request-scoped** — it lives for 10-60
  seconds (the duration of one request/response cycle with LLM
  round-trips), then is discarded.
- OLS is a **single-tenant service per cluster**. Concurrent users
  are typically in the single digits, not tens or hundreds.
- `max_workers` defaults to 1 (one uvicorn worker process). The
  sync endpoints run in a threadpool (default ~8-12 threads), but
  practical concurrency is bounded by the number of humans asking
  questions simultaneously.
- The OLS process already holds **hundreds of MB** in resident
  memory (embedding models, RAG indexes, in-memory conversation
  cache). Transient offloaded content in the tens-of-MB range is
  a small fraction of baseline.

Disk storage introduces complexity that exists solely to manage a
resource (filesystem) that the deployment doesn't actually need:

- `offload_storage_path` configuration and validation
- Directory creation on first offload
- File write I/O and error handling
- Fallback-to-truncation-on-disk-failure code path
- File cleanup in try/finally (must handle partial cleanup,
  already-deleted files, pod restarts)
- Path traversal security surface (mitigated by allowlist, but
  the attack surface exists only because files exist)
- Read-only filesystem concerns in hardened containers

## Decision

Store offloaded content in memory (`dict[str, str]` on the
`OffloadManager`) instead of on disk. The retrieval tool reads
from this dict. Content is released when the `OffloadManager`
is dereferenced at the end of the request (Python GC).

All other ADR-001 decisions remain unchanged: the offloading
threshold, placeholder format, retrieval tool design, conditional
registration, budget enforcement, and security allowlist all apply
exactly as specified.

### Safety caps to prevent OOM

Two caps bound memory usage, both enforced in
`OffloadManager.try_offload`:

1. **Per-output cap (50 MB):** Any single tool output exceeding
   50 MB is not offloaded — falls back to truncation. Unchanged
   from ADR-001.

2. **Per-request total cap (100 MB):** The `OffloadManager`
   maintains a running sum of bytes stored. When cumulative
   offloaded content exceeds 100 MB, subsequent offloads fall
   back to truncation with a logged warning. This is a single
   integer counter — no shared state, no locking, fully
   request-scoped.

Fallback is always truncation. If either cap is hit, the system
behaves identically to today's behavior for the affected output.
Earlier offloaded content remains searchable.

### Memory impact analysis

| Scenario | Concurrent requests | Offloaded per request | Avg size | Total held |
|---|---|---|---|---|
| Typical | 3 | 1 output | 500 KB | 1.5 MB |
| Busy | 5 | 2 outputs | 2 MB | 20 MB |
| Stress | 10 | 3 outputs | 5 MB | 150 MB |
| Theoretical max | 10 | all hitting cap | 100 MB | 1 GB |

The theoretical maximum (10 concurrent requests × 100 MB cap)
is 1 GB. This requires 10 simultaneous users each triggering
multiple MCP tools returning very large outputs — not a realistic
scenario for a single-tenant cluster assistant. The per-request
cap ensures no single request can exceed 100 MB regardless of
how many tools fire.

## Alternatives Considered

### Disk storage (ADR-001 Option 4, amended)

Write to temporary files, clean up in try/finally.

- **Pros:** Memory-neutral after write; works for arbitrarily
  large content.
- **Cons:** Requires config (storage path), filesystem
  availability, file cleanup, I/O error handling, path security
  considerations, read-only filesystem compatibility. All this
  complexity serves a scenario (sustained high memory pressure
  from many concurrent large offloads) that doesn't match the
  actual deployment model.

### Global process-wide memory cap

Track total offloaded bytes across all concurrent requests using
shared state. Reject new offloads when the global total exceeds
a limit.

- **Pros:** Absolute guarantee on total memory from offloading.
- **Cons:** Requires shared mutable state with synchronization
  (lock or atomic) across request handlers. Adds coupling between
  independent requests. Unnecessary given low concurrency and the
  per-request cap already bounding individual requests.

## Consequences

### Positive

- **Eliminates** `offload_storage_path` config, file I/O, file
  cleanup, disk-failure fallback path, and filesystem security
  surface.
- **Simpler implementation:** ~30-40% less code in
  `offloaded_content.py` compared to disk-based approach.
- **Works on read-only filesystems** without fallback or special
  handling.
- **Faster:** No disk I/O latency for offload or retrieval.
- **No cleanup needed:** Python GC reclaims memory when the
  manager is dereferenced. No try/finally for file deletion
  (try/finally for the manager lifecycle itself is still good
  practice but failure to clean up doesn't leave orphaned files).

### Negative

- **Memory pressure under extreme concurrency:** If many
  concurrent requests each offload large content, total process
  memory increases. Mitigated by per-request cap (100 MB) and
  the reality of single-digit concurrent users.
- **No swap-to-disk for very large content:** With disk storage,
  the OS could page out file content. In-memory content stays
  resident. Mitigated by the 50 MB per-output and 100 MB
  per-request caps.

### Changes to ADR-001

Only the **storage mechanism** changes. This table maps ADR-001
decisions to their status:

| ADR-001 Decision | Status |
|---|---|
| 1. Offloading replaces truncation path | Unchanged |
| 2. Request-scoped lifetime | Unchanged (GC instead of file delete) |
| 3. Single search tool with windowed output | Unchanged |
| 4. max_tokens_per_tool_output as threshold | Unchanged |
| 5. In-memory allowlist for ref_id | Unchanged (dict value is str instead of Path) |
| 6. Register retrieval tool only when offloaded | Unchanged |
| 7. Retrieval results subject to budget | Unchanged |
| 8. Fallback to truncation | Simplified (only size caps, no disk failure) |
| 9. Single module | Unchanged |
| 10. Attachment offloading separate story | Unchanged |

### Changes to story AC

| AC | Impact |
|---|---|
| AC 1 (write to temp file) | Reworded: "stored in memory" instead of "written to disk" |
| AC 6 (files deleted on completion) | Simplified: content released by GC when manager is dereferenced |
| AC 7 (truncation fallback for disk) | Simplified: fallback only for size cap violations, not disk errors |
| AC 9 (storage path configurable) | Removed: no storage path needed. `max_tokens_per_tool_output` on ModelParameters remains. |
