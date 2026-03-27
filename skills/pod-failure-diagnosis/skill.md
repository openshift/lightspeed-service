---
name: pod-failure-diagnosis
description: Troubleshoot CrashLoopBackOff, ImagePullBackOff, Pending, Error, or OOMKilled status. Use when a workload keeps restarting, fails to start, or is crash-looping.
---

# Pod Failure Diagnosis

When a user reports a pod that is not running, follow this structured triage to identify root cause and provide remediation.

## 1. Classify the Failure Mode

Get the pod status and recent events to determine which failure category applies:

- **CrashLoopBackOff** — container starts but exits repeatedly
- **ImagePullBackOff** — container image cannot be pulled
- **Pending** — pod is not scheduled to any node
- **Error / Init:Error** — init container or main container failed on startup
- **Terminating (stuck)** — pod is not cleaning up

Ask the user for namespace and pod/deployment name if not provided. If they describe symptoms instead ("my app keeps restarting"), map to the correct category before proceeding.

## 2. CrashLoopBackOff Triage

If the pod is crash-looping:

1. Retrieve container logs for the **current** attempt to see the latest error.
2. Retrieve container logs for the **previous** attempt (`--previous`) — the crash reason is often clearer there.
3. Check pod events for OOMKilled signals — this means the container exceeded its memory limit.
4. If OOMKilled: compare the container's memory limit against actual usage and recommend increasing it or fixing the memory leak.
5. If application error: cite the specific log line and recommend the fix (config error, missing dependency, failed health check, etc.).

Do not suggest "just increase resources" unless OOMKilled is confirmed.

## 3. ImagePullBackOff Triage

If the image cannot be pulled:

1. Check the exact image reference in the pod spec — look for typos, wrong tags, or missing registry prefix.
2. Check pod events for the specific pull error message (authentication required, not found, timeout).
3. For authentication errors: verify the imagePullSecrets on the pod and the referenced Secret's content.
4. For "not found" errors: verify the image exists in the registry with the specified tag.
5. For registry connectivity: check if the node can reach the registry (relevant for air-gapped or proxy environments).

Report the exact error message from events — do not guess which sub-case applies.

## 4. Pending Pod Triage

If the pod is stuck in Pending:

1. Check pod events for scheduling failure reasons.
2. **Insufficient resources**: compare the pod's resource requests against available node capacity. Show the arithmetic.
3. **Taints/tolerations**: identify which nodes have taints the pod does not tolerate.
4. **Node selectors / affinity**: verify the pod's constraints match at least one available node.
5. **PVC binding**: if the pod mounts a PersistentVolumeClaim, check if the PVC is Bound. If Pending, diagnose the PVC (missing StorageClass, insufficient capacity, zone mismatch).

For each sub-case, provide the specific fix — do not list all possibilities when the events already tell you which one applies.

## 5. Provide Remediation

Once root cause is identified:

1. State the root cause in one sentence with the supporting evidence (event message, log line, or metric).
2. Provide one or two actionable commands or manifest changes the user can apply.
3. If the fix involves deleting or force-replacing a resource, warn explicitly before suggesting it.
4. If the issue is an application bug (not a platform issue), say so clearly and recommend the user check their application code/config.

## Quality Standards

- Always cite the specific event message or log line that reveals the cause — do not provide generic troubleshooting checklists.
- Distinguish between application errors and platform/infrastructure issues. The remediation path is different.
- If multiple pods are failing, prioritize the earliest failure — cascading failures often share a root cause.
- Do not suggest destructive actions (delete pod, force delete, remove finalizers) without explicit warning about consequences.
- If evidence is insufficient to determine root cause, ask one focused clarification question rather than speculating.
