---
name: degraded-operator-recovery
description: Troubleshoot ClusterOperator in Degraded, Unavailable, or not Progressing state. Use when operator status shows error conditions, reconciliation failures, or degraded health checks.
---

# Degraded Operator Recovery

When a user reports unhealthy cluster operators or a stuck upgrade, follow this structured approach to identify the blocking condition and provide recovery steps.

## 1. Assess Cluster Operator Health

Start by listing all cluster operators and their status conditions:

- Identify operators with `Degraded=True` or `Available=False`.
- Note operators with `Progressing=True` — these may be mid-reconciliation and need time, not intervention.
- If multiple operators are degraded, identify dependencies. For example, if `kube-apiserver` is degraded, other operators that depend on the API server will also report issues.

Focus on the **root cause operator** — the one whose degradation is not explained by another operator's failure.

## 2. Read the Blocking Condition

For each degraded operator:

1. Read the operator's status conditions — the `message` field on the `Degraded` condition usually contains the specific error.
2. Check the `lastTransitionTime` to understand how long the operator has been in this state.
3. Look for common patterns in the condition message:
   - Certificate expiry or rotation failures
   - Webhook configuration errors
   - Failed rollout of an operand deployment
   - Resource contention (CPU/memory on control plane nodes)
   - Quorum loss (etcd-specific)

Do not skip this step. The condition message is the single most informative piece of data.

## 3. Inspect Managed Operands

If the condition message is not sufficient:

1. Identify the operator's managed deployments, daemonsets, or statefulsets (usually in `openshift-*` namespaces).
2. Check if any operand pods are in CrashLoopBackOff, Pending, or Error state.
3. If operand pods are failing, triage them using the same approach as pod failure diagnosis — check logs and events.
4. For control plane operators (etcd, kube-apiserver, kube-controller-manager, kube-scheduler): check static pod status on control plane nodes.

## 4. Check for Upgrade-Related Issues

If this is happening during or after a cluster upgrade:

1. Check `oc get clusterversion` for the upgrade status and any reported failures.
2. Determine if the operator is stuck waiting for a node reboot (MachineConfigPool not updated).
3. Check if pending CSRs need approval — node certificate renewal during upgrade can block operators.
4. For transient `Progressing=True` during upgrade: advise waiting (up to the operator's expected rollout window) before intervening.

Distinguish between "upgrade in progress" (normal) and "upgrade stuck" (needs intervention).

## 5. Provide Recovery Steps

Once the blocking condition is identified:

1. State which operator is degraded and what the blocking condition is.
2. Provide specific recovery actions:
   - **Pending CSRs**: approve them with `oc adm certificate approve`.
   - **Failed operand pod**: follow pod triage to fix the underlying issue.
   - **Certificate issues**: check if cert rotation can be triggered or if manual renewal is needed.
   - **Resource contention**: identify the pressure source on control plane nodes.
   - **Webhook errors**: check if the webhook service is available and the CA bundle is correct.
3. If the issue is internal to the operator and cannot be resolved by the user, recommend opening a support case with the specific condition message.

## Quality Standards

- Always start with `oc get clusteroperators` before diving deeper — this prevents chasing symptoms of a different root cause.
- For etcd, kube-apiserver, and kube-controller-manager: warn about control plane impact before suggesting any restart or remediation.
- Never suggest force-deleting operator-managed resources without explicit warning — operators reconcile state and force-deletion can cause split-brain.
- If the operator is `Progressing=True`, advise waiting before intervening — premature action can make the situation worse.
- Distinguish upgrade-related transient degradation from persistent failures that need action.
