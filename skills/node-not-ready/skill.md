---
name: node-not-ready
description: Troubleshoot NotReady or SchedulingDisabled node status. Use when a node is down, unschedulable, or needs to be drained and restored.
---

# Node Not Ready Diagnosis

When a user reports a node that is NotReady, flapping, or has evicted workloads unexpectedly, follow this structured approach to identify the cause and restore the node.

## 1. Identify Affected Nodes

List all nodes and their statuses:

1. Note which nodes are NotReady and how long they have been in that state. Recent (minutes) vs. prolonged (hours/days) requires different urgency.
2. Determine if the affected node is a **control plane node** or a **worker**. Control plane nodes are higher priority — losing quorum affects the entire cluster.
3. Check if the node is `SchedulingDisabled` (cordoned) — this may be intentional (maintenance) or a side effect of a MachineHealthCheck remediation.

## 2. Read Node Conditions

Describe the affected node and examine its conditions:

1. **MemoryPressure=True**: the node is running low on memory. Identify the top memory-consuming pods and recommend eviction or resource limit adjustments.
2. **DiskPressure=True**: the node is running low on disk. Check for large container logs, unused images, or full persistent volumes. Recommend cleanup actions.
3. **PIDPressure=True**: the node is exhausting process IDs. Identify pods with excessive process creation (fork bombs, misconfigured worker pools).
4. **NetworkUnavailable=True**: the node's network plugin is not ready. Check SDN/OVN-Kubernetes pod health on that specific node.
5. **Ready=False with no other pressure conditions**: kubelet is not posting status. This usually means the kubelet process is down or the node is unreachable.

The specific condition tells you exactly where to look next — do not run through all possibilities if the condition is clear.

## 3. Check Infrastructure Layer

If the node conditions suggest the node itself is unreachable or the kubelet is down:

1. Check the Machine and MachineSet objects — is the machine reported as running by the infrastructure provider?
2. Check if a MachineHealthCheck has already detected the issue and is remediating (creating a replacement machine). If so, advise waiting for the remediation to complete before manual intervention.
3. For bare-metal or user-provisioned infrastructure: the user needs to check the host directly (SSH, console, BMC). OLS cannot diagnose host-level issues without cluster-side evidence.

## 4. Check Certificate Issues

Node certificate problems are a common cause of NotReady, especially after extended downtime:

1. Check for pending Certificate Signing Requests (CSRs) — nodes that were offline during certificate rotation may need their CSRs manually approved.
2. If CSRs are pending: approve them and verify the node returns to Ready state.
3. If no CSRs are pending but the node was offline for a long time: the kubelet certificates may have expired entirely and the node may need to be re-joined.

## 5. Check Network Plugin Health

If the condition is NetworkUnavailable:

1. Identify which network plugin the cluster uses (OpenShift SDN or OVN-Kubernetes).
2. Check the network plugin pods running on the affected node — are they in CrashLoopBackOff or not scheduled?
3. Check the node's network plugin logs for specific errors (failed to configure interface, VXLAN/Geneve tunnel issues, OVS database corruption).

## 6. Provide Recovery Steps

Once the cause is identified:

1. State the specific condition and evidence.
2. Provide the targeted fix:
   - **Resource pressure**: identify consumers, suggest eviction or resource adjustments.
   - **Kubelet down**: check machine status, recommend restart or machine replacement.
   - **Pending CSRs**: approve them.
   - **Network plugin failure**: restart the network plugin pod on the affected node or investigate the specific error.
3. After remediation, verify the node transitions back to Ready.

## Quality Standards

- Always report how long the node has been NotReady — this affects triage urgency and likely causes.
- Check whether a MachineHealthCheck is already remediating before suggesting manual action — do not compete with automated remediation.
- Warn before suggesting cordon/drain if the cluster is already capacity-constrained — draining a node when others are also unhealthy can cause cascading evictions.
- Distinguish control plane nodes from workers in all recommendations. Restarting a control plane node has different risk than restarting a worker.
- If the node is flapping (oscillating between Ready and NotReady), focus on the transition events rather than the current state.
