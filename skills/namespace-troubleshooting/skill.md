---
name: namespace-troubleshooting
description: Troubleshoot namespace stuck in Terminating state, ResourceQuota exhaustion, or RBAC permission denied errors. Use when resources cannot be created or forbidden errors occur.
---

# Namespace Troubleshooting

When a user reports issues at the namespace/project level — stuck deletion, inability to create resources, or permission errors — follow this structured approach to identify the specific blocker and provide a targeted fix.

## 1. Check Namespace Status

Start by confirming the namespace exists and its phase:

1. If the namespace is **Active**: the issue is likely quota, LimitRange, or RBAC. Proceed to the relevant section based on the user's symptom.
2. If the namespace is **Terminating**: proceed to the stuck termination triage.
3. If the namespace does not exist: confirm with the user whether it was recently deleted or never created.

## 2. Stuck Terminating Namespace

If the namespace is stuck in Terminating state:

1. Check the namespace's `metadata.finalizers` — these are the hooks preventing deletion.
2. List all resources still remaining in the namespace. Kubernetes cannot remove the namespace until all resources with finalizers are cleaned up.
3. Identify which specific resources are blocking:
   - **Custom Resources with finalizers**: the operator managing them may be absent or broken. Check if the CRD's controller is running.
   - **PersistentVolumeClaims**: the underlying PV may be stuck in Released state or the storage provisioner is not cleaning up.
   - **Pods with long graceful termination**: check if pods have unusually long `terminationGracePeriodSeconds` or are stuck in pre-stop hooks.
4. For each blocking resource, explain what the finalizer protects and the consequences of removing it.
5. If a finalizer must be removed manually, provide the specific command but warn that this skips cleanup — the external resource (PV, cloud resource, etc.) may be orphaned.

Never suggest blanket removal of namespace finalizers ("just patch the namespace to remove all finalizers"). This skips all cleanup and leaks resources. Always identify and address the specific blocking resources first.

## 3. Quota Exhaustion

If deployments or pods fail to create with quota-related errors:

1. List the ResourceQuota objects in the namespace and their current usage vs. limits.
2. Show the arithmetic: what the new pod/deployment requests + what is already consumed vs. the quota limit.
3. Identify which specific resource is exhausted (cpu, memory, pods count, services, configmaps, etc.).
4. Recommend the appropriate fix:
   - If the quota is legitimately too low: increase it (and note who has permission to do so).
   - If existing resources are consuming more than expected: identify the top consumers.
   - If completed Jobs or failed pods are consuming quota: suggest cleaning them up.

Always show the numbers. "Quota exceeded" without showing the actual usage vs. limit is not actionable.

## 4. LimitRange Violations

If pod creation fails with LimitRange-related errors:

1. List the LimitRange objects in the namespace.
2. Compare the pod's resource requests and limits against the LimitRange constraints:
   - **Min/Max violations**: the pod requests less than the minimum or more than the maximum allowed.
   - **Default applied unexpectedly**: if the pod has no resource requests, the LimitRange default is applied — this may conflict with the application's actual needs.
   - **MaxLimitRequestRatio**: the ratio between the pod's limit and request exceeds the allowed ratio.
3. Show exactly which constraint is violated and by how much.
4. Recommend adjusting either the pod's resources or the LimitRange, depending on which is the source of truth.

## 5. RBAC / Permission Issues

If the user gets `Forbidden` errors when operating in the namespace:

1. Ask for or identify the exact error message — it contains the verb, resource, and API group being denied.
2. Check the user's RoleBindings in the namespace and their ClusterRoleBindings.
3. Identify whether the user has a Role that includes the required verb + resource combination.
4. If the permission is missing: recommend creating a specific RoleBinding. Specify the exact Role or ClusterRole to bind — do not default to "give them admin."
5. If the user should not have the permission: confirm this is expected and explain who does have it.

Always specify the exact verb+resource the user is missing (e.g., "create deployments.apps") rather than vaguely saying "insufficient permissions."

## 6. Project-Specific Considerations

OpenShift Projects are namespaces with additional metadata. If the issue involves Projects specifically:

1. Check if a ProjectRequest template is in use — it may inject default quotas, LimitRanges, or NetworkPolicies that cause unexpected restrictions.
2. Check if the user's ability to create Projects is controlled by a `self-provisioner` ClusterRoleBinding — removal of this binding prevents users from creating their own Projects.
3. For multi-tenant clusters: check if NetworkPolicies in the namespace are blocking expected traffic between namespaces.

## Quality Standards

- For stuck Terminating: always list the specific resources and finalizers blocking deletion. Never suggest removing namespace finalizers without first identifying what they protect.
- For quota: show the arithmetic (requested + existing usage vs. limit). An error message without numbers is not actionable.
- For RBAC: specify the exact verb+resource the user is missing, not just "add admin role." Least-privilege matters.
- For LimitRange: show which constraint is violated and by how much, not just that a violation exists.
- If the issue spans multiple categories (e.g., quota is full AND RBAC prevents the user from adjusting it), address both and clarify the dependency.
