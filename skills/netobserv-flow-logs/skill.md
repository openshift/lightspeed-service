---
name: netobserv-flow-logs
description: Query NetObserv network flow logs in Loki using LogQL and flow field labels (SrcK8S_Namespace, DstK8S_Namespace, K8S_FlowLayer). Use for per-flow investigation, conversation records, packet drops in logs, DNS flows, or when Prometheus netobserv_* metrics are not enough.
---

# NetObserv Flow Logs (Loki)

NetObserv stores **enriched network flow records** in Loki. Each log line represents aggregated or conversation-tracked flow data with Kubernetes, network, and optional DNS/drop fields.

Use this skill when the user needs **individual flows**, conversation history, or label combinations that Prometheus `netobserv_*` metrics cannot express. For bandwidth totals and rates, prefer the `netobserv-metrics` skill first.

## 1. Choose the Right Tooling

| Need | Approach |
|------|----------|
| Filtered flow table in NetObserv UI semantics | NetObserv console plugin or `netobserv_*` MCP tools (kubernetes-mcp-server) when available |
| Raw LogQL, label discovery, LokiStack on OpenShift | `loki_list_instances` → `loki_label_names` / `loki_label_values` → `loki_query_range` (obs-mcp) |

On OpenShift with LokiStack **openshift-network** mode:

- Use tenant **`network`** (`X-Scope-OrgID`) when querying through the gateway.
- Do **not** use `kubernetes_namespace_name` — that label is for container/application logs, not NetObserv flows.

## 2. Discover Labels Before Writing LogQL

1. Confirm `FlowCollector` has Loki enabled (`spec.loki` configured).
2. List Loki labels for a short time window — only use **indexed** fields as stream selectors (see `reference.md`).
3. Start with low-cardinality matchers: `SrcK8S_Namespace`, `DstK8S_Namespace`, `K8S_FlowLayer`, `_RecordType`.
4. Add `SrcK8S_OwnerName` / `DstK8S_OwnerName` or `SrcK8S_Type` / `DstK8S_Type` when narrowing to a workload.

Load `reference.md` for the full Loki-indexed label list and field semantics from the flows format specification.

## 3. Common LogQL Patterns

**Flows involving a namespace (source or destination):**

```logql
{SrcK8S_Namespace="my-app"} or {DstK8S_Namespace="my-app"}
```

**Application-layer flows only:**

```logql
{K8S_FlowLayer="app"}
```

**Regular flow logs (exclude connection tracking control records):**

```logql
{_RecordType="flowLog"}
```

**Combine namespace + layer:**

```logql
{SrcK8S_Namespace="netobserv", K8S_FlowLayer="app"} or {DstK8S_Namespace="netobserv", K8S_FlowLayer="app"}
```

Use a **short time range** and **low limit** first, then widen. Broad queries without label matchers are expensive.

## 4. Fields That Are Not Loki Labels

Many flow fields exist in the JSON payload but are **not** indexed as Loki stream labels (for example `SrcK8S_Name`, `DstK8S_Name`, `Proto`, `SrcPort`, `DstPort`, `Bytes`, `DnsName`).

- Do not put non-indexed field names in the stream selector `{...}` unless `loki_label_names` confirms they exist.
- For console-style filters on those fields, use NetObserv plugin filter syntax or `netobserv_list_flows` when available.
- If log line parsing is required, use LogQL pipeline stages only after confirming the deployment exposes those fields in the line format.

## 5. Map Questions to Fields

| User intent | Indexed labels / fields |
|-------------|-------------------------|
| Traffic to/from a namespace | `SrcK8S_Namespace`, `DstK8S_Namespace` |
| Pod / Service / Node name | `SrcK8S_Name`, `DstK8S_Name` (not indexed — use plugin filters or metrics) |
| Deployment / StatefulSet | `SrcK8S_OwnerName`, `DstK8S_OwnerName` |
| Pod vs Service vs Node | `SrcK8S_Type`, `DstK8S_Type` |
| Ingress / egress at node | `FlowDirection` (`0` Ingress, `1` Egress, `2` Inner) |
| App vs infrastructure flows | `K8S_FlowLayer` (`app` or `infra`) |
| TCP vs UDP | `Proto` (`6` = TCP, `17` = UDP) — verify via label discovery |
| Packet drops in logs | `PktDropPackets`, `PktDropLatestDropCause` (typically in line JSON; PacketDrop feature required) |
| DNS | `DnsName`, `DnsLatencyMs` (DNSTracking feature required) |
| Connection tracking | `_RecordType`: `flowLog`, `newConnection`, `heartbeat`, `endConnection` |

## 6. Empty Results

If LogQL returns no streams:

1. **Wrong labels** — `kubernetes_namespace_name` or container log labels will not match flow logs.
2. **Wrong tenant** — OpenShift network flows use tenant `network`.
3. **RBAC** — Loki gateway may enforce OpenShift authorization; the user must be allowed to view pods in filtered namespaces.
4. **No export** — FlowCollector not writing to this LokiStack, or time range has no data.
5. **Feature disabled** — DNS or drop fields require eBPF features in `FlowCollector`.

Report which case applies using evidence (label list, error message, FlowCollector status).

## 7. Relation to Metrics

- **Metrics** (`netobserv_*` in Prometheus): aggregates, dashboards, alerts.
- **Flow logs** (Loki): drill-down, arbitrary label combinations, conversation records.

Use both: metrics for "how much" and "trend", flow logs for "which flows" and "exact endpoints".

## Quality Standards

- Prefer `SrcK8S_Namespace` / `DstK8S_Namespace` over generic Kubernetes logging labels.
- Always confirm labels with discovery tools before assuming a field is indexed.
- Cite the LogQL selector used and the time range.
- Do not fabricate flow records when queries return zero streams — explain likely causes.
