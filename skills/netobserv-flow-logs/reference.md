# NetObserv flow fields — Loki and filters reference

Source: [Network flows format reference](https://github.com/netobserv/network-observability-operator/blob/main/docs/flows-format.adoc) (`docs/flows-format.adoc`).

Flow records are used for Kafka export, Prometheus `FlowMetric` labels, and **Loki storage**. The **Loki label** column indicates whether a field can be used in LogQL **stream selectors** `{label="value"}`.

The **Filter ID** column is the name used in NetObserv console **Quick Filters** (`FlowCollector` `spec.consolePlugin.quickFilters`).

## Loki-indexed labels (stream selectors)

Use these in `{...}` matchers when confirmed by `loki_label_names`:

| Field | Filter ID | Description |
|-------|-----------|-------------|
| `SrcK8S_Namespace` | `src_namespace` | Source namespace |
| `DstK8S_Namespace` | `dst_namespace` | Destination namespace |
| `SrcK8S_OwnerName` | `src_owner_name` | Source owner (Deployment, StatefulSet, …) |
| `DstK8S_OwnerName` | `dst_owner_name` | Destination owner |
| `SrcK8S_Type` | `src_kind` | Source K8s kind (Pod, Service, Node, …) |
| `DstK8S_Type` | `dst_kind` | Destination K8s kind |
| `SrcK8S_Zone` | `src_zone` | Source availability zone |
| `DstK8S_Zone` | `dst_zone` | Destination availability zone |
| `K8S_ClusterName` | `cluster_name` | Cluster name |
| `K8S_FlowLayer` | `flow_layer` | `app` or `infra` |
| `FlowDirection` | `node_direction` | `0` Ingress, `1` Egress, `2` Inner (node observation point) |
| `_RecordType` | `type` | `flowLog`, `newConnection`, `heartbeat`, `endConnection` |

## Common fields — not Loki labels (use console filters or line JSON)

These appear in the flow format but **Loki label = no**. Do not assume they work in `{...}` without checking `loki_label_names`:

| Field | Filter ID | Notes |
|-------|-----------|-------|
| `SrcK8S_Name` | `src_name` | Pod, Service, or Node name — careful cardinality |
| `DstK8S_Name` | `dst_name` | Same for destination |
| `Proto` | `protocol` | IANA L4 protocol (`6` TCP, `17` UDP) |
| `SrcPort` | `src_port` | |
| `DstPort` | `dst_port` | |
| `SrcAddr` | `src_address` | IP address |
| `DstAddr` | `dst_address` | IP address |
| `Bytes` | n/a | Byte count for the flow |
| `Packets` | n/a | Packet count |
| `DnsName` | `dns_name` | Requires DNSTracking feature |
| `DnsLatencyMs` | `dns_latency` | Requires DNSTracking |
| `PktDropPackets` | n/a | Requires PacketDrop feature |
| `PktDropLatestDropCause` | `pkt_drop_cause` | Requires PacketDrop |
| `TimeFlowRttNs` | `time_flow_rtt` | RTT in nanoseconds — requires FlowRTT |

## LogQL examples

```logql
# Namespace scope (most common)
{SrcK8S_Namespace="netobserv"} or {DstK8S_Namespace="netobserv"}

# Application flows only
{K8S_FlowLayer="app"}

# Exclude connection-tracking control messages
{_RecordType="flowLog"}

# Owner + namespace
{SrcK8S_Namespace="default", SrcK8S_OwnerName="my-deployment"}
```

## OpenShift LokiStack notes

- Flow logs for NetObserv on OpenShift often use Loki tenant **`network`**.
- Gateway paths may be under `/api/logs/v1/<tenant>/...` when using openshift-network mode.
- Empty streams with no error often indicate **RBAC** (user cannot view pods in the namespace used in the query) or **wrong tenant/labels**.

## Labels to avoid for NetObserv flows

| Label | Why |
|-------|-----|
| `kubernetes_namespace_name` | Container/application log schema, not flow logs |
| `namespace` (generic) | Unless confirmed on the Loki stack, prefer `SrcK8S_Namespace` / `DstK8S_Namespace` |

## Cardinality (FlowMetrics / labels)

From the flows format **Cardinality** column:

- **fine** — safer as metric or filter labels
- **careful** — narrow with filters before using as labels
- **avoid** — high cardinality (IPs, byte counts, timestamps)

Mixing `SrcK8S_Name` and `DstK8S_Name` in the same metric label set can multiply cardinality severely.

## Further reading

- [Metrics.md](https://github.com/netobserv/network-observability-operator/blob/main/docs/Metrics.md) — Prometheus `netobserv_*` metrics from the same flow data
- [HealthRules.md](https://github.com/netobserv/network-observability-operator/blob/main/docs/HealthRules.md) — alerts built on NetObserv metrics
- Full field table: [flows-format.adoc](https://github.com/netobserv/network-observability-operator/blob/main/docs/flows-format.adoc)
