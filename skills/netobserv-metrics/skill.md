---
name: netobserv-metrics
description: Query and interpret NetObserv network flow metrics in Prometheus (netobserv_*). Use for namespace traffic, packet drops, RTT, DNS latency, DNS NXDOMAIN errors, network policy events, or missing NetObserv metrics on OpenShift. Do not conclude no DNS failures from absent alerts alone.
---

# NetObserv Metrics

NetObserv exports **Prometheus metrics** derived from network flow records. They are scraped by cluster monitoring (not shipped by the NetObserv operator itself). Metric names in Prometheus are prefixed with `netobserv_`.

Use this skill when the user asks about network bandwidth, flow counts, packet drops, latency, DNS performance, DNS NXDOMAIN, or whether NetObserv metrics exist.

## Critical rules

1. **Do not stop after alerts.** For DNS, drops, RTT, or policy symptoms, absent firing alerts does **not** mean no problem — run domain metrics and flow logs (§7).
2. **Do not use `{namespace="..."}`** on `netobserv_*` series. Scope with `SrcK8S_Namespace` and/or `DstK8S_Namespace`.
3. **DNS NXDOMAIN:** query `netobserv_namespace_dns_latency_seconds_count` with `DnsFlagsResponseCode="NXDomain"` (exact casing) before reporting no DNS errors.
4. **DNS RCODE is return traffic:** NXDOMAIN and other response codes label the **client namespace on `DstK8S_Namespace`**, not `SrcK8S_Namespace`. Always query `DstK8S_Namespace="NS"` first for DNS failures; use `SrcK8S_Namespace` only as a fallback.
5. **Empty Prometheus → query Loki:** If DNS metrics return no data after both Src/Dst filters, run flow-log queries (`netobserv-flow-logs` skill) before concluding there is no traffic.

## 1. Discover What Exists

Before writing PromQL:

1. Confirm **NetObserv is installed** — look for a `FlowCollector` in the NetObserv namespace (default `netobserv`).
2. List metrics with a `netobserv` filter (for example `list_metrics` with regex `netobserv_.*` when observability MCP tools are available).
3. If no `netobserv_*` series appear, metrics may be disabled in `FlowCollector` `spec.processor.metrics.includeList` or not yet scraped — check configuration before inventing metric names.

Do not guess metric names. Names in configuration omit the `netobserv_` prefix; in Prometheus they appear as `netobserv_<name>` (for example `netobserv_namespace_ingress_packets_total`).

## 2. Map the Question to a Metric Family

| User intent | Typical metric (Prometheus name) | Notes |
|-------------|----------------------------------|-------|
| Bytes in/out per namespace | `netobserv_namespace_ingress_bytes_total`, `netobserv_namespace_egress_bytes_total` | Use `rate()` over a range |
| Packets in/out per namespace | `netobserv_namespace_ingress_packets_total`, `netobserv_namespace_egress_packets_total` | Ingress packets enabled by default |
| Flow count per namespace | `netobserv_namespace_flows_total` | Enabled by default |
| Per-workload traffic | `netobserv_workload_*` | Higher cardinality — prefer aggregation |
| Node-level traffic | `netobserv_node_*` | Node ingress/egress bytes and packets |
| Packet drops | `netobserv_namespace_drop_packets_total`, `netobserv_workload_drop_packets_total` | Requires `PacketDrop` eBPF feature |
| TCP RTT / latency | `netobserv_namespace_rtt_seconds`, `netobserv_workload_rtt_seconds` | Requires `FlowRTT` feature |
| DNS latency | `netobserv_namespace_dns_latency_seconds` | Requires `DNSTracking` feature |
| Network policy denials | `netobserv_namespace_network_policy_events_total` | Requires `NetworkEvents` feature |
| IPSec flows | `netobserv_node_ipsec_flows_total` | Requires `IPSec` feature |

Load `reference.md` in this skill directory for the full predefined metric list and feature gates.

## 3. Write PromQL

### Label model (critical)

`netobserv_namespace_*` and `netobserv_workload_*` series use **flow field labels**, not a generic `namespace` label:

| Label | Use |
|-------|-----|
| `SrcK8S_Namespace` | Source Kubernetes namespace |
| `DstK8S_Namespace` | Destination Kubernetes namespace |
| `K8S_FlowLayer` | `infra` (DNS, node traffic) or `app` (workload traffic) |
| `DnsFlagsResponseCode` | DNS RCODE on DNS latency metrics — `NoError`, **`NXDomain`**, etc. On **response** flows the client namespace is usually **`DstK8S_Namespace`**. |

**Do not** query `{namespace="my-app"}` on NetObserv flow metrics — it returns empty and looks like “no data”. Scope with `SrcK8S_Namespace` and/or `DstK8S_Namespace`. Health alert templates may show `{{ $labels.namespace }}` after `label_replace`; raw metrics still use `SrcK8S_Namespace`.

**Counters** (names ending in `_total` or histogram `_count`): use `rate()` or `increase()` over a range window (for example `[2m]` or `[5m]`).

Examples:

```promql
# Namespace ingress bandwidth (bytes/s) — source side
sum(rate(netobserv_namespace_ingress_bytes_total{SrcK8S_Namespace="my-app"}[5m]))

# Top namespaces by egress packets
topk(10, sum(rate(netobserv_namespace_egress_packets_total{SrcK8S_Namespace!=""}[5m])) by (SrcK8S_Namespace))

# Packet drops involving a namespace (either direction)
sum(rate(netobserv_namespace_drop_packets_total{SrcK8S_Namespace="my-app"}[5m]))
  or sum(rate(netobserv_namespace_drop_packets_total{DstK8S_Namespace="my-app"}[5m]))
```

**Histograms** (RTT, DNS latency): use `histogram_quantile` on `_bucket` series, or `rate()` on `_count` / `_sum`.

Prefer **namespace-level** metrics unless the user needs pod/workload detail. `workload_*` metrics multiply cardinality and can stress Prometheus when many are enabled.

## 4. When Metrics Are Missing

If the user expects a metric but it is absent:

1. Check `FlowCollector` `spec.processor.metrics.includeList` — only listed metrics are generated (names without `netobserv_` prefix).
2. Check required **eBPF features** in `spec.agent.ebpf.features` (for example `PacketDrop`, `FlowRTT`, `DNSTracking`, `NetworkEvents`, `IPSec`).
3. Note default metrics: several `namespace_*` and `workload_*` counters are on by default; workload variants may replace namespace defaults when Loki is disabled (see reference).

Report the specific gap (metric name + feature or includeList entry) rather than a generic “NetObserv broken” conclusion.

## 5. Custom Metrics (FlowMetric)

Users can define counters or histograms via the `FlowMetric` CR (`flows.netobserv.io/v1alpha1`) in the FlowCollector namespace. Warn about **label cardinality** — avoid combining high-cardinality Src and Dst labels without filters.

For FlowMetric examples and console chart PromQL, load `reference.md`.

## 6. Related Data (Not Prometheus)

- **Per-flow records and LogQL** use Loki flow logs (different labels such as `SrcK8S_Namespace`) — not the same as `netobserv_*` metrics. Use the `netobserv-flow-logs` skill, NetObserv console, or Loki MCP tools for flow-level investigation.
- **Network health alerts** (packet drops by device, DNS errors, external traffic trends) are built from these metrics — see OpenShift Network Health or `PrometheusRule` resources from NetObserv.

## 7. Investigation Workflow (Required for Troubleshooting)

For every NetObserv investigation, run **all applicable layers** before concluding. Do not stop after alerts or a single generic metric.

1. **Alerts** — list firing or recent NetObserv health rules for the target namespace (see table below).
2. **Domain metrics** — query the symptom-specific `netobserv_*` series scoped with **`SrcK8S_Namespace` / `DstK8S_Namespace`** (not a generic `namespace` label).
3. **Flow logs** — cross-check with Loki or NetObserv flow tools (`netobserv-flow-logs` skill) for names, error codes, drop causes, or RTT fields metrics cannot show.

Always filter Prometheus queries to the **namespace the user named**. Health recording rules use names like `netobserv:health:*:namespace:*` — still scope by the user's namespace label when present.

### Health rules (Prometheus alerts)

| Symptom | Alert / rule name | Primary metrics |
|---------|-------------------|-----------------|
| DNS resolution failures | `DNSErrors`, `DNSNxDomain` | `netobserv_namespace_dns_latency_seconds_*` (if latency also suspected) |
| Slow DNS | (trend from DNS latency histogram) | `netobserv_namespace_dns_latency_seconds_bucket`, `_count`, `_sum` |
| Kernel packet drops | `PacketDropsByKernel` | `netobserv_namespace_drop_packets_total`, `netobserv_node_drop_packets_total` |
| NetworkPolicy denials | `NetpolDenied` | `netobserv_namespace_network_policy_events_total` (not `drop_packets_total` alone) |
| High TCP RTT | `LatencyHighTrend` | `netobserv_namespace_rtt_seconds` — use `histogram_quantile` on `_bucket` |
| Ingress / HTTPS errors | `Ingress5xxErrors` | `netobserv:health:ingress_5xx_errors:namespace:*` |
| IPsec issues | `IPsecErrors` | `netobserv_node_ipsec_flows_total` |

**Policy vs kernel drops:** `netobserv_namespace_drop_packets_total` counts **kernel/OVS packet drops** (PacketDrop feature). On OpenShift, NetworkPolicy denials often appear as **`PktDropLatestDropCause="OVS_DROP_EXPLICIT"`** in flow logs — that is policy-related even though the field is under PacketDrop.

For policy investigation, always query **both**:

1. `netobserv_namespace_network_policy_events_total{action="drop",…}` (NetworkEvents feature)
2. Flow logs with `packetLoss=dropped` or `PktDropLatestDropCause~OVS_DROP`

**Zero `drop_packets_total` with a wrong or missing namespace label does not rule out NetworkPolicy blocks.** Scope with `SrcK8S_Namespace` / `DstK8S_Namespace` and try `K8S_FlowLayer="app"` (pod-to-pod) and `"infra"`.

### Symptom → required PromQL patterns

Use `NS` as the target namespace from the user query. DNS traffic is usually `K8S_FlowLayer="infra"`.

**DNS NXDOMAIN / DNS errors** (namespace `NS`) — RCODE is on **DNS response (return) traffic**; filter **`DstK8S_Namespace="NS"` first**. Use `_count` with `DnsFlagsResponseCode`; value is **`NXDomain`** (exact casing):

```promql
# Primary — client namespace on return path (matches NetObserv DNSNxDomain alert)
topk(10,
  sum(rate(netobserv_namespace_dns_latency_seconds_count{
    DnsFlagsResponseCode="NXDomain",
    K8S_FlowLayer="infra",
    DstK8S_Namespace="NS"
  }[2m])) by (DnsFlagsResponseCode, SrcK8S_Namespace, DstK8S_Namespace)
)

# Fallback — try source side or drop K8S_FlowLayer if still empty
sum(rate(netobserv_namespace_dns_latency_seconds_count{
  DnsFlagsResponseCode="NXDomain",
  SrcK8S_Namespace="NS"
}[2m])) by (DnsFlagsResponseCode, SrcK8S_Namespace, DstK8S_Namespace)
```

Discover which label holds the namespace when unsure:

```promql
topk(10, sum(rate(netobserv_namespace_dns_latency_seconds_count{DnsFlagsResponseCode="NXDomain"}[5m])) by (SrcK8S_Namespace, DstK8S_Namespace))
```

All non-success DNS codes (return traffic — prefer `DstK8S_Namespace`):

```promql
sum(rate(netobserv_namespace_dns_latency_seconds_count{DnsFlagsResponseCode!="NoError",K8S_FlowLayer="infra",DstK8S_Namespace="NS"}[2m])) by (DnsFlagsResponseCode, SrcK8S_Namespace, DstK8S_Namespace)
```

**No firing `DNSNxDomain` alert does not mean no NXDOMAIN** — the alert fires on a percentage threshold; direct metric queries can still show NXDOMAIN rate.

**DNS latency** (namespace `NS`):

```promql
histogram_quantile(0.99, sum(rate(netobserv_namespace_dns_latency_seconds_bucket{SrcK8S_Namespace="NS",K8S_FlowLayer="infra"}[2m])) by (le))
sum(rate(netobserv_namespace_dns_latency_seconds_count{SrcK8S_Namespace="NS",K8S_FlowLayer="infra"}[2m]))
```

**TCP RTT** (namespace `NS`):

```promql
histogram_quantile(0.99, sum(rate(netobserv_namespace_rtt_seconds_bucket{SrcK8S_Namespace="NS"}[2m])) by (le))
sum(rate(netobserv_namespace_rtt_seconds_sum{SrcK8S_Namespace="NS"}[2m])) / sum(rate(netobserv_namespace_rtt_seconds_count{SrcK8S_Namespace="NS"}[2m]))
```

**Network policy denials** (namespace `NS`) — **`action="drop"`** is required on the events metric:

```promql
sum(rate(netobserv_namespace_network_policy_events_total{
  action="drop",
  SrcK8S_Namespace="NS"
}[2m])) by (type, direction, SrcK8S_Namespace, DstK8S_Namespace)
or sum(rate(netobserv_namespace_network_policy_events_total{
  action="drop",
  DstK8S_Namespace="NS"
}[2m])) by (type, direction, SrcK8S_Namespace, DstK8S_Namespace)
```

**Packet drops involving a namespace** (kernel or OVS — pair with policy flows above):

```promql
sum(rate(netobserv_namespace_drop_packets_total{
  K8S_FlowLayer="app",
  SrcK8S_Namespace="NS"
}[2m])) by (SrcK8S_Namespace, DstK8S_Namespace)
or sum(rate(netobserv_namespace_drop_packets_total{
  K8S_FlowLayer="app",
  DstK8S_Namespace="NS"
}[2m])) by (SrcK8S_Namespace, DstK8S_Namespace)
```

On OpenShift, policy-blocked pod traffic may show **`OVS_DROP_EXPLICIT`** in flow `PktDropLatestDropCause` while a naïve `drop_packets_total` query returns empty — check flow logs and `network_policy_events_total` before concluding “no drops”.

Cite the **exact metric name**, **label selectors** (`DstK8S_Namespace` for DNS RCODE, `SrcK8S_Namespace` for egress/app traffic, `DnsFlagsResponseCode`, …), and **numeric result** in the final answer. If a query returns empty, retry with `DstK8S_Namespace` (DNS responses) then `SrcK8S_Namespace`, then drop `K8S_FlowLayer`, then query Loki flow logs — before concluding data is missing.

## Quality Standards

- Cite the exact `netobserv_*` metric name returned by discovery or the FlowCollector config.
- Show PromQL with an explicit range window for counters.
- Warn before recommending many `workload_*` metrics or custom FlowMetric labels with high cardinality.
- Distinguish “metric not enabled” from “no traffic” — zero `rate()` can be legitimate.
- If tools cannot query Prometheus, tell the user which metric and query to run rather than fabricating numbers.
