---
name: netobserv-metrics
description: Query and interpret NetObserv network flow metrics in Prometheus (netobserv_*). Use for namespace or workload traffic, packet drops, RTT latency, DNS latency, network policy events, or missing NetObserv metrics on OpenShift.
---

# NetObserv Metrics

NetObserv exports **Prometheus metrics** derived from network flow records. They are scraped by cluster monitoring (not shipped by the NetObserv operator itself). Metric names in Prometheus are prefixed with `netobserv_`.

Use this skill when the user asks about network bandwidth, flow counts, packet drops, latency, DNS performance, or whether NetObserv metrics exist.

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

**Counters** (names ending in `_total`): always use `rate()` or `increase()` over a range window (for example `[5m]`).

Examples:

```promql
# Namespace ingress bandwidth (bytes/s)
sum(rate(netobserv_namespace_ingress_bytes_total{namespace="my-app"}[5m]))

# Top namespaces by egress packets
topk(10, sum(rate(netobserv_namespace_egress_packets_total[5m])) by (namespace))

# Packet drops in a namespace (if metric exists)
sum(rate(netobserv_namespace_drop_packets_total{namespace="my-app"}[5m]))
```

**Histograms** (RTT, DNS latency): use `histogram_quantile` on `_bucket` series, or average via `_sum` / `_count` as documented for custom FlowMetrics.

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

## Quality Standards

- Cite the exact `netobserv_*` metric name returned by discovery or the FlowCollector config.
- Show PromQL with an explicit range window for counters.
- Warn before recommending many `workload_*` metrics or custom FlowMetric labels with high cardinality.
- Distinguish “metric not enabled” from “no traffic” — zero `rate()` can be legitimate.
- If tools cannot query Prometheus, tell the user which metric and query to run rather than fabricating numbers.
