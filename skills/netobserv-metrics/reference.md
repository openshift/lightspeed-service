# NetObserv predefined metrics reference

Source: [NetObserv operator Metrics documentation](https://github.com/netobserv/network-observability-operator/blob/main/docs/Metrics.md).

## Naming

- Configure in `FlowCollector` `spec.processor.metrics.includeList` using the **short name** (no prefix).
- Prometheus exposes `netobserv_<short_name>` (for example `namespace_egress_packets_total` → `netobserv_namespace_egress_packets_total`).

## Predefined metrics

`*` = enabled by default. `**` = also enabled by default when Loki is disabled (workload metrics may replace namespace counterparts).

### Core traffic

| Short name | Default | Notes |
|------------|---------|-------|
| `namespace_egress_bytes_total` | | |
| `namespace_egress_packets_total` | | |
| `namespace_ingress_bytes_total` | | |
| `namespace_ingress_packets_total` | * | |
| `namespace_flows_total` | * | |
| `node_egress_bytes_total` | * | |
| `node_egress_packets_total` | | |
| `node_ingress_bytes_total` | * | |
| `node_ingress_packets_total` | * | |
| `node_flows_total` | | |
| `workload_egress_bytes_total` | * | High cardinality |
| `workload_egress_packets_total` | ** | High cardinality |
| `workload_ingress_bytes_total` | * | High cardinality |
| `workload_ingress_packets_total` | ** | High cardinality |
| `workload_flows_total` | ** | High cardinality |
| `workload_sampling` | | |
| `node_to_node_ingress_flows_total` | * | |

### Feature `PacketDrop` (`spec.agent.ebpf.features`, privileged)

| Short name | Default |
|------------|---------|
| `namespace_drop_bytes_total` | |
| `namespace_drop_packets_total` | * |
| `node_drop_bytes_total` | |
| `node_drop_packets_total` | * |
| `workload_drop_bytes_total` | ** |
| `workload_drop_packets_total` | ** |

### Feature `FlowRTT`

| Short name | Default |
|------------|---------|
| `namespace_rtt_seconds` | * |
| `node_rtt_seconds` | |
| `workload_rtt_seconds` | ** |

### Feature `DNSTracking`

| Short name | Default |
|------------|---------|
| `namespace_dns_latency_seconds` | * |
| `node_dns_latency_seconds` | |
| `workload_dns_latency_seconds` | ** |

### Feature `NetworkEvents`

| Short name | Default |
|------------|---------|
| `namespace_network_policy_events_total` | * |
| `node_network_policy_events_total` | |
| `workload_network_policy_events_total` | |

### Feature `IPSec`

| Short name | Default |
|------------|---------|
| `node_ipsec_flows_total` | * |

## PromQL patterns

**Counter rate (bandwidth or QPS):**

```promql
sum(rate(netobserv_namespace_ingress_bytes_total{namespace="TARGET"}[2m]))
```

**Top-N by namespace:**

```promql
topk(10, sum(rate(netobserv_namespace_egress_packets_total[5m])) by (namespace))
```

**Histogram p99 (custom or RTT metrics with `_bucket`):**

```promql
histogram_quantile(0.99, sum(rate(netobserv_namespace_rtt_seconds_bucket[2m])) by (le))
```

**Histogram average:**

```promql
sum(rate(METRIC_sum[2m])) / sum(rate(METRIC_count[2m]))
```

## FlowMetric CR (custom metrics)

- API: `flows.netobserv.io/v1alpha1`, kind `FlowMetric`
- Create in FlowCollector namespace (default `netobserv`)
- Types: `Counter`, `Histogram`
- Use `labels`, `filters`, `valueField`, `direction` per [FlowMetric spec](https://github.com/netobserv/network-observability-operator/blob/main/docs/FlowMetric.md)
- Flow field catalog: [flows-format.adoc](https://github.com/netobserv/network-observability-operator/blob/main/docs/flows-format.adoc) — respect cardinality (`fine` vs `careful` fields)
- Samples: https://github.com/netobserv/netobserv-operator/tree/main/config/samples/flowmetrics

### Example counter (external ingress bytes)

```yaml
apiVersion: flows.netobserv.io/v1alpha1
kind: FlowMetric
metadata:
  name: flowmetric-cluster-external-ingress-traffic
spec:
  metricName: cluster_external_ingress_bytes_total
  type: Counter
  valueField: Bytes
  direction: Ingress
  labels: [DstK8S_HostName, DstK8S_Namespace, DstK8S_OwnerName, DstK8S_OwnerType]
  filters:
  - field: SrcSubnetLabel
    matchType: Absence
```

Prometheus name: `netobserv_cluster_external_ingress_bytes_total`

## Operational impact

- More metrics in `includeList` increase Prometheus load.
- Prefer namespace metrics for cluster-wide questions; enable `workload_*` only when needed.
