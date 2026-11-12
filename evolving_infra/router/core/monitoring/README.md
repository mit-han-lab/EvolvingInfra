<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
SPDX-License-Identifier: Apache-2.0
-->

# sgl-router monitoring

The router exposes Prometheus metrics on `/metrics` (text/plain, version 0.0.4) on its
serving port: `--port`, `30900` under `evolving-infra serve`.

## Files

- `grafana-dashboard.json` — importable Grafana dashboard, **SGLang Router**
  (uid `sgl-router-experimental`).

## Metrics

Every scrape returns these families. The dashboard graphs all of them except
`sgl_router_requests_total` and `sgl_router_ingress_tokenize_errors_total`.

| Metric | Type | What it shows |
|---|---|---|
| `sgl_router_requests_total` | Counter | **Edge intake** — every request received at the router HTTP boundary, by `route`, `method`, counted before worker dispatch (true intake) |
| `sgl_router_responses_total` | Counter | **Edge responses** — every response returned, by `route`, `method`, `status_code` (incl. early-exit 400/413/503). `requests_total - responses_total` = received-but-not-answered |
| `sgl_router_worker_requests_total` | Counter | Per-worker **dispatches** by `worker_url`, `model_id`, `mode`, `outcome` (recorded after dispatch; blind to pre-dispatch drops) |
| `sgl_router_request_duration_seconds` | Histogram | End-to-end request latency by `model_id` |
| `sgl_router_ttft_seconds` | Histogram | Time to first token (streaming) by `model_id` |
| `sgl_router_overlap_blocks` | Histogram | Cache-aware-zmq overlap blocks by `model_id` |
| `sgl_router_active_load` | Gauge | Per-worker prefill-token / decode-block load |
| `sgl_router_workers` | Gauge | Registered worker count by `mode` |
| `sgl_router_worker_health` | Gauge | Per-worker health (1=breaker admits, 0=open) |
| `sgl_router_worker_cb_state` | Gauge | Per-worker circuit breaker state (0=closed, 1=open, 2=half_open) |
| `sgl_router_worker_inflight_requests` | Gauge | In-flight requests per worker |
| `sgl_router_stale_requests_total` | Counter | Stale-request cancellations |
| `sgl_router_decode_affinity_total` | Counter | PD decode-affinity outcomes |
| `sgl_router_sticky_total` | Counter | Sticky-session selection outcomes |
| `sgl_router_ingress_tokenize_errors_total` | Counter | Chat requests whose router-side tokenization failed and fell back to engine-side tokenization |

The `sgl_router_workers` / `sgl_router_worker_*` gauges are sampled from the
live worker registry on every scrape, so a removed worker stops emitting
series immediately rather than leaving a stale value.

These families appear only when the setting that produces them is on; the dashboard does not
graph them:

| Families | Emitted with |
|---|---|
| `sgl_router_decode_charged_tokens`, `sgl_router_decode_charged_requests` (gauges, per decode worker) | `--pd-decode-policy min-inflight-tokens` or `po2-tokens`, or `--pd-late-decode choose` |
| `sgl_router_late_d_*`: requests, decisions, role gate, rebinds, fit none, finals, mismatch, legs, failures, link messages dropped, pending, link up | `--pd-late-decode choose` (the `paper` preset under `--policy symmetric_pd`) |
| `sgl_router_late_d_packed_prompts_total`, `sgl_router_late_d_worker_refused_total` | `--pd-late-decode choose` or `--pd-late-decode-packed-prompt` |
| `sgl_router_decode_po2_smg_*` | `--pd-decode-policy power-of-two-smg` |
| `sgl_router_image_layout_*` | `--image-layout-config` |
| `sgl_router_late_d_image_source_*` | `--late-d-image-source-url` |

## Prometheus scrape config

Point Prometheus at the router's `/metrics` endpoint:

```yaml
scrape_configs:
  - job_name: sgl-router
    metrics_path: /metrics
    static_configs:
      - targets:
          - '127.0.0.1:30900'   # router host:port
```

## Import into Grafana

1. **Dashboards → New → Import**.
2. Upload `grafana-dashboard.json` (or paste its contents).
3. When prompted, select your Prometheus data source for the `Datasource`
   variable. The dashboard uses a templated data source, so it imports into
   any Grafana without editing the JSON.

The top bar exposes `model_id` and `worker_url` template variables (both
default to *All*) to scope the panels.
