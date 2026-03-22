# Grafana Generator

Agent skill for Claude Code / Codex that reads your entire codebase, detects the stack, discovers all HTTP endpoints, and generates a working Grafana + Prometheus + node_exporter observability stack using podman.

## Supported Stacks

| Language | Framework | Metrics Library |
|----------|-----------|-----------------|
| Java 8/11/17/19/21/25+ | Spring Boot | Micrometer + Prometheus |
| Rust (Tokio) | Actix-web | actix-web-prom |
| Rust (Tokio) | Axum | axum-prometheus |
| Python 3.x | Django | django-prometheus |
| Go | Gin Gonic | prometheus/client_golang |

## What it Generates

```
project-root/
  grafana/
    dashboards/
      dashboard.json
    provisioning/
      datasources/
        datasources.yaml
      dashboards/
        dashboards.yaml
  prometheus/
    prometheus.yml
  podman-compose.yaml
  run-grafana.sh
  stop-grafana.sh
  test-grafana.sh
```

## Dashboard Panels

### Machine Metrics
- CPU Usage %
- Memory Usage % and bytes
- Disk Usage %
- Disk I/O (read/write bytes and IOPS)
- Network In/Out (bytes/sec)

### Application Metrics
- Total Requests/sec
- Total Errors/sec
- Error Rate %
- Requests/sec per Endpoint
- Errors/sec per Endpoint
- Latency Percentiles (p50, p70, p90, p95, p99, p99.9) overlaid
- Latency Distribution Heatmap

## Install

```bash
./install.sh
```

## Uninstall

```bash
./uninstall.sh
```

## Usage

In Claude Code:
```
/grafana-generator
```

## How to Use the Generated Stack

1. Start your application on the detected port
2. Run `./run-grafana.sh`
3. Open http://localhost:3000
4. Run `./test-grafana.sh` to verify everything is working
5. Run `./stop-grafana.sh` to stop the stack

## Requirements

- podman and podman-compose installed
- Your application must expose a Prometheus metrics endpoint
