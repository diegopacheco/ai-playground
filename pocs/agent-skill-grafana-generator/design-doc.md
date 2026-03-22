# Grafana Generator — Design Doc

## Overview

A Claude Code / Codex agent skill that reads the entire codebase, auto-detects the language/framework, discovers all HTTP endpoints, and generates a working Grafana + Prometheus observability stack running in podman containers. The generated dashboard covers two areas: machine-level metrics (CPU, memory, disk, I/O, network) and application-level metrics (request counters per endpoint, error counters per endpoint, latency distribution at p50, p70, p90, p95, p99, p99.9).

The skill only generates the observability stack. It does not start or manage the application itself — the user runs their app separately.

## Supported Stacks (Auto-Detected)

| Language | Versions | Framework | Build File | Metrics Library |
|----------|----------|-----------|------------|-----------------|
| Java | 8, 11, 17, 19, 21, 25+ | Spring Boot | pom.xml / build.gradle | Micrometer + Prometheus registry |
| Rust | stable | Tokio + Actix-web, Tokio + Axum | Cargo.toml | actix-web-prom, axum-prometheus, prometheus-client |
| Python | 3.x | Django | requirements.txt / pyproject.toml | django-prometheus |
| Go | 1.x | Gin Gonic | go.mod | prometheus/client_golang + gin middleware |

No other stacks are supported. If the detected stack does not match one of these, the skill stops and tells the user.

## Detection Phase

### Step 1 — Identify the Stack

Scan the project root for build files in this order:

1. `pom.xml` or `build.gradle` → Java / Spring Boot. Read Java version from `<java.version>`, `sourceCompatibility`, or `release` property.
2. `Cargo.toml` → Rust. Check dependencies for `actix-web` + `tokio` or `axum` + `tokio`.
3. `requirements.txt` or `pyproject.toml` → Python. Check for `django` in dependencies.
4. `go.mod` → Go. Check for `github.com/gin-gonic/gin` in requires.

### Step 2 — Discover the Application Port

| Stack | Where to Find Port |
|-------|--------------------|
| Spring Boot | `application.properties` or `application.yml` → `server.port` (default 8080) |
| Actix-web | `HttpServer::new(...).bind("addr:port")` in source code (default 8080) |
| Axum | `TcpListener::bind("addr:port")` or `.serve()` call (default 3000) |
| Django | `manage.py runserver` default or `settings.py` (default 8000) |
| Gin Gonic | `router.Run(":port")` in source code (default 8080) |

### Step 3 — Discover All HTTP Endpoints

The skill reads every source file and extracts endpoint definitions:

| Stack | Pattern to Scan |
|-------|-----------------|
| Spring Boot | `@GetMapping`, `@PostMapping`, `@PutMapping`, `@DeleteMapping`, `@PatchMapping`, `@RequestMapping` annotations with path values |
| Actix-web | `web::resource("path").route(web::get().to(...))`, `#[get("path")]`, `#[post("path")]` macros |
| Axum | `Router::new().route("path", get(handler))`, `.route("path", post(handler))` |
| Django | `urlpatterns` entries in `urls.py` files, `path("...", view)`, `re_path(...)` |
| Gin Gonic | `router.GET("path", ...)`, `router.POST("path", ...)`, `group.GET(...)` route registrations |

All discovered endpoints are used to generate per-endpoint panels in the dashboard.

### Step 4 — Check Metrics Library Presence

The skill checks if the Prometheus metrics library is already in the project dependencies.

Per-stack dependency check:

| Stack | Required Dependencies | Metrics Endpoint |
|-------|----------------------|------------------|
| Spring Boot | `spring-boot-starter-actuator` + `micrometer-registry-prometheus` in pom.xml/build.gradle | `/actuator/prometheus` |
| Actix-web | `actix-web-prom` in Cargo.toml | `/metrics` |
| Axum | `axum-prometheus` or `metrics-exporter-prometheus` in Cargo.toml | `/metrics` |
| Django | `django-prometheus` in requirements.txt/pyproject.toml + middleware in `settings.py` MIDDLEWARE list | `/metrics` |
| Gin Gonic | `github.com/prometheus/client_golang` in go.mod | `/metrics` |

If the required dependencies are missing, the skill prints the exact dependency snippet to add (Maven XML, Cargo.toml entry, pip package, or go get command) and the minimal instrumentation code needed. The dashboard is still generated assuming standard metric names.

## Prometheus Metric Names per Stack

### Request Counter (total requests per endpoint)

| Stack | Metric Name | Labels |
|-------|------------|--------|
| Spring Boot | `http_server_requests_seconds_count` | `method`, `uri`, `status` |
| Actix-web | `http_requests_total` | `method`, `path`, `status` |
| Axum | `http_requests_total` | `method`, `path`, `status` |
| Django | `django_http_requests_total_by_view_transport_method` | `view`, `method`, `transport` |
| Gin Gonic | `gin_requests_total` | `method`, `path`, `status` |

### Error Counter (error requests per endpoint)

Derived from the request counter by filtering on `status` label:
- `status=~"5.."` for server errors
- `status=~"4.."` for client errors

### Latency Distribution (histogram)

| Stack | Metric Name | Labels |
|-------|------------|--------|
| Spring Boot | `http_server_requests_seconds` | `method`, `uri`, `status` (histogram with `le` buckets) |
| Actix-web | `http_requests_duration_seconds` | `method`, `path` (histogram) |
| Axum | `http_requests_duration_seconds` | `method`, `path` (histogram) |
| Django | `django_http_requests_latency_seconds_by_view_method` | `view`, `method` (histogram) |
| Gin Gonic | `gin_request_duration_seconds` | `method`, `path` (histogram) |

Percentiles computed via `histogram_quantile()`: p50 (0.5), p70 (0.7), p90 (0.9), p95 (0.95), p99 (0.99), p99.9 (0.999).

### Histogram Bucket Warning

For p99.9 to be meaningful, the application needs fine-grained histogram buckets. Default bucket configurations vary:

| Stack | Default Buckets | Sufficient for p99.9? |
|-------|----------------|----------------------|
| Spring Boot (Micrometer) | 0.001 to 30s, 20+ buckets | Yes |
| Actix-web (actix-web-prom) | 0.005 to 10s, ~11 buckets | Marginal — skill prints recommendation to add finer buckets |
| Axum (axum-prometheus) | 0.005 to 10s, ~11 buckets | Marginal — skill prints recommendation to add finer buckets |
| Django (django-prometheus) | Prometheus client defaults, ~11 buckets | Marginal — skill prints recommendation |
| Gin Gonic | Prometheus client defaults, ~11 buckets | Marginal — skill prints recommendation |

When default buckets are too coarse, the skill prints the code snippet to configure finer buckets.

## Machine-Level Metrics

Collected by `node_exporter` running as a container alongside Grafana and Prometheus.

| Category | Metric | PromQL |
|----------|--------|--------|
| CPU | CPU usage % | `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)` |
| Memory | Memory used % | `(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100` |
| Memory | Memory used bytes | `node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes` |
| Disk | Disk usage % | `(1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100` |
| Disk | Disk I/O read bytes/s | `rate(node_disk_read_bytes_total[1m])` |
| Disk | Disk I/O write bytes/s | `rate(node_disk_written_bytes_total[1m])` |
| Network | Network received bytes/s | `rate(node_network_receive_bytes_total[1m])` |
| Network | Network transmitted bytes/s | `rate(node_network_transmit_bytes_total[1m])` |

## Dashboard Layout

Single Grafana dashboard JSON with two row sections:

### Row 1 — Machine Metrics

| Panel | Type | Width |
|-------|------|-------|
| CPU Usage % | Time series | 8 |
| Memory Usage % | Time series | 8 |
| Memory Used (bytes) | Time series | 8 |
| Disk Usage % | Gauge | 6 |
| Disk I/O (read + write) | Time series | 9 |
| Disk I/O IOPS | Time series | 9 |
| Network In (bytes/s) | Time series | 12 |
| Network Out (bytes/s) | Time series | 12 |

### Row 2 — Application Metrics

| Panel | Type | Width |
|-------|------|-------|
| Total Requests/sec (all endpoints) | Stat | 6 |
| Total Errors/sec (all endpoints) | Stat | 6 |
| Error Rate % | Time series | 12 |
| Requests/sec per Endpoint | Time series | 12 |
| Errors/sec per Endpoint | Time series | 12 |
| Latency All Percentiles (p50, p70, p90, p95, p99, p99.9 overlaid) | Time series | 24 |
| Latency Distribution Heatmap | Heatmap | 24 |

The latency panel uses a single time series with 6 overlaid lines (one per percentile) for compact comparison.

Dashboard properties:
- `uid` — deterministic, derived from project directory name
- `refresh` — 10s auto-refresh
- `time` — default last 15 minutes
- `templating` — datasource variable for Prometheus

## Grafana Configuration

The podman-compose sets these Grafana environment variables:
- `GF_SECURITY_ADMIN_PASSWORD=admin` — default admin password
- `GF_AUTH_ANONYMOUS_ENABLED=true` — no login required to view dashboards
- `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer` — anonymous users get viewer role

The user opens `http://localhost:3000` and sees the dashboard immediately without login.

## Generated Files

### Output Directory Layout

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

### podman-compose.yaml

Three containers:

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| grafana | docker.io/grafana/grafana:latest | 3000 | Dashboard UI |
| prometheus | docker.io/prom/prometheus:latest | 9090 | Metrics store, scrapes app + node_exporter |
| node-exporter | docker.io/prom/node-exporter:latest | 9100 | Machine-level metrics |

Volumes:
- `./grafana/provisioning` mounted into Grafana at `/etc/grafana/provisioning`
- `./grafana/dashboards` mounted into Grafana at `/var/lib/grafana/dashboards`
- `./prometheus/prometheus.yml` mounted into Prometheus at `/etc/prometheus/prometheus.yml`

### prometheus.yml

Scrape configs:
- `job: app` — scrapes the application at `host.containers.internal:<app-port>/metrics` (or `/actuator/prometheus` for Spring Boot)
- `job: node` — scrapes node-exporter at `node-exporter:9100/metrics`

Scrape interval: 5s.

### run-grafana.sh

Starts the observability stack only (not the application). Runs `podman-compose up -d`, then waits for Grafana to be healthy using a loop that checks `curl -s http://localhost:3000/api/health` with max sleep 1. Prints the Grafana URL when ready. The user is responsible for starting their application separately.

### stop-grafana.sh

Runs `podman-compose down` to stop and remove all containers.

### test-grafana.sh

Verifies the observability stack is working:
1. Checks Grafana is healthy via `curl -s http://localhost:3000/api/health`
2. Checks Prometheus is healthy via `curl -s http://localhost:9090/-/healthy`
3. Checks the dashboard is loaded via Grafana API `curl -s http://localhost:3000/api/dashboards/uid/<dashboard-uid>`
4. Checks the Prometheus datasource is configured via `curl -s http://localhost:3000/api/datasources`
5. Prints pass/fail for each check

### datasources.yaml (Grafana provisioning)

Configures Prometheus as the default datasource pointing to `http://prometheus:9090`.

### dashboards.yaml (Grafana provisioning)

Configures Grafana to load dashboards from `/var/lib/grafana/dashboards` on startup.

## Skill Project Structure

```
agent-skill-grafana-generator/
  SKILL.md              — the entire agent skill definition
  install.sh            — copies SKILL.md to ~/.claude/skills/grafana-generator/
  uninstall.sh          — removes ~/.claude/skills/grafana-generator/
  design-doc.md         — this document
  README.md             — usage instructions
  sample/               — sample generated output for a Spring Boot app
```

## install.sh

1. Creates `$HOME/.claude/skills/grafana-generator/` directory
2. Copies `SKILL.md` into that directory
3. Prints confirmation with the skill trigger name

## uninstall.sh

1. Removes `$HOME/.claude/skills/grafana-generator/` directory
2. Prints confirmation

## Generation Flow

```
1. Detect stack (build file scan)
       |
       v
2. Detect app port (config/source scan)
       |
       v
3. Discover all HTTP endpoints (source code scan)
       |
       v
4. Check metrics library presence (dependency scan)
       |
       v
5. Print warnings if metrics library missing or histogram buckets too coarse
       |
       v
6. Generate prometheus.yml (with correct scrape target and metrics path)
       |
       v
7. Generate datasources.yaml + dashboards.yaml (provisioning)
       |
       v
8. Generate dashboard.json (machine + app panels with correct metric names)
       |
       v
9. Generate podman-compose.yaml (grafana + prometheus + node-exporter)
       |
       v
10. Generate run-grafana.sh + stop-grafana.sh + test-grafana.sh
```

## Edge Cases

- **Unsupported stack detected**: Stop and tell the user which stacks are supported
- **No build file found**: Stop and tell the user no supported project was detected
- **No endpoints found**: Generate dashboard with machine metrics only, warn about missing app metrics
- **Metrics library not in dependencies**: Generate dashboard anyway, print the exact dependency snippet and minimal instrumentation code the user needs to add
- **Histogram buckets too coarse for p99.9**: Print code snippet to configure finer buckets for the detected stack
- **Non-standard metrics endpoint**: Spring Boot uses `/actuator/prometheus`, others use `/metrics`. If a custom path is detected in code, use it
- **Existing grafana/ directory**: Warn before overwriting, ask for confirmation
- **Monorepo**: Scan from current working directory only, not repo root
- **Multiple ports detected**: Use the first one found, print which port was selected
- **Java version not in supported list**: Warn but proceed if Spring Boot is detected, since metric names are framework-dependent not JDK-dependent
- **Django middleware not registered**: If `django-prometheus` is in dependencies but `django_prometheus.middleware` is not in MIDDLEWARE list in settings.py, warn the user with the exact middleware entry to add
