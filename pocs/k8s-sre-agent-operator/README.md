# Kovalski: K8s SRE Agent Operator

Kubernetes SRE Agent Operator written in Rust 1.93+ (edition 2024) that runs inside a Kind cluster. The `kovalski` CLI runs on the host and uses Claude CLI locally for AI-powered log analysis and auto-remediation of broken deployments.

<img src="logo.png" width="400">

## Stack

* Rust 1.93+ (edition 2024)
* kube-rs
* axum
* tokio
* reqwest
* Claude CLI (local host process)
* podman
* Kind (Kubernetes in Docker)
* React + TypeScript + Vite
* TanStack React Query
* bun

## Operator Endpoints

* `GET /logs` - reads all pod logs across namespaces
* `GET /status` - runs `kubectl get all -A`
* `GET /diagnostics` - collects diagnostic data from unhealthy pods/deployments
* `POST /apply` - applies raw YAML to the cluster
* `POST /fix` - server-side fix (legacy)
* `GET /api/status` - JSON cluster objects with YAML
* `GET /api/logs` - pod logs
* `GET /api/history` - event history
* `POST /api/fix` - JSON fix result

## kovalski CLI

```
kovalski logs           Read all pod logs from the cluster
kovalski status         Show all resources in the cluster (kubectl get all)
kovalski logs-summary   Summarize logs using Claude AI
kovalski fix            Fix broken deployments using Claude AI
kovalski ui             Open the web UI in the browser
```

The `fix` command collects diagnostics from the operator, sends them to Claude CLI locally, saves fixed specs to `fixed-specs/`, and applies them to the cluster.

The `logs-summary` command fetches all logs and asks Claude to summarize findings with actionable recommendations.

## Web UI

Run `kovalski ui` to open the web UI with 4 tabs:

* **Cluster** - All cluster objects grouped by kind, click any row to see YAML in a fullscreen modal
* **Logs** - Pod logs with auto-refresh toggle (5s)
* **Fix** - Run fix and see diagnostics, Claude analysis, kubectl output
* **History** - Timeline of all cluster changes with expandable details

## Broken Specs

The `specs/` folder contains intentionally broken K8s manifests:

* `broken-deployment-wrong-image.yaml` - references non-existent image `nginxxxxx:latesttttt`
* `broken-deployment-bad-port.yaml` - readiness/liveness probes on port 9999 while nginx listens on 80
* `broken-deployment-missing-env.yaml` - postgres without required `POSTGRES_PASSWORD` env var

## How to Run

```bash
./build.sh
./start.sh
```

```bash
kovalski status
kovalski logs
kovalski logs-summary
kovalski fix
```

```bash
./stop.sh
```

## Build

```bash
./build.sh
```
