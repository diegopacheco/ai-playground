# Kovalski: K8s SRE Agent Operator - Design Doc

## Overview

A Kubernetes SRE Agent Operator written in Rust that runs inside a Kind cluster. The operator exposes REST endpoints for cluster inspection. The `kovalski` CLI tool runs on the host, calls the operator for cluster data, and uses Claude CLI locally for AI-powered diagnostics and auto-remediation.

## Stack

- **Language**: Rust 1.93+ (edition 2024)
- **K8s Client**: kube-rs
- **Async Runtime**: tokio
- **HTTP Server**: axum
- **HTTP Client**: reqwest (CLI)
- **AI Engine**: Claude CLI invoked locally on the host via `tokio::process::Command`
- **Container Runtime**: podman
- **Cluster**: Kind (Kubernetes in Docker)
- **UI**: React + TypeScript + Vite
- **UI Data Fetching**: TanStack React Query
- **UI Package Manager**: bun

## Architecture

```
+---------------------+
|   kovalski CLI      |
|   (runs on host)    |
+-----|----------|----+
      |          |
      | HTTP     | claude -p (local)
      v          v
+-------------+  +-----------------+
| SRE Agent   |  | Claude CLI      |
| Operator    |  | (host process)  |
| (K8s Pod)   |  +-----------------+
|             |
| GET /logs        |
| GET /status      |
| GET /diagnostics |
| POST /apply      |
| POST /fix        |
| GET /api/status  |
| GET /api/history |
| POST /api/fix    |
| Static UI files  |
+---|--------------+
    |
    v
+----------------+
| K8s API Server |
| (kube-rs)      |
+----------------+
```

## Operator REST Endpoints

### GET /logs
Reads pod logs across all namespaces via kube-rs. Returns aggregated log output.

### GET /status
Runs `kubectl get all -A` and returns the output.

### GET /diagnostics
Collects diagnostic data from unhealthy pods and deployments in the default namespace:
- Pod conditions, container states, restart counts
- Waiting/terminated reasons
- Container logs (tail 20 lines)
- Deployment ready vs desired replicas
- Container specs (image, ports, probes)
Skips sre-agent pods.

### POST /apply
Receives raw YAML in the request body, writes it to a temp file, and runs `kubectl apply --validate=false -f` on it.

### POST /fix (legacy, server-side)
Server-side fix flow that calls Claude CLI from inside the pod. Kept for compatibility but the CLI-driven flow is preferred.

### GET /api/status
Returns JSON `Vec<ClusterObject>` with namespace, kind, name, status, ready, restarts, age, yaml. Used by the web UI.

### GET /api/history
Returns JSON `Vec<HistoryEvent>` with timestamp, event_type, summary, details, success. Used by the web UI.

### POST /api/fix
Returns JSON `FixResult` with diagnostics, claude_response, kubectl_output, success. Used by the web UI.

## kovalski CLI

The `kovalski` binary runs on the host and provides these commands:

### kovalski logs
Calls `GET /logs` and prints all pod logs.

### kovalski status
Calls `GET /status` and prints `kubectl get all -A` output.

### kovalski fix
1. Calls `GET /diagnostics` to get cluster issues from the operator
2. Runs `claude -p` locally on the host with the diagnostics as context
3. Extracts YAML from Claude's response
4. Saves each fixed manifest to `fixed-specs/` directory
5. Sends the YAML to `POST /apply` on the operator to apply fixes

### kovalski logs-summary
1. Calls `GET /logs` to get all pod logs from the operator
2. Runs `claude -p` locally on the host to summarize findings
3. Prints what is running, what is failing, why, and recommended actions

### kovalski k8s
Scans the current directory, uses Claude CLI to analyze the project, generate all artifacts, build and deploy to the cluster. No arguments needed - the LLM figures out the app name and port from the source code.

```
kovalski k8s
```

1. Scans the current directory for source files (go, rs, py, js, ts, java, rb, etc.)
2. Calls Claude CLI to analyze the project and detect the app name and port
3. Calls Claude CLI to generate a `Containerfile` based on the project source code
4. Builds the container image with `podman build`
5. Saves and loads the image into the Kind cluster via `kind load image-archive`
6. Calls Claude CLI to generate K8s manifests (1 Deployment, 1 Service type LoadBalancer)
7. Saves the YAML to `specs/<name>.yaml` (user can edit these later if needed)
8. Runs `kubectl apply -f` to deploy
9. Waits for the pod to be ready
10. Prints the LoadBalancer external IP if available (MetalLB)

### kovalski deploy
1. Applies `specs/sre-agent-operator.yaml` to the current cluster via `kubectl apply -f`
2. Waits for the sre-agent-operator pod to be ready (polls every 1s)
3. Installs ServiceAccount, ClusterRole, ClusterRoleBinding, Deployment, and Service

### kovalski ui
Opens the web UI in the default browser.

### Environment
- `KOVALSKI_URL` - base URL of the SRE agent (default: `http://localhost:30080`)

## Claude CLI Integration

Follows the same pattern as `agent-debate-club`. Spawned as a child process on the host:

```rust
pub fn build_command(prompt: &str) -> (String, Vec<String>) {
    (
        "claude".to_string(),
        vec![
            "-p".to_string(),
            prompt.to_string(),
            "--dangerously-skip-permissions".to_string(),
        ],
    )
}
```

Spawned via `tokio::process::Command` with stdout/stderr capture and a 180s timeout.

## Broken Specs (specs/)

The `specs/` folder contains intentionally broken K8s manifests so `kovalski fix` has real problems to solve:

- **Wrong image name**: `nginxxxxx:latesttttt` (non-existent image)
- **Bad probes**: readiness/liveness probes on port 9999 while nginx listens on 80
- **Missing env vars**: postgres without required `POSTGRES_PASSWORD`

## Fixed Specs (fixed-specs/)

When `kovalski fix` runs, the corrected YAML manifests are saved to `fixed-specs/`, named by deployment (e.g. `fixed-specs/broken-bad-port.yaml`).

## Scripts

### build.sh
Builds the UI (`bun install && bun run build`) and both Rust binaries (`sre-agent` and `kovalski`) via `cargo build --release`.

### start.sh
1. Creates a Kind cluster with `kind-config.yaml` (control-plane + worker, port mapping 30080)
2. Builds the UI with bun and copies dist into operator build context
3. Builds the operator container image with podman
4. Saves and loads the image into Kind via `kind load image-archive`
5. Applies all specs from `specs/` (broken deployments + operator)
6. Waits for the operator pod to be ready
7. Starts `kubectl port-forward` in background (30080 -> 8080)

### stop.sh
1. Kills the port-forward process
2. Deletes the Kind cluster

## Project Structure

```
k8s-sre-agent-operator/
  design-doc.md
  README.md
  kind-config.yaml
  build.sh
  start.sh
  stop.sh
  specs/
    broken-deployment-wrong-image.yaml
    broken-deployment-bad-port.yaml
    broken-deployment-missing-env.yaml
    sre-agent-operator.yaml
  fixed-specs/
    (generated by kovalski fix)
  operator/
    Cargo.toml
    Containerfile
    src/
      main.rs
      cli/
        main.rs
      routes/
        mod.rs
        logs.rs
        fix.rs
        status.rs
        diagnostics.rs
        apply.rs
        api_status.rs
        history.rs
      k8s/
        mod.rs
        diagnostics.rs
        applier.rs
      agent/
        mod.rs
        claude.rs
        runner.rs
      history.rs
  ui/
    package.json
    vite.config.ts
    index.html
    src/
      main.tsx
      App.tsx
      api.ts
      index.css
      pages/
        ClusterPage.tsx
        LogsPage.tsx
        FixPage.tsx
        HistoryPage.tsx
      components/
        StatusBadge.tsx
        YamlModal.tsx
  test/
    k8s/
      main.go
      go.mod
      Containerfile
    cluster/
      kind-config.yaml
      start.sh
      stop.sh
```

## Web UI

The operator serves a web UI built with React, Vite, TypeScript, and TanStack Query. Access via `kovalski ui` or directly at the operator URL.

### Stack
- React + TypeScript
- Vite (build tool)
- TanStack React Query (data fetching)
- bun (package manager)
- react-syntax-highlighter (YAML display)

### Tabs

1. **Cluster** - Shows all cluster objects (Pods, Deployments, Services, ReplicaSets) grouped by kind in tables. Click any row to see its full YAML in a modal with fullscreen toggle.
2. **Logs** - Displays all pod logs with auto-refresh toggle (5s interval, ON/OFF badge).
3. **Fix** - Click "Run Fix" to trigger server-side fix. Shows diagnostics, Claude analysis, and kubectl output sections.
4. **History** - Timeline of all cluster changes (fixes, applies) with expandable details.

### API Endpoints (JSON)
- `GET /api/status` - Returns `Vec<ClusterObject>` with namespace, kind, name, status, ready, restarts, age, yaml
- `GET /api/logs` - Returns pod logs as text
- `GET /api/history` - Returns `Vec<HistoryEvent>` with timestamp, event_type, summary, details, success
- `POST /api/fix` - Returns `FixResult` with diagnostics, claude_response, kubectl_output, success

### Build
The UI is built with `bun run build` and the dist is copied into the operator container at `/app/static`. The operator serves it via `tower-http::ServeDir` as a fallback service with SPA routing.

## Test Environment

### test/k8s/
A simple Go HTTP app used for testing `kovalski k8s`. Serves `/` and `/health` on port 8080.

### test/cluster/
A standalone Kind cluster with MetalLB for testing `kovalski deploy` and `kovalski k8s`:
- `start.sh` - Creates Kind cluster, installs MetalLB with L2 advertisement, builds and loads the test-app and sre-agent-operator images
- `stop.sh` - Deletes the Kind cluster

Usage:
```
cd test/cluster && ./start.sh
kovalski deploy
cd ../k8s && kovalski k8s
```

## Flow

1. `./build.sh` - builds UI and Rust binaries (operator + CLI)
2. `./start.sh` - Kind cluster comes up with broken deployments and UI
3. `kovalski status` - see all resources and their states
4. `kovalski logs` - see raw pod logs
5. `kovalski logs-summary` - AI-powered summary of cluster health
6. `kovalski fix` - AI diagnoses issues, generates fixed YAML, saves to `fixed-specs/`, applies to cluster
7. `kovalski status` - verify pods are now healthy
8. `kovalski ui` - open the web UI in browser
9. `kovalski deploy` - install sre-agent on any cluster
10. `kovalski k8s` - analyze project, generate Containerfile + K8s manifests, build, deploy
11. `./stop.sh` - tear down
