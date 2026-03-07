# Kovalski: K8s SRE Agent Operator

Kubernetes SRE Agent Operator written in Rust 1.93+ (edition 2024) that runs inside a Kind cluster. It exposes REST endpoints to inspect pod logs and automatically fix broken deployments using Claude CLI as the AI reasoning engine.

<img src="logo.png" width="400">

## Stack

* Rust 1.93+ (edition 2024)
* kube-rs
* axum
* tokio
* Claude CLI (child process)
* Kind (Kubernetes in Docker)

## Endpoints

* `GET /logs` - reads all pod logs across namespaces
* `POST /fix` - collects diagnostics from failing pods/deployments, sends context to Claude CLI, gets corrected YAML, applies fixes to the cluster

## Broken Specs

The `specs/` folder contains intentionally broken K8s manifests:

* `broken-deployment-wrong-image.yaml` - references non-existent image `nginxxxxx:latesttttt`
* `broken-deployment-bad-port.yaml` - readiness/liveness probes on port 9999 while nginx listens on 80
* `broken-deployment-missing-env.yaml` - postgres without required `POSTGRES_PASSWORD` env var

## How to Run

```bash
export ANTHROPIC_API_KEY="your-key"
./start.sh
```

```bash
curl http://localhost:30080/logs
curl -X POST http://localhost:30080/fix
```

```bash
./stop.sh
```

## Build

```bash
cd operator
cargo build --release
```
