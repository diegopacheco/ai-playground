# K8s SRE Agent Operator - Design Doc

## Overview

A Kubernetes SRE Agent Operator written in Rust that runs inside a Kind cluster and provides REST endpoints to inspect and automatically fix broken deployments using Claude CLI as the AI reasoning engine.

## Stack

- **Language**: Rust 1.93+ (edition 2024)
- **K8s Client**: kube-rs
- **Async Runtime**: tokio
- **HTTP Server**: axum
- **AI Engine**: Claude CLI invoked as a child process via `tokio::process::Command`
- **Cluster**: Kind (Kubernetes in Docker)

## Architecture

```
+-------------------+       +-------------------------+
|   User / curl     | ----> |  SRE Agent Operator     |
|                   |       |  (Rust Pod in Kind)      |
+-------------------+       |                         |
                            |  GET /logs              |
                            |  POST /fix              |
                            +---|---------------------+
                                |
                    +-----------+-----------+
                    |                       |
            +-------v-------+     +--------v--------+
            | K8s API Server |     | Claude CLI      |
            | (kube-rs)      |     | (child process) |
            +----------------+     +-----------------+
```

## REST Endpoints

### GET /logs

Reads pod logs across all namespaces in the cluster. Returns aggregated log output so the user can see what is happening.

### POST /fix

1. Collects diagnostic data from the cluster:
   - Pod status and events for failing pods
   - Pod logs from crashlooping or erroring containers
   - Deployment describe output
2. Builds a prompt with all the diagnostic context
3. Invokes Claude CLI: `claude -p "<prompt>" --dangerously-skip-permissions`
4. Parses the response to extract corrected YAML specs
5. Writes the fixed specs to the `specs/` folder
6. Applies them to the cluster via `kubectl apply -f specs/`
7. Returns the fix summary to the caller

## Claude CLI Integration

Follows the same pattern as `agent-debate-club`:

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

Spawned via `tokio::process::Command` with stdout/stderr capture and a timeout.

## Broken Specs (specs/)

The `specs/` folder contains intentionally broken K8s manifests so `/fix` has real problems to solve. Broken scenarios include:

- **Wrong image name**: deployment referencing a non-existent container image
- **Bad port configuration**: container port mismatch or service targeting wrong port
- **Missing environment variables**: app crashing because required env vars are not set
- **Resource limit issues**: memory limit too low causing OOMKill
- **Bad readiness/liveness probes**: probe pointing to wrong path or port

## Scripts

### start.sh

1. Creates a Kind cluster using `kind-config.yaml`
2. Builds the Rust operator container image
3. Loads the image into Kind
4. Applies all specs from `specs/` folder (including the broken ones)
5. Deploys the SRE agent operator into the cluster

### stop.sh

1. Deletes the Kind cluster

## Project Structure

```
k8s-sre-agent-operator/
  design-doc.md
  README.md
  kind-config.yaml
  start.sh
  stop.sh
  specs/
    broken-deployment-wrong-image.yaml
    broken-deployment-bad-port.yaml
    broken-deployment-missing-env.yaml
    sre-agent-operator.yaml
  operator/
    Cargo.toml
    Containerfile
    src/
      main.rs
      routes/
        mod.rs
        logs.rs
        fix.rs
      k8s/
        mod.rs
        diagnostics.rs
        applier.rs
      agent/
        mod.rs
        claude.rs
        runner.rs
```

## Flow

1. User runs `start.sh` - Kind cluster comes up with broken deployments
2. User calls `GET /logs` - sees errors and crashloops
3. User calls `POST /fix` - operator gathers diagnostics, asks Claude to reason about the failures, gets corrected YAML, applies fixes
4. User calls `GET /logs` again - sees pods recovering
5. User runs `stop.sh` to tear down
