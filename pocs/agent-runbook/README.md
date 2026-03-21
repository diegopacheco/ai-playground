# Runbook Generator Agent

Claude Code skill that reads an entire codebase, detects service architecture and dependencies, then auto-generates operational runbooks for common failure modes.

## How It Works

The agent performs a 4-pass scan of the codebase:

1. **Infrastructure Discovery** — Containerfiles, Terraform, K8s manifests, CI/CD pipelines, load balancer configs
2. **Service Architecture Discovery** — HTTP endpoints, database connections, message queues, caches, external API calls, resilience patterns
3. **Dependency Mapping** — builds a dependency graph with startup order, fallback detection, and chain propagation
4. **Failure Mode Identification** — matches detected components against a curated failure catalog ranked by blast radius

The output is a single `runbook.md` with:
- Service overview and component inventory
- Dependency map with fallback annotations
- Startup and shutdown order
- Health check commands
- Failure runbooks with real file paths, real ports, real CLI commands
- Configuration drift checks
- Completeness score (% of detected components covered)

## Sample Output

See [sample/runbook.md](sample/runbook.md) for a real generated runbook from a Go/React auction service.

## Install

```bash
./install.sh
```

Copies the skill to `~/.claude/skills/runbook-generator/`. After install, use `/runbook-generator` in Claude Code.

## Uninstall

```bash
./uninstall.sh
```

## Usage

Inside any project directory in Claude Code:

```
/runbook-generator
```

The agent scans the codebase and writes `runbook.md` to the project root.

## Supported Stacks

| Layer | Technologies |
|---|---|
| Languages | Java, Go, Rust, Python, Node.js/TypeScript |
| Frameworks | Spring Boot, Gin, Axum, Actix, Express, Flask, FastAPI |
| Databases | PostgreSQL, MySQL, MongoDB, SQLite, Redis |
| Queues | Kafka, RabbitMQ, SQS, NATS |
| Infra | Kubernetes, Podman, Docker, Terraform, Helm |
| CI/CD | GitHub Actions, Jenkins, GitLab CI |

## Files

| File | Purpose |
|---|---|
| `SKILL.md` | Agent skill definition |
| `design-doc.md` | Architecture and design decisions |
| `install.sh` | Install skill to Claude Code |
| `uninstall.sh` | Remove skill from Claude Code |
| `sample/runbook.md` | Generated runbook from a real project |
