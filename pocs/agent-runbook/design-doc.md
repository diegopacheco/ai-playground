# Runbook Generator Agent — Design Doc

## Problem

Operational runbooks are tedious to write manually, often outdated, and inconsistent across services. Engineers skip writing them, leading to longer incident response times and tribal knowledge dependency. Given a codebase, an agent should be able to read the full service architecture and auto-generate actionable runbooks for common failure modes.

## Goal

Build a Claude Code agent skill (`runbook-generator`) that reads an entire codebase, identifies the service architecture (APIs, databases, queues, caches, external dependencies), and produces a `runbook.md` with operational procedures for common failure scenarios.

## Scope

### In Scope
- Read and analyze full codebase (source code, configs, infra files, docker/k8s manifests)
- Detect service architecture: endpoints, databases, message queues, caches, external API calls
- Detect infrastructure: container orchestration, load balancers, health checks, monitoring
- Identify common failure modes based on detected architecture
- Generate `runbook.md` with structured operational procedures
- Support polyglot codebases (Java, Go, Rust, Python, Node.js)

### Out of Scope
- Runtime analysis or live monitoring integration
- Automatic remediation or self-healing
- Generating alerts or dashboards
- Multi-repo service mesh analysis (single repo only)

## Architecture Detection Strategy

The agent performs a multi-pass scan of the codebase:

### Pass 1 — Infrastructure Discovery
Scan for infrastructure definitions:
- `Containerfile`, `Dockerfile`, `docker-compose.yml`, `podman-compose.yml`
- `*.tf`, `*.hcl` (Terraform)
- `k8s/`, `kubernetes/`, `helm/`, `charts/` directories
- `nginx.conf`, `haproxy.cfg`, load balancer configs
- `.github/workflows/`, `Jenkinsfile`, `.gitlab-ci.yml` (CI/CD)

### Pass 2 — Service Architecture Discovery
Scan source code for:
- HTTP endpoints (REST controllers, route definitions, handler registrations)
- Database connections (JDBC, connection strings, ORM configs, migration files)
- Message queue producers/consumers (Kafka, RabbitMQ, SQS, NATS)
- Cache layers (Redis, Memcached, in-memory caches)
- External API calls (HTTP clients, gRPC stubs, SDK usage)
- Scheduled jobs / cron tasks
- Health check endpoints
- Authentication/authorization middleware

### Pass 3 — Dependency Mapping
Build a dependency graph:
- Which services talk to which databases
- Which services produce/consume from which queues
- Which services call which external APIs
- Which services depend on which caches
- Startup order and initialization dependencies

### Pass 4 — Failure Mode Identification
Based on detected architecture, identify applicable failure modes from a catalog:

| Component | Failure Modes |
|---|---|
| Database | Connection pool exhaustion, replication lag, disk full, slow queries, deadlocks, migration failures |
| Message Queue | Consumer lag, dead letter queue overflow, broker unreachable, message poisoning |
| Cache | Cache miss storm, eviction spike, connection timeout, memory pressure |
| External API | Timeout, rate limiting, certificate expiry, DNS failure, breaking change |
| HTTP Service | High latency, 5xx spike, OOM kill, thread pool exhaustion, health check failure |
| Container/K8s | CrashLoopBackOff, OOMKilled, pod eviction, image pull failure, resource quota exceeded |
| CI/CD | Build failure, deployment rollback, config drift, secret rotation |
| Authentication | Token expiry, identity provider outage, certificate rotation |
| Storage/Disk | Disk full, log rotation failure, temp file accumulation |
| Network | DNS resolution failure, TLS handshake failure, connection reset |

## Runbook Output Structure

The generated `runbook.md` follows this structure:

```
# Operational Runbook — {Service Name}

## Service Overview
- Architecture summary (auto-detected)
- Component inventory
- Dependency map (text-based)
- Critical paths

## Health Checks
- Endpoints to verify
- Expected responses
- How to check each dependency

## Failure Runbooks

### {Failure Mode Title}
**Severity:** Critical | High | Medium | Low
**Affected Component:** {component}
**Symptoms:**
- What you will observe

**Diagnosis Steps:**
1. Step-by-step commands to identify root cause

**Resolution Steps:**
1. Step-by-step commands to fix

**Rollback Procedure:**
1. How to revert if resolution fails

**Prevention:**
- What to do to prevent recurrence

---
(repeat for each failure mode)

## Emergency Contacts / Escalation
- Placeholder for teams to fill in

## Appendix
- Detected endpoints
- Detected dependencies
- Detected configuration files
```

## Agent Execution Flow

```
1. User invokes /runbook-generator
2. Agent globs for all source, config, and infra files
3. Pass 1: Infrastructure Discovery
4. Pass 2: Service Architecture Discovery
5. Pass 3: Dependency Mapping
6. Pass 4: Failure Mode Identification (match detected components to failure catalog)
7. For each identified failure mode:
   a. Read relevant source code for specifics (ports, endpoints, connection params)
   b. Generate diagnosis commands tailored to the actual stack
   c. Generate resolution steps with real file paths and config values
8. Assemble runbook.md
9. Output summary of what was detected and how many runbooks were generated
```

## Skill Definition (SKILL.md)

- **name:** `runbook-generator`
- **description:** Reads the full codebase, detects service architecture and dependencies, then generates operational runbooks for common failure modes into `runbook.md`.
- **allowed-tools:** `[Glob, Grep, Read, Bash, Write]`
- **trigger:** `/runbook-generator`
- **output:** `runbook.md` in project root

## Key Design Decisions

### Why read the whole codebase instead of asking the user?
The value is in automation. Engineers already know what their service does — the pain is translating that into structured runbooks. The agent should discover architecture from code, not require manual input.

### Why a single runbook.md instead of multiple files?
One file is easier to search during an incident. `Ctrl+F` in a single doc beats navigating a directory tree at 3am.

### Why text-based dependency maps instead of diagrams?
Text renders everywhere — GitHub, terminals, editors. No external tools needed. Mermaid or ASCII art can be a future enhancement.

### Why a failure mode catalog instead of pure inference?
Pure inference from code is unreliable. A curated catalog of failure modes per component type, filtered by what the agent actually detects in the codebase, gives high-quality results with good coverage.

## Tailoring Strategy

The agent tailors runbook content to the actual codebase:
- Database runbooks include the real connection string location, migration tool, and schema files
- API runbooks include the actual endpoints, ports, and health check paths
- Container runbooks reference the actual Containerfile, compose file, and image names
- Commands use the actual tech stack (e.g., `psql` for Postgres, `redis-cli` for Redis, `kubectl` for K8s)

## Design Principles

### 4-Pass Architecture Scan
Infrastructure first, then service code, then dependency mapping, then failure mode matching. Each pass builds on the previous one. Pass 1 tells Pass 2 what runtimes to look for. Pass 2 feeds the dependency graph in Pass 3. Pass 3 determines which failure modes from the catalog are relevant in Pass 4. No pass runs blind.

### Tailored to the Actual Codebase
Runbooks reference real file paths, real ports, real config locations, real CLI tools for the detected stack. Not generic advice. If the agent finds `application.yml` with `spring.datasource.url=jdbc:postgresql://db:5432/orders`, the database runbook says "check `application.yml:14`" and uses `psql -h db -p 5432 -d orders`, not "check your database configuration file".

### Severity Ranking by Blast Radius
Each failure mode gets a severity based on how much of the system it takes down, not just how likely it is. A database going down on the critical path is Critical. A cache miss storm that degrades latency but doesnt break functionality is Medium. This lets on-call engineers prioritize during multi-failure incidents.

### Runbook Completeness Score
After generation, the agent outputs a completeness score: what percentage of detected components have at least one runbook covering them. If the agent detects Redis, Postgres, Kafka, and an external payment API but only generates runbooks for 3 of 4, the score is 75% and the gap is flagged. This gives teams visibility into blind spots.

### Dependency Chain Failure Propagation
The agent traces dependency chains to generate cascade runbooks. If Service A depends on Database B, and Database B depends on Disk C, the runbook for "Disk C full" includes the downstream impact: "Database B will reject writes, causing Service A to return 503 on POST endpoints". This turns isolated component runbooks into system-aware incident guides.

### Environment-Aware Commands
The agent detects whether the project uses Docker/Podman, K8s, bare metal, or cloud-managed services and generates diagnosis/resolution commands for the right environment. A Postgres runbook for K8s uses `kubectl exec` to get a shell. The same Postgres runbook for podman-compose uses `podman exec`. No guessing which context you are in.

### Configuration Drift Detection Runbook
The agent compares infra-as-code definitions (Terraform, Helm values, compose files) against what the application code expects (env vars read, ports bound, hostnames resolved). Mismatches get a dedicated runbook: "Config says port 8080 but app binds 8081 — heres how to diagnose and fix".

### Graceful Degradation Paths
For each critical dependency, the agent checks if the codebase has fallback logic (circuit breakers, retry policies, fallback responses, feature flags). If fallbacks exist, the runbook documents them: "Redis is down but the app falls back to DB queries — expect higher latency but no outage". If no fallback exists, the runbook flags it: "No fallback detected — Redis failure causes full service outage".

### Startup and Shutdown Order Runbook
Based on dependency mapping, the agent generates the correct startup sequence and safe shutdown sequence. "Start Postgres first, then Redis, then the backend, then the frontend". This prevents the classic "why is the service crash-looping" — because it started before its database.

### Log and Metric Pointers
For each failure mode, the agent scans for logging statements and metric emissions near the relevant code paths. The runbook includes: "Check logs for `connection refused` at `src/db/pool.go:47`" or "Look for metric `http_request_duration_seconds` spiking". Gives on-call engineers the exact breadcrumbs to follow.

## Limitations

- Static analysis only — cannot detect runtime-only dependencies or dynamic service discovery
- Single repo scope — cross-service dependencies are detected only if defined in config
- Failure catalog is finite — novel/custom failure modes wont be covered
- Generated runbooks are a starting point — teams should review and customize
