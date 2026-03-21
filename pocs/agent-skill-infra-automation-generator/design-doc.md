# Infra Automation Generator — Design Doc

## Overview

A Claude Code / Codex agent skill that scans a project, auto-detects the language/framework, and generates infrastructure files based on user choices. The skill presents a single multi-choice menu, asks targeted follow-ups, then generates everything in the correct dependency order.

## Supported Languages (Auto-Detected)

| Language | Frameworks |
|----------|-----------|
| Java | Spring Boot, Quarkus, Micronaut |
| Go | Gin, Echo, Chi, stdlib |
| Rust | Actix-web, Axum, Rocket |
| Node.js | Express, Fastify, NestJS |
| Python | Django, Flask, FastAPI |

Detection is done by scanning for build files (`pom.xml`, `build.gradle`, `go.mod`, `Cargo.toml`, `package.json`, `requirements.txt`, `pyproject.toml`) and framework-specific imports/annotations.

## Generators

### 1. Containerfile Generator

Generates an optimized multi-stage `Containerfile` tailored to the detected language, plus:

- `podman-compose.yaml`
- `start.sh`
- `stop.sh`
- `test.sh`

The Containerfile uses multi-stage builds to minimize image size. The compose file wires up the app with any detected dependencies (databases, caches). Scripts use podman-compose (never docker-compose).

### 2. Kubernetes Manifest Generator

Generates plain K8s YAML manifests:

- `Deployment` — replicas, resource limits, health probes based on framework conventions
- `Service` — ClusterIP by default, port derived from app config
- `ConfigMap` — externalized config from detected properties/env files
- `HPA` — horizontal pod autoscaler with sensible defaults (min 2, max 10, 70% CPU target)
- `Namespace` — optional, named after the project

All manifests reference the container image name from the Containerfile generator if both are selected.

### 3. Helm Chart Generator

Generates a complete Helm chart structure:

```
chart/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
    configmap.yaml
    hpa.yaml
    _helpers.tpl
```

Values are parameterized from the detected app config. If the user also selected K8s manifests, the skill warns about overlap and asks whether to generate both or only the Helm chart.

### 4. OpenTofu Generator

Generates OpenTofu modules for AWS infrastructure:

- `main.tf` — primary resources
- `variables.tf` — input variables with defaults
- `outputs.tf` — useful outputs (endpoints, ARNs)
- `providers.tf` — AWS provider config
- `backend.tf` — S3 backend stub

Supported AWS patterns:

| Pattern | Resources |
|---------|-----------|
| VPC | VPC, subnets (public/private), NAT gateway, route tables, security groups |
| ECS | ECS cluster, task definition, service, ALB, target group, IAM roles |
| RDS | RDS instance, subnet group, security group, parameter group |
| S3 | Bucket, versioning, encryption, lifecycle rules |
| ElastiCache | Redis cluster, subnet group, security group |

The skill asks which patterns to include based on what it detects in the project (e.g., database connection strings trigger RDS, cache config triggers ElastiCache).

### 5. Ansible Generator

Alternative to K8s. Generates Ansible playbooks for provisioning and deploying the app to bare VMs:

- `playbook.yaml` — main playbook
- `inventory/hosts.yaml` — inventory template
- `roles/app/tasks/main.yaml` — app deployment tasks
- `roles/app/templates/` — config templates (Jinja2)
- `group_vars/all.yaml` — variables

The user chooses between K8s (default) or Ansible. They cannot select both.

## User Interaction Flow

### Step 1 — Auto-Detection

The skill scans the project and reports what it found:

```
Detected: Java / Spring Boot (pom.xml)
  - Port: 8080
  - Database: PostgreSQL (spring.datasource.url)
  - Cache: Redis (spring.redis.host)
```

### Step 2 — Multi-Choice Menu

```
What do you want to generate? (select all that apply)

[x] 1. Containerfile + podman-compose + scripts
[ ] 2. Kubernetes manifests
[ ] 3. Helm chart
[ ] 4. OpenTofu modules (AWS)
[ ] 5. Ansible playbooks
```

### Step 3 — Follow-Up Questions (only if relevant)

- If both K8s manifests AND Helm chart are selected: "Helm chart is a superset of K8s manifests. Generate both anyway? (y/n)"
- If both K8s AND Ansible are selected: "K8s and Ansible are alternatives. Which one? (k8s is default)"
- If OpenTofu is selected: "Which AWS patterns? (VPC, ECS, RDS, S3, ElastiCache)" — pre-checked based on detected dependencies

### Step 4 — Generation

Generators run in dependency order:

```
1. Containerfile   (produces image name used by downstream generators)
       |
       v
2. K8s / Helm / Ansible   (references container image)
       |
       v
3. OpenTofu   (references container image for ECS task definition)
```

If Containerfile is not selected but K8s/Helm/OpenTofu are, the skill uses a placeholder image name `<project-name>:latest` and notes it in the output.

## Project Structure

```
skills/
  infra-automation-generator/
    SKILL.md              — skill definition and orchestrator prompt
    install.sh            — installs the skill into Claude Code
    uninstall.md          — uninstall instructions
    README.md             — usage docs
```

The entire skill lives in a single `SKILL.md` file. No sub-skill files. The orchestrator logic (menu, detection, dispatch, generation) is all prompt-driven — no external code dependencies.

## install.sh

The install script:
1. Detects the Claude Code skills directory (`~/.claude/skills/` or project-level)
2. Copies `SKILL.md` into the skills directory
3. Registers the skill name `infra-automation-generator`
4. Prints confirmation

## uninstall.md

Documents manual removal steps:
1. Remove the skill file from the skills directory
2. Remove any skill registration from settings

## Generation Principles

- All generated files follow the detected project conventions (port, app name, dependencies)
- No comments in generated files (per user preference)
- Compact formatting — no unnecessary blank lines
- Containerfile uses podman, never docker
- Scripts (`start.sh`, `stop.sh`, `test.sh`) use podman-compose
- Sleep values never exceed 1 second; loops with condition checks for readiness
- OpenTofu over Terraform naming, but HCL syntax is identical
- Generated files are written to an `infra/` directory at the project root, except Containerfile and compose which go at root level

## Output Directory Layout

```
project-root/
  Containerfile
  podman-compose.yaml
  start.sh
  stop.sh
  test.sh
  infra/
    k8s/
      namespace.yaml
      deployment.yaml
      service.yaml
      configmap.yaml
      hpa.yaml
    helm/
      Chart.yaml
      values.yaml
      templates/
        ...
    tofu/
      main.tf
      variables.tf
      outputs.tf
      providers.tf
      backend.tf
    ansible/
      playbook.yaml
      inventory/
        hosts.yaml
      roles/
        app/
          tasks/main.yaml
          templates/...
      group_vars/
        all.yaml
```

## Edge Cases

- **No build file detected**: Ask the user for language/framework manually
- **Monorepo**: Scan from current working directory, not repo root
- **Existing infra files**: Warn before overwriting, ask for confirmation
- **No dependencies detected**: Generate minimal configs without database/cache resources
- **Multiple databases detected**: Generate resources for all of them, each in its own module/manifest
