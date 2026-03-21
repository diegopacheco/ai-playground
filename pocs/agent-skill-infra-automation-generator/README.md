# Infra Automation Generator

```
  ___ _  _ ___ ___    _
 |_ _| \| | __| _ \  /_\
  | || .` | _||   / / _ \
 |___|_|\_|_| |_|_\/_/ \_\
     _   _   _ _____ ___
    /_\ | | | |_   _/ _ \
   / _ \| |_| | | || (_) |
  /_/ \_\\___/  |_| \___/
   ___ ___ _  _
  / __| __| \| |
 | (_ | _|| .` |
  \___|___|_|\_|
```

A Claude Code / Codex agent skill that scans your project, detects the stack, and generates production-ready infrastructure files in seconds.

## What It Does

Point it at any project. It figures out your language, framework, port, and dependencies — then generates exactly the infra files you need.

### Generators

| Generator | What You Get |
|-----------|-------------|
| **Containerfile** | Multi-stage `Containerfile` + `podman-compose.yaml` + `start.sh` / `stop.sh` / `test.sh` |
| **Kubernetes** | `Deployment`, `Service`, `ConfigMap`, `HPA`, `Namespace` manifests |
| **Helm Chart** | Full chart with `values.yaml`, templated manifests, and `_helpers.tpl` |
| **OpenTofu** | AWS modules — VPC, ECS, RDS, S3, ElastiCache — pick what you need |
| **Ansible** | Playbook, inventory, roles, and group vars for bare VM deployment |

### Supported Stacks

| Language | Frameworks |
|----------|-----------|
| Java | Spring Boot, Quarkus, Micronaut |
| Go | Gin, Echo, Chi, stdlib |
| Rust | Actix-web, Axum, Rocket |
| Node.js | Express, Fastify, NestJS |
| Python | Django, Flask, FastAPI |

## How It Works

```
1. SCAN        Reads build files, source code, and configs
                 |
2. DETECT      Identifies language, framework, port, databases, caches
                 |
3. ASK         Presents a menu — pick one or more generators
                 |
4. RESOLVE     Handles conflicts (K8s vs Ansible, K8s + Helm overlap)
                 |
5. GENERATE    Creates all files in dependency order
                 |
6. SUMMARY     Lists everything generated + next steps
```

## Install

```bash
cd skills/infra-automation-generator
chmod +x install.sh
./install.sh
```

## Usage

In Claude Code or Codex, say any of:

- `generate infra`
- `generate containerfile for my project`
- `create k8s manifests`
- `generate helm chart`
- `create opentofu modules`
- `generate ansible playbook`

The skill will scan your project and walk you through the options.

### The Menu

```
What do you want to generate? (enter numbers separated by commas)

  1. Containerfile + podman-compose + start/stop/test scripts
  2. Kubernetes manifests (Deployment, Service, ConfigMap, HPA)
  3. Helm chart
  4. OpenTofu modules (AWS)
  5. Ansible playbooks

Example: 1,2,4
```

Pick what you need. The skill handles dependencies between generators automatically — Containerfile runs first so K8s/Helm/OpenTofu can reference the image name.

### Smart Follow-Ups

- **K8s + Helm selected?** It warns you that Helm is a superset and asks if you want both.
- **K8s + Ansible selected?** It asks you to pick one (K8s is default).
- **OpenTofu selected?** It asks which AWS patterns you need (VPC, ECS, RDS, S3, ElastiCache) and pre-suggests based on your detected dependencies.

## Output Structure

```
your-project/
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
        _helpers.tpl
        deployment.yaml
        service.yaml
        configmap.yaml
        hpa.yaml
    tofu/
      providers.tf
      backend.tf
      variables.tf
      outputs.tf
      vpc.tf
      ecs.tf
      rds.tf
      s3.tf
      elasticache.tf
    ansible/
      playbook.yaml
      inventory/
        hosts.yaml
      roles/
        app/
          tasks/
            main.yaml
          templates/
            app.env.j2
      group_vars/
        all.yaml
```

Only the generators you selected produce files. Nothing extra.

## Uninstall

See [uninstall.md](skills/infra-automation-generator/uninstall.md).

## Design

See [design-doc.md](design-doc.md) for the full design document with architecture decisions and rationale.
