# Design Doc — Prometheus Runbooks Agent Skill

## Problem
On-call engineers waste the first minutes of an incident translating a firing
Prometheus alert into "what do I actually do now". The alert tells you a metric
crossed a threshold; it rarely tells you the likely cause or the first steps.
We want an agent skill that reads the live alerts from any Prometheus server and,
for each firing alert, proposes a concrete runbook step.

## Goal
A generic, installable Claude Code skill that:
1. Reads active alerts from a Prometheus HTTP API (`/api/v1/alerts`).
2. For every firing alert, proposes a runbook: likely cause, ordered
   investigation/remediation steps, a verification check, and an escalation hint.
3. Works against any Prometheus instance and any alert names — not tied to the
   sample app.

## Non-Goals
- No silencing, deleting, or modifying alerts. Read-only.
- No auto-remediation. The skill proposes steps for a human to run.
- No Alertmanager routing/notification setup. We read alerts straight from
  Prometheus, which already exposes them.
- No third-party Python packages.

## Components

### 1. The skill (`skill/prometheus-runbooks/`)
- `SKILL.md` — the agent instructions. Contains the workflow, the rules
  (read-only, never invent data, propose don't execute), a generic **Pattern
  Library** mapping alert categories (availability, errors, CPU, memory, disk,
  latency, queue, certificates) to proposed steps, an **Unknown Alert**
  fallback, and the output format.
- `scripts/fetch_alerts.py` — Python standard library only. Resolves the
  Prometheus URL (arg → `PROMETHEUS_URL` → `http://localhost:9090`), calls
  `/api/v1/alerts`, and prints a normalized JSON array
  (`name`, `state`, `severity`, `instance`, `labels`, `annotations`,
  `runbook_url`, `activeAt`, `value`). It fails loudly on connection errors so
  the agent never fabricates alerts.

### 2. Install / uninstall
- `install.sh` — copies `skill/prometheus-runbooks/` into
  `~/.claude/skills/prometheus-runbooks/` (replacing any prior copy) and lists
  the installed files.
- `uninstall.sh` — removes `~/.claude/skills/prometheus-runbooks/`.

### 3. Sample (`sample/`)
A self-contained podman stack that produces real firing alerts so the skill can
be tried end to end.
- `app/` — a Python stdlib HTTP server exposing `/metrics` with deliberately
  high values (`app_error_rate 0.95`, `app_memory_usage_ratio 0.92`,
  `app_request_latency_seconds 1.8`).
- `prometheus/` — Prometheus built with the config baked in (no volume mounts,
  to avoid SELinux/relabel issues on the podman machine). Scrapes the app, the
  Prometheus itself, and a deliberately missing target (`app:9999`) to trigger
  `InstanceDown`.
- `prometheus/alert.rules.yml` — four rules: `HighErrorRate` (critical),
  `HighMemoryUsage` (warning), `HighLatency` (warning), `InstanceDown`
  (critical). `for: 10s` so they reach `firing` quickly.
- `podman-compose.yml`, `start.sh`, `stop.sh`, `test.sh`.

## Data Flow
```
metrics app  --/metrics-->  Prometheus  --evaluates rules-->  active alerts
                                                  |
                              fetch_alerts.py  <--/api/v1/alerts
                                                  |
                                        normalized JSON
                                                  |
                                 SKILL.md (agent reasoning + Pattern Library)
                                                  |
                                   proposed runbook per firing alert
```

## Key Decisions
- **Read straight from Prometheus, not Alertmanager.** Prometheus exposes active
  alerts at `/api/v1/alerts`; this keeps the sample minimal and the skill works
  whether or not Alertmanager is deployed. The same script also works against an
  Alertmanager-fronted setup because the alerts still live in Prometheus.
- **Config baked into the Prometheus image** instead of bind mounts. On the mac
  podman machine, SELinux relabeling of bind-mounted files is unreliable;
  `COPY` in a Containerfile sidesteps it entirely.
- **Reasoning lives in the agent, fetching lives in the script.** The script is
  dumb and deterministic (just data). The mapping from alert to runbook is the
  agent applying the Pattern Library plus the alert's own labels/annotations.
  This is what makes it generic: unknown alerts still get a sensible proposal.
- **`runbook_url` annotation wins.** If a team already wrote a runbook and linked
  it in the alert annotation, the skill surfaces that first instead of guessing.

## Success Criteria
- `sample/start.sh` brings up the stack; `sample/test.sh` waits for and prints
  at least one firing alert and exits PASS.
- `fetch_alerts.py http://localhost:9090` prints the firing alerts as JSON.
- `install.sh` places the skill under `~/.claude/skills/prometheus-runbooks/`;
  `uninstall.sh` removes it.
- Invoking the skill produces a per-alert runbook proposal that uses the real
  alert names, instances, and values.

## Testing
- `sample/test.sh` — readiness loop + firing-alert assertion (PASS/FAIL).
- Manual: run `fetch_alerts.py` against the running sample and confirm the JSON
  contains `HighErrorRate`, `HighMemoryUsage`, `HighLatency`, `InstanceDown`.
- Install/uninstall round trip verified by listing the installed files.
