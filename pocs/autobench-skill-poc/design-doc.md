# AutoBench Skill - Design Document

## Overview

AutoBench is a Claude Code skill that automates iterative performance benchmarking across multiple programming languages. It runs in waves, where each wave attempts optimizations, measures results, and records whether changes improved or degraded performance. The skill enforces honest benchmarking — no shortcuts, hardcoded values, or cheating.

## Supported Languages

| Language | Version |
|---|---|
| Java | 25 |
| Go | 1.25+ |
| Rust | 1.93+ |
| Zig | 0.15+ |
| Scala | 3.7.3 |
| TypeScript | 5.x (with Bun) |

## Benchmark Types

**A) CSV Analytics** — Measure analytics processing over 1M CSV files. Focus on parsing speed, memory usage, and throughput.

**B) HTTP CRUD Stack** — Measure HTTP latency and RPS on a CRUD application using podman + PostgreSQL + nginx + k6 stress testing.

**C) WebServer UUID** — Measure server performance and RPS on a webserver that returns a UUID per request.

**D) Custom** — User types a free-form benchmark description.

## Wave System

A wave is one full cycle of: propose optimizations → user approves → implement → benchmark → record findings.

The user chooses wave count at startup: **1, 3, 5, or 10 waves**.

### Wave Lifecycle

```
┌─────────────────────────────────────────────────┐
│                   WAVE N                        │
│                                                 │
│  1. Agent analyzes current code + past findings │
│  2. Agent lists possible optimizations          │
│     (checkboxes — user approves/rejects each)   │
│  3. Agent implements approved optimizations     │
│  4. Agent runs bench.sh to measure performance  │
│  5. Agent compares results to previous wave     │
│  6. Agent records everything in findings.md     │
│  7. Next wave or done                           │
└─────────────────────────────────────────────────┘
```

## User Interaction Flow

```
User runs: /autobench
         │
         ▼
┌──────────────────────────┐
│ Select language           │
│ (Java/Go/Rust/Zig/Scala/ │
│  TypeScript)              │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Select benchmark type     │
│ A) CSV Analytics          │
│ B) HTTP CRUD Stack        │
│ C) WebServer UUID         │
│ D) Custom (free text)     │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Select wave count         │
│ 1 / 3 / 5 / 10           │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Wave loop begins          │
│ (repeat N times)          │
└──────────────────────────┘
```

## Key Artifacts

### bench.sh
Generated script that runs the actual measurements. Produces structured output with metrics like:
- RPS (requests per second)
- Latency (p50, p95, p99)
- Throughput (MB/s or records/s)
- Memory usage
- CPU usage

This script must be reproducible and honest — no caching tricks, no hardcoded results, no shortcuts.

### findings.md
Cumulative log of all wave results. Format:

```markdown
# AutoBench Findings

## Wave 1 — 2026-03-01
### What was tried
- [optimization description]

### Results
- Metric before: X
- Metric after: Y
- Verdict: BETTER / WORSE / NEUTRAL

### Why
- [explanation of why it helped or hurt]

---

## Wave 2 — 2026-03-01
...
```

### install.sh
Installs the skill + command into Claude Code:
- Copies skill files to `~/.claude/skills/autobench/`
- Registers the `/autobench` command
- Sets up required hooks if any

### uninstall.sh
Removes the skill + command from Claude Code:
- Removes skill files from `~/.claude/skills/autobench/`
- Removes the `/autobench` command registration
- Cleans up hooks

## Optimization Categories

The agent considers optimizations across all layers:

| Layer | Optimizations |
|---|---|
| **Code** | Algorithm choice, data structures, memory allocation, concurrency, SIMD, zero-copy |
| **Architecture** | Connection pooling, async I/O, batching, pipelining, caching strategies |
| **Database** | Indexing, query optimization, prepared statements, bulk operations, connection tuning |
| **Infrastructure** | nginx tuning, kernel params, container resource limits, network config |
| **Design** | Schema changes, denormalization, protocol choice (HTTP/2, gRPC), serialization format |

## Anti-Cheat Rules

The agent must never:
1. Hardcode benchmark results
2. Cache responses that bypass actual computation
3. Skip work that the benchmark intends to measure
4. Use pre-computed data when the benchmark measures computation
5. Return static responses instead of actual processing
6. Reduce dataset size to fake better numbers
7. Disable logging/tracing only during benchmarks
8. Use compiler flags that break correctness for speed

Every optimization must be legitimate and the bench.sh output must reflect real work.

## File Structure

```
autobench-skill-poc/
├── design-doc.md
├── install.sh
├── uninstall.sh
├── skills/
│   └── autobench/
│       └── SKILL.md
├── commands/
│   └── autobench.md
├── templates/
│   ├── bench-csv.sh
│   ├── bench-crud.sh
│   └── bench-uuid.sh
├── findings.md          (generated at runtime)
└── bench.sh             (generated at runtime)
```

## Skill Trigger

The skill is triggered by the `/autobench` command. The command file at `commands/autobench.md` invokes the skill defined in `skills/autobench/SKILL.md`.

## Comparison & Reporting

- Generate a `report.md` at the end with a summary table comparing all waves side-by-side (wave number, metric, delta %)
- Include ASCII charts in findings.md showing performance trends across waves

## Baseline Wave

- Wave 0 is an automatic "naive implementation" baseline — no optimizations, just the simplest correct code
- This gives a clean starting point to measure all improvements against
- All subsequent waves compare their results to Wave 0 and to the previous wave

## Rollback Mechanism

- If a wave makes things worse, the agent offers to rollback that change before the next wave
- The "worse" finding stays recorded in findings.md — all findings are always preserved regardless of rollback
- The next wave builds on the best-known version, not the degraded one
- findings.md marks rolled-back waves clearly so the full history is transparent

## Scope Boundaries

**In scope:**
- Code generation for the chosen language
- Benchmark script generation
- Iterative optimization across waves
- Performance measurement and comparison
- Findings documentation

**Out of scope:**
- Installing language toolchains (user must have them)
- Cloud deployment
- CI/CD integration
- Cross-language comparison in a single run
