---
name: agent-skill-linter
description: Lint a codebase on code quality, best practices, design principles, build and test status, per-test speed (slow at or above 5s), cyclomatic complexity, and expressive naming, then render the results as a modern web report. Use when the user runs /lint or /lint-site, or asks to lint, score, or audit the quality of a project.
---

# agent-skill-linter

A hybrid linter. Deterministic signals come from host tooling; semantic signals
come from your judgment. The two halves merge into one `report.json`.

## Layout (after install)

- `~/.claude/skills/agent-skill-linter/engine/lint.mjs` — deterministic analyzer
- `~/.claude/skills/agent-skill-linter/engine/assemble.mjs` — merges deterministic + semantic into the final report
- `~/.claude/skills/agent-skill-linter/site/` — the web report stack (Podman)
- commands `/lint` and `/lint-site`

## Output (written into the target repo)

- `.lint/deterministic.json` — produced by the engine
- `.lint/semantic.json` — produced by you (the model) during `/lint`
- `.lint/report.json` — merged final report
- `.lint/history/<timestamp>.json` — one trimmed entry per run, for trends

## Scored categories and weights

build 20, tests 20, complexity 15, principles 15, bestPractices 12, codeQuality 10, naming 8.

- Deterministic: build, tests (including the slow flag at 5s), complexity (cyclomatic per function), plus function-length and file-length checks under codeQuality.
- Semantic (your judgment): naming, principles (SOLID, DRY, KISS, separation of concerns), bestPractices, and overall codeQuality.

## How to run

`/lint [path]` then `/lint-site [path]`. See the two command files for the exact steps.
