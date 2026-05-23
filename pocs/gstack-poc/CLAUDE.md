# CLAUDE.md — pocs/gstack-poc

## Design System

Always read `DESIGN-SYSTEM.md` before making any visual or UI decision in this POC.
All font choices, colors, spacing, border radius, motion, and aesthetic direction
are defined there. Do not deviate without explicit user approval.

## Product Context

`DESIGN.md` is the product design doc: problem statement, architecture decisions,
implementation tasks (T1-T14, DT1-DT14), failure modes, and the review trail.
Read it before starting any implementation task.

## Implementation Boundary

All code stays inside `pocs/gstack-poc/`. Do not write files above this directory.
Suggested layout:

```
pocs/gstack-poc/
  runner/        standalone TS package — Playwright + LLM orchestrator
  protocol/      shared message types (action events, frame events, status events)
  web/           Next.js app — three-pane playground UI
  infra/         Containerfile, podman-compose.yml, deploy config
  DESIGN.md
  DESIGN-SYSTEM.md
  CLAUDE.md      (this file)
```

## Tooling Conventions

Per the parent `~/.claude/CLAUDE.md`:

- Use podman / podman-compose, never docker / docker-compose.
- Containerfile, not Dockerfile.
- Compact bash scripts, no comments, max 1s sleeps.
- Never use the words "demo", "demonstration", or "example" in code or copy.
- No comments in code unless capturing a non-obvious WHY.
