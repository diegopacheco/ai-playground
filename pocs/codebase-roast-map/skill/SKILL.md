---
name: codebase-roast-map
description: Use when a user wants a funny local pain map of a repository showing complex files, churn hotspots, stale code, ownership risk, weak tests, bug-heavy areas, and maintainability risk through /roast or /roast-map.
---

# codebase-roast-map

Use this skill to inspect a repository and produce a funny, evidence-based pain map.

## Commands

When the user asks for a repo roast, run:

```bash
node ~/.agents/skills/codebase-roast-map/scripts/roast-map.mjs report "$PWD"
```

If that path is missing, try:

```bash
node ~/.codex/skills/codebase-roast-map/scripts/roast-map.mjs report "$PWD"
```

If running under Claude Code, use:

```bash
node ~/.claude/skills/codebase-roast-map/scripts/roast-map.mjs report "$PWD"
```

When the user asks for the visual repo map, run:

```bash
node ~/.agents/skills/codebase-roast-map/scripts/roast-map.mjs map "$PWD"
```

If that path is missing, try:

```bash
node ~/.codex/skills/codebase-roast-map/scripts/roast-map.mjs map "$PWD"
```

If running under Claude Code, use:

```bash
node ~/.claude/skills/codebase-roast-map/scripts/roast-map.mjs map "$PWD"
```

## Behavior

The scanner reads the local repository only. It may generate `.roast-map` inside the target repository. It must not edit source files.

Use `/roast` for terminal output.

Use `/roast-map` for the local visual UI.

## Signals

Rank files using:

- Churn from git history
- Bug-related commit subjects
- Contributor count
- Last touched date
- Line count
- Nesting depth
- Function-like block count
- Suspicious markers
- Missing nearby tests

## Output

The map output includes:

- `.roast-map/index.html`
- `.roast-map/data.json`
- `.roast-map/summary.md`

Keep the tone funny, short, and tied to evidence.
