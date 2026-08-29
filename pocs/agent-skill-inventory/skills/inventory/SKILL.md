---
name: inventory
description: Scans a codebase three times and produces one self-contained light-theme searchable HTML report with seven tabs - architecture overview with a hand-drawn diagram, core modules with pros and cons, database schema, committers, tech debt, observability and tests. Use when the user runs /inventory or asks for a codebase inventory, an architecture overview, a tech-debt map, a module breakdown, or a full audit of what a repository contains.
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# Codebase Inventory

Three scan passes over a codebase, then one HTML report. The passes are not
optional and not merged: each has a different job and the third one throws
work away.

Argument: an optional path to scan. Default is the current directory.

## Step 0 — set up

```bash
SKILL_DIR="$(dirname "$0")"                    # the directory holding this SKILL.md
TARGET="$(cd "${1:-.}" && pwd)"
REPORT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/inventory-$(basename "$TARGET")-XXXXXX")"
python3 "$SKILL_DIR/scripts/scan.py" "$TARGET" "$REPORT_DIR/facts.json"
```

Resolve `SKILL_DIR` for real; it is where `scripts/`, `assets/` and `prompts/`
live. Print `REPORT_DIR` to the user now — it is where the report will land and
it is also printed at the top of the report itself.

## Step 1 — pass 1, collect

Read `prompts/pass-1-collect.md` and follow it. Substitute
`$facts_json` = `$REPORT_DIR/facts.json`, `$repo_root` = `$TARGET`,
`$report_dir` = `$REPORT_DIR`. The output contract is `prompts/schema.md`.

Read real files. `facts.json` tells you where to look; it does not tell you
what the code means.

## Step 2 — pass 2, verify

Read `prompts/pass-2-verify.md` and follow it. Every path is checked against
disk, every number against `facts.json`. Corrections are recorded in
`verification.notes`.

## Step 3 — pass 3, adversarial

Read `prompts/pass-3-adversarial.md` and follow it. Cut anything unproven,
resolve contradictions between tabs, enforce the writing rules, rank and cap.
Set `verification.passes` to 3.

## Step 4 — render

```bash
python3 "$SKILL_DIR/scripts/render.py" \
  "$REPORT_DIR/facts.json" "$REPORT_DIR/analysis.json" "$REPORT_DIR"
```

The renderer validates before it writes. When it exits non-zero it prints the
exact problems: a path that does not exist, a module without 5 pros and 5 cons,
a count that disagrees with the scan, an edge to a node that is not there. Fix
`analysis.json` and run it again.

Never edit `render.py` or `template.html` to make a report pass. The validator
is the point.

## Step 5 — hand it over

```bash
open "$REPORT_DIR/index.html"      # macOS
```

Tell the user the report directory path and give a short summary: module count,
the top three tech-debt items, the test picture, and the one thing you would
fix first.

## What goes in each tab

1. **Architecture** — hand-drawn diagram from `architecture.nodes` and `.edges`,
   plus the evidence table. Nodes are real things in the repo. Layout is
   generated, so keep node labels short and set `layer` when you know the flow.
2. **Modules** — a card per core module with an emoji icon; the modal holds the
   3-5 line description, the main files with full paths, 5 pros and 5 cons
   where each con carries a 2-line reason and a 3-line fix.
3. **Schema** — only when a schema was found. Table count, main tables, size,
   worst queries with the reason and fix, 5 pros and 5 cons.
4. **Committers** — avatar and name, ordered by commit count, with what each
   person works on. Everything clickable.
5. **Tech debt** — up to 10 items per module, searchable cards; the modal holds
   why it is bad, how to fix it, the anti-patterns behind it, and real examples
   with file path and line.
6. **Observability** — a score for logging and a score for defensive code,
   with findings, plus the dashboards and alerts found in the repository.
7. **Tests** — files and cases per test type, what is covered as x of y, and
   the top 10 problems with the test suite itself.

## Rules

* No metaphors anywhere in the report. Say the mechanism.
* Every path in the report must exist. The validator enforces this.
* Never invent a number. Counts come from `facts.json`.
* When a section has no data, say so; do not fill it with generic advice.
* The report is one file. Do not add assets next to it.
