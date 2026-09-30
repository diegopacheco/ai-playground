---
description: Lint the target codebase and write .lint/report.json plus a history entry
argument-hint: "[path]"
---

Lint the codebase at `$ARGUMENTS` (default: the current working directory). Steps:

1. Run the deterministic engine:
   `node ~/.claude/skills/agent-skill-linter/engine/lint.mjs $ARGUMENTS`
   This builds each module, runs its tests, captures per-test timing (flagging
   tests at or above 5 seconds), computes cyclomatic complexity per function, and
   writes `<target>/.lint/deterministic.json`.

2. Read `<target>/.lint/deterministic.json` and read the source files it lists.
   Judge the semantic categories the engine cannot: naming (expressive,
   intent-revealing identifiers), principles (SOLID, DRY, KISS, separation of
   concerns), bestPractices (language idioms, validation, resource handling), and
   overall codeQuality.

3. Write `<target>/.lint/semantic.json` with two keys:
   - `scores`: integer 0-100 for naming, principles, bestPractices, codeQuality.
   - `rules`: a list, each with id, category, status (pass | warn | fail),
     severity, detail, findings (where + message), and a `samples` object with a
     `bad` and a `good` snippet illustrating the rule.

4. Merge and persist:
   `node ~/.claude/skills/agent-skill-linter/engine/assemble.mjs $ARGUMENTS`
   This writes `<target>/.lint/report.json` and appends `<target>/.lint/history/<timestamp>.json`.

5. Print the overall score, the per-category breakdown, the count of slow tests,
   and the top flagged rules.

Only lint trusted repositories: this builds and runs the target's tests.
