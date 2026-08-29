---
name: bug-triage
description: Triages one bug against the real codebase and renders a self-contained light-theme HTML report in a temp folder covering what the bug is, the files involved, a test that reproduces it, why it is bad, the minimal fix, the files to touch, whether it breaks the DB or a public contract, and whether the fix is safe. Use when the user runs /bug-triage or asks to triage, diagnose, investigate, or root-cause a bug from a branch, a Jira or Linear issue, a GitHub issue, a bug id, or a plain description.
allowed-tools: [Bash, Read, Grep, Glob, WebFetch, AskUserQuestion]
---

# Bug Triage

When invoked, you take one bug from whatever the user gives you, read the actual code to find its cause, and write a light-theme report to a temp folder. Every claim in the report comes from a file you read. You do not fix the bug — you triage it.

## Global Context
- User request / scope: $ARGUMENTS — empty, a branch name, a git or issue URL, a Jira/Linear/GitHub issue id, or a plain description of the bug
- Renderer: `scripts/render.mjs` (Node, no dependencies)
- Template: `assets/template.html`
- Output: `$TMPDIR/bug-triage-<slug>-<timestamp>/index.html` plus the `triage.json` it was rendered from

## Rules
- Read the code before you write anything. Never describe a file you have not opened.
- Every file path in the report is absolute.
- The reproduction test must run against the real code as it is today and fail because of this bug. No pseudo-code.
- The fix is the smallest change that removes the cause. No refactors, no cleanups, no adjacent improvements.
- If you cannot find the cause, say so in the report instead of guessing. A wrong triage costs more than an honest gap.
- Do not modify the project. The only things you write are the report folder and, if the user asks, a scratch copy of the test.
- Do not add comments to any command or file you produce.

## Step 1 — Work out what the bug is
Read `$ARGUMENTS` and pick the case:

- **Empty** — triage the branch you are on. Run `git status -sb`, `git log --oneline -15`, and `git diff main...HEAD` to find what changed, then ask the user which bug they mean if the branch shows more than one candidate.
- **A branch name** — `git diff main...<branch>` and `git log --oneline main..<branch>` to read what that branch changed.
- **A Jira / Linear / GitHub URL** — fetch it with WebFetch and pull out the title, the description, and the reproduction steps. If the fetch is blocked by a login, say so in one line and ask the user to paste the bug text. Keep the URL either way — it goes in the report as `bug.url`.
- **A bug id with no URL** (`PROJ-1234`, `ENG-88`) — search the repo for it (`git log --grep`, `grep -r`) and ask the user for the link or the description if nothing turns up.
- **A plain description** — that is the bug. Use it as written.

## Step 2 — Find the cause in the code
Search the codebase for the symptom: the error text, the endpoint, the function, the field. Follow the call path from the entry point to the line that is actually wrong. Read the tests around it — an existing test that passes while the bug is live tells you what the code believes it is doing.

You need to end this step with: the exact file and line where the behavior goes wrong, and the reason it goes wrong.

## Step 3 — Write the reproduction test
Write a test in the project's own test framework (read `package.json`, `pom.xml`, `build.gradle`, `pyproject.toml`, `Cargo.toml` to find it). It must call the real code and fail today for this bug. Run it if you can, and use the real failure output as `repro.expectation`.

## Step 4 — Build the triage json
Write `triage.json` in the scratch/temp folder. Shape:

```json
{
  "bug": {
    "name": "short factual title",
    "id": "PROJ-1234",
    "url": "https://company.atlassian.net/browse/PROJ-1234",
    "tracker": "Jira",
    "severity": "critical|high|medium|low",
    "branch": "fix/checkout-total",
    "repo": "name of the repo"
  },
  "description": ["line 1", "line 2", "line 3"],
  "files": [{ "path": "/abs/path/File.java", "lines": "88-104", "role": "one line on why it is involved" }],
  "repro": {
    "path": "/abs/path/to/test file",
    "run": "npm test -- cart.test.js",
    "language": "javascript",
    "start_line": 1,
    "code": "the full test",
    "expectation": "the failure it produces today"
  },
  "why_bad": ["impact on users", "impact on data", "impact on the team"],
  "solution": {
    "summary": "the smallest change that removes the cause",
    "language": "diff",
    "code": "unified diff or the replacement lines",
    "notes": ["what you deliberately did not change"]
  },
  "files_to_touch": [{ "path": "/abs/path/File.java", "change": "what changes there" }],
  "breaking": {
    "verdict": "No | Yes | Yes, for consumers of X",
    "db": false,
    "api": false,
    "consumers": false,
    "detail": "why, in plain words"
  },
  "safety": {
    "verdict": "Safe | Safe with a caveat | Risky",
    "detail": "why",
    "risks": ["what could still go wrong"],
    "verification": ["how to prove the fix worked"]
  }
}
```

Field rules:
- `description` — 3 to 5 short lines, direct-skill style. First line says what breaks. No metaphors, no filler, no restating the title.
- `files` — every file involved, full absolute path, with the line range when you know it.
- `why_bad` — real consequences you can point at: wrong money, lost data, a 500, a security hole, a silent corruption. Not "it is bad practice".
- `breaking` — set `db` true only for a schema or migration change, `api` true only for a change to a public endpoint, payload, or exported signature, `consumers` true only when a caller outside this repo has to change.
- `safety` — "Safe" means: no schema change, no contract change, covered by the reproduction test, and reversible by a revert. Anything else is a caveat you name.
- Leave `url` out when there is no tracker link. The report then shows the bug description as the source.

## Step 5 — Render
```bash
node "$HOME/.claude/skills/bug-triage/scripts/render.mjs" <triage.json>
```

It validates the required fields, writes `index.html` and `triage.json` into a fresh temp folder, and prints `REPORT <path>`. Pass a second argument to choose the folder. If it exits with a missing-field error, fill the field in and run it again — never edit the HTML by hand.

## Step 6 — Report back
Give the user, in plain words: what the bug is, where it is (absolute path and line), the one-line fix, whether anything breaks, and whether it is safe. Then the report path exactly as the renderer printed it, and offer to open it:

```bash
open <path>/index.html
```
