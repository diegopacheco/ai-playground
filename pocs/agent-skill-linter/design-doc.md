# Design Doc — `agent-skill-linter`

A Claude Code skill that lints a codebase against a rich, opinionated rule set
(not just a Sonar wrapper) and renders the results as a modern, searchable web
report. Ships with global install/uninstall scripts and a screenshot-rich README.

Status: draft for review. No code yet — this document defines scope, architecture,
and the data contract so implementation can start from an agreed design.

---

## 1. Goals

- A globally installed Claude Code skill exposing two commands: `/lint` and `/lint-site`.
- `/lint` analyzes a target repository and produces a single structured report
  plus an appended history entry.
- Rules go beyond a static analyzer: code quality, best practices, design
  principles, build status, test status, **per-test timing with a slow flag at
  `>= 5s`**, cyclomatic complexity, and expressive naming — among others.
- `/lint-site` renders the report as a modern, white-themed web app with global
  search/filter and at least five tabs.
- The web stack (React + Vite + Bun frontend, Java 25 + Spring Boot 4 backend)
  runs entirely in Podman, fully abstracted from the user — they only see a URL.
- A polished `README.md` with screenshots captured via Playwright.
- One-command install and uninstall against a global Claude installation.

## 2. Non-goals

- Not a CI gate or server; it runs on demand, locally.
- Not a Sonar/PMD/ESLint replacement or wrapper. We compute our own signals with
  minimal dependencies.
- No auto-fixing of code. The skill reports; it does not rewrite the target repo.
- No cloud services, accounts, or telemetry.

## 3. Architecture at a glance

```
┌──────────────────────────────────────────────────────────────┐
│ Claude Code (global ~/.claude)                                 │
│   commands: /lint   /lint-site       skill: agent-skill-linter │
└───────────────┬────────────────────────────────┬──────────────┘
                │ /lint                           │ /lint-site
                ▼                                 ▼
   ┌──────────────────────────┐    ┌───────────────────────────────┐
   │ HOST  (no containers)     │    │ PODMAN (podman-compose)        │
   │  1 language detection     │    │  ┌───────────────┐ ┌────────┐  │
   │  2 deterministic engine ──┼──▶ │  │ Java 25 /     │ │ React  │  │
   │      build / test / time  │    │  │ Spring Boot 4 │◀│ Vite   │  │
   │      / cyclomatic compl.  │    │  │ REST API      │ │ Bun    │  │
   │  3 semantic engine (Claude)│   │  └──────┬────────┘ └────────┘  │
   │        │                  │    │         │ mounts (read-only):  │
   │        ▼                  │    │         │  .lint/ + target repo │
   │  .lint/report.json        │◀───┼─────────┘                      │
   │  .lint/history/*.json     │    └───────────────────────────────┘
   └──────────────────────────┘
```

Key separation of concerns:

- **`/lint` runs on the host and needs no containers.** It is fast and produces
  data only. The deterministic signals are computed by small, dependency-light
  analyzers; the semantic signals are produced by Claude.
- **`/lint-site` is the only part that uses Podman.** The backend is purely a
  read/aggregate/serve layer over the data `/lint` already wrote. It does **not**
  rebuild or re-test the target repo, so there is no duplicated build logic and
  no heavy toolchains baked into containers.

## 4. The two commands

### 4.1 `/lint [path]`

Default target is the current working directory; an optional path overrides it.

Pipeline:

1. **Detect** languages and build tools by scanning for marker files
   (`pom.xml`, `build.gradle*`, `package.json`, `bun.lockb`, `Cargo.toml`, etc.).
   Each detected language activates an adapter.
2. **Deterministic engine** (host, via adapters):
   - Build the project and record pass/fail plus warning count.
   - Run tests and capture per-method name, class/file, duration, and status.
     Any test at or above the 5-second threshold is flagged `slow`.
   - Compute cyclomatic complexity per function/method by counting decision
     points (branches, loops, `case`, `catch`, boolean `&&`/`||`, ternaries).
     This is library-free and language-agnostic at the heuristic level — no Sonar.
   - Compute supporting metrics: function length, file length, nesting depth,
     magic-number count, comment ratio, duplication signal.
3. **Semantic engine** (Claude): judgment-based rules — expressive naming,
   adherence to principles (SOLID, DRY, KISS, YAGNI, separation of concerns),
   language-specific best practices, and an overall code-quality read. Each
   finding carries a file/line and a short rationale.
4. **Score** every category 0–100 and compute a weighted overall score (§7).
5. **Write** `.lint/report.json` and append `.lint/history/<timestamp>.json`.
6. **Summarize** in the terminal: overall score, per-category scores, build/test
   status, count of slow tests, and the top failing rules.

`/lint` produces a terminal summary and the data files. It does not open the site.

### 4.2 `/lint-site`

1. Verify `.lint/report.json` exists (instruct the user to run `/lint` first if not).
2. Bring up the Podman stack via `start.sh` (frontend + backend), waiting on
   container readiness with a poll loop, never a long sleep.
3. Mount `.lint/` and the target repo read-only into the backend so it can serve
   report data, trend aggregations across history, and source files for the code
   viewer.
4. Print the URL and open it. The user never sees Podman, Maven, or Bun details.
5. `stop.sh` tears the stack down.

## 5. Rule catalog (v1)

Rules are grouped into scored categories. Each rule has: stable `id`, category,
`type` (deterministic | semantic), severity, weight, pass/fail status, a list of
findings (`file:line` + message), and a good-vs-bad code pair for the Rules tab.

| Category       | Representative rules                                                            | Type |
|----------------|---------------------------------------------------------------------------------|------|
| Build          | Build succeeds; zero build warnings                                             | det. |
| Tests          | All tests pass; no test slower than 5s; no skipped tests                        | det. |
| Complexity     | Cyclomatic complexity per function under threshold; nesting depth under limit   | det. |
| Naming         | Expressive identifiers; no single-letter or abbreviation-only names; intent-revealing | sem. |
| Principles     | SOLID, DRY, KISS, YAGNI, separation of concerns                                 | sem. |
| Best practices | Language idioms, error handling, resource cleanup, immutability where apt       | sem. |
| Code quality   | Function/file length, dead code, magic numbers, duplication, comment ratio      | mixed|

The catalog is data-driven so rules can be added without touching the renderer.

## 6. Data model

The contract between `/lint` (producer) and `/lint-site` (consumer). Described as
fields rather than code; the implementation will pin this with a JSON schema.

**`report.json`**

| Field | Meaning |
|-------|---------|
| `meta` | skill version, schema version, timestamp, repo path, git commit, run duration |
| `languages[]` | name, file count, lines of code, detected build tool |
| `build` | status (pass/fail), tool, warning count, log reference |
| `tests` | totals (passed/failed/skipped), `slowThresholdSec` = 5, and `methods[]` |
| `tests.methods[]` | name, class/file, durationMs, status, `slow` boolean |
| `complexity` | per-function entries (file, function, line, cyclomatic, threshold, `exceeds`) and a distribution summary |
| `rules[]` | id, category, title, type, severity, weight, status, `findings[]`, `samples` (bad/good) |
| `scores` | `overall` plus a per-category breakdown |

**History:** each run is appended as `.lint/history/<timestamp>.json` (same shape,
trimmed of heavy fields). The backend reads the directory to build trend series
for Tab 3. `.lint/` is added to the target repo's ignore rules by the install flow.

## 7. Scoring model

- Each category is scored 0–100.
- Deterministic categories map measured values to scores: build fail forces
  Build to 0; each slow test and each failing test reduces Tests proportionally;
  functions over the complexity threshold reduce Complexity proportionally.
- Semantic categories are scored from the ratio of passing to total weighted
  rules in that category.
- **Overall** is a weighted average across categories. Initial weights (tunable):
  Build 20, Tests 20, Complexity 15, Principles 15, Best practices 12,
  Code quality 10, Naming 8.

## 8. The web report — five tabs

White, modern theme. A persistent global search bar at the top filters the
active tab's content (test names, rule ids/titles, file paths, finding messages).

- **Tab 1 — Dashboard.** Big overall-score gauge, per-category score cards/radials,
  build status, test pass rate, slow-test count, complexity summary, and a
  delta-vs-last-run sparkline. Highly visual at a glance.
- **Tab 2 — Tests.** Every test method as a sortable row: name, file/class,
  duration, status. A "slow only" toggle (`>= 5s`), per-file rollups, and a
  duration histogram with the slowest tests highlighted.
- **Tab 3 — Charts & trends.** Time series from history: overall score over time,
  slow-test count over time, complexity distribution, test count, and rule
  pass-rate trend.
- **Tab 4 — Rules.** All rules grouped by category with pass/fail badges and
  violation counts. Expanding a rule reveals its findings (`file:line`) and a
  good-vs-bad code pair illustrating the rule.
- **Tab 5 — Code viewer.** A file tree on the left; the selected file rendered
  with line numbers and syntax coloring, with inline markers where rules fired.
  File-name search filters the tree.

## 9. Tech stack and containerization

- **Frontend:** React + Vite, built and served with Bun. Charts and syntax
  highlighting lean on hand-rolled SVG and a minimal highlighter to honor the
  "fewest libraries" constraint; this is a tradeoff against richer off-the-shelf
  widgets and is called out in Open Questions.
- **Backend:** Java 25 + Spring Boot 4. REST endpoints for scores, tests, trends
  (history aggregation), rules, the file tree, and raw source for the viewer.
- **Podman only** (no Docker): `Containerfile` per service, `podman-compose.yml`
  wiring frontend → backend, plus `start.sh`, `stop.sh`, `test.sh`. Bash scripts
  carry no comments or emojis, use readiness poll loops with at most a 1-second
  sleep, and never wait longer than a minute.
- The target repo and `.lint/` are mounted read-only into the backend.

## 10. README and Playwright screenshots

The README is authored once against a seed run. With the site up via `/lint-site`,
Claude drives the Playwright browser to navigate each of the five tabs (plus a
search interaction), captures PNGs into `docs/screenshots/`, and the README
embeds them with a short walkthrough of each tab, install/uninstall steps, and
the rule catalog. Screenshots are regenerated whenever the UI changes materially.

## 11. Install / uninstall

- `install.sh`: verifies Podman is present, copies the skill into
  `~/.claude/skills/agent-skill-linter/`, installs the two command files into the
  global commands directory, and is idempotent. No comments, no emojis.
- `uninstall.sh`: removes the skill directory and the two command files.

## 12. Proposed layout

```
agent-skill-linter/
  SKILL.md
  commands/            lint.md, lint-site.md
  install.sh  uninstall.sh
  README.md
  engine/              detection + dependency-light analyzers
    adapters/          java, node/bun, ... (build, test, timing, complexity)
  schema/              report.schema.json
  site/
    backend/           Java 25 + Spring Boot 4
    frontend/          React + Vite + Bun
    Containerfile.backend  Containerfile.frontend
    podman-compose.yml start.sh stop.sh test.sh
  docs/screenshots/
```

## 13. Security and safety

- Building and testing a target repo executes that repo's code. Only point
  `/lint` at trusted repositories. This will be stated plainly in the README.
- Containers mount the repo read-only and expose ports on loopback only.

## 14. Assumptions

- The target repo builds and tests with standard tooling for its detected stack.
- `~/.claude/` is the global install location and is writable.
- Podman and podman-compose are installed and runnable by the user.
- v1 ships Java (Maven/Gradle) and JS/TS (npm/Bun) adapters; the adapter
  interface is the extension point for more languages later.

## 15. Open questions / decisions for later

- Which languages beyond Java and JS/TS are in scope for v1 vs later?
- Charting and syntax highlighting: fully hand-rolled (fewest deps) vs a small
  library for a richer look — where exactly to draw the line.
- Cyclomatic complexity and slow-test thresholds: confirm defaults (CC limit,
  the 5s slow flag) and whether they should be configurable per run.
- Should `/lint-site` auto-run `/lint` when no report exists, or stay strict?
- History retention: keep all runs or cap/rotate the history directory?

## 16. Suggested build order

1. Report schema + scoring model (the contract everything depends on).
2. `/lint` host pipeline: detection → deterministic engine → semantic engine →
   report + history, with a terminal summary.
3. Backend API over report + history.
4. Frontend shell, theme, global search, then the five tabs.
5. Podman compose + start/stop/test scripts.
6. install.sh / uninstall.sh.
7. Playwright screenshot capture + README.
```
