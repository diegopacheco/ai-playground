# JIT Testing Skill — Design Document

## 1. Goal

A Claude Code skill that, on a code change (diff), generates **catching tests** designed to fail when the diff introduces a regression, runs them, and reports candidate bugs to the developer. Tests are **ephemeral** — written to a scratch location, executed, results harvested, then discarded. They are never committed to the repository.

The approach is based on Meta's *Just-in-Time Catching Test Generation* paper (Becker et al., FSE Companion 2026, arXiv:2601.22832v1).

## 2. Core Concepts (from the paper)

- **Hardening test**: passes at generation time, lands in the repo to prevent future regression. Not what we do.
- **Catching test**: fails at generation time by design, to expose a bug in the proposed change.
- **Weak catch**: a test that passes on the parent revision and fails on the diff.
- **Strong catch**: a weak catch whose failure reveals a real semantic bug (true positive), not just an intended behavior change.
- **General oracle** vs **implicit oracle**: implicit catches only crashes; we need a general oracle (intended behavior) to decide if a behavior change is wrong.

The skill must generate weak catches, then reduce false positives so the developer is not drowned in noise.

## 3. Workflows

We implement both workflows described in the paper:

### 3.1 Dodgy Diff Workflow (intent-unaware)

Treat the diff itself as if it were a mutant of its parent. Ask the LLM to generate tests that distinguish parent behavior from diff behavior. High recall, more false positives.

```
parent code + diff code
        ↓
LLM: generate tests that pass on parent
        ↓
run on parent  → must PASS  (discard otherwise)
        ↓
run on diff    → must FAIL  (otherwise discard)
        ↓
weak catches
```

### 3.2 Intent-Aware Workflow

Infer the diff's intent from code + diff title/summary/commit message. Enumerate risks (ways the intent could be implemented wrongly). Build mutants of the parent representing each risk. Generate tests that kill each mutant. Run those tests against the actual diff.

```
diff (code + title + summary)
        ↓
LLM: infer intent
        ↓
LLM: enumerate risks
        ↓
LLM: build mutants of parent capturing each risk
        ↓
keep mutants that build & pass existing tests
        ↓
LLM: generate tests that pass on parent, fail on mutant
        ↓
run those tests on diff → keep failures = weak catches
```

The paper shows intent-aware doubles the catch-per-diff rate over dodgy diff.

## 4. Worked Example (Java) — the user-provided shipping policy case

The skill must be able to handle scenarios like:

- Parent: `qualifiesForFreeShipping` returns `total >= 5000`.
- Diff title: *"Add free shipping for store-pickup orders"*.
- Submitted diff: introduces pickup branch but accidentally drops threshold to `500`.
- Intent inferred: *"add free shipping for pickup orders only"*.
- Risk inferred: *"threshold accidentally lowered for non-pickup orders"*.
- Mutant: parent with threshold = 500.
- Generated catching test: delivery order at $49.99 must not get free shipping.
- Behavior: PASS on parent, FAIL on mutant, FAIL on diff → weak catch surfaced to engineer.

This case is the acceptance fixture for the Java targets.

## 5. Assessors (false-positive reduction)

Each weak catch is scored in `[-1.0, 1.0]`. Three assessors run, results combined into a ranked list. Only catches above threshold are surfaced.

### 5.1 RubFake (rule-based)

Pattern-matches false-positive smells from the execution trace and test code. Patterns from paper Table 2:

| Pattern | Source | Effect |
|---|---|---|
| broken_test_runner | execution log | high FP |
| reflection used | trace + code | high FP |
| type mismatch | trace + code | high FP |
| mock_broken / bad_mock_smell | trace | high FP |
| should_be_private / must_be_protected | trace + code | high FP |
| not_implemented_exception | trace | high FP |
| key_value_pair_change | trace | high FP |
| undefined_variable | trace | high FP |
| expecting_particular_calls | trace + code | high FP |
| web_server_down | trace | high FP |
| flakiness markers | trace + code | low FP |

True-positive patterns (paper Table 3):

| Pattern | Signal |
|---|---|
| unexpected_key_change | key out of bounds, keys not touched by diff |
| empty_container | container empty exception, container not touched |
| create_failure | object construction exception, structure not touched |
| changed_bool | asserted true ↔ false, defining expression not touched |
| refactor_violation | intent was refactor but behavior changed |
| dead_code_removal_violation | intent was removal but behavior changed |
| null_value | value became null, defining expression not touched |
| monotonic_change | intent was additive, existing behavior changed |
| rbac_change | access control flip on a role the diff did not target |

### 5.2 LLM-as-Judge

Send the LLM: test code, execution trace, parent/diff, inferred intent. Ask: *"Is this failure an unexpected bug or expected behavior change?"* Get back classification + rationale.

### 5.3 Final score

Combine RubFake score + LLM-judge score. Discard `score < threshold_low`. Surface `score > threshold_high`. Anything in between → present to user with both rationales.

## 6. User Interaction Model

The paper's key finding: do **not** show the developer a test case first. Show a **sense-check message** in plain English describing the behavior change, e.g.:

> "On the parent, `qualifiesForFreeShipping(deliveryOrder=$49.99)` returned `false`. On your change, it returns `true`. Is this expected?"

If the developer says "yes, expected" → discard. If "no, that's wrong" or "let me look" → show the test code and the trace.

Dismissal must take seconds, not minutes.

## 7. Ephemeral Test Storage

Tests are **never** committed. They live in:

```
<repo-root>/.jit-testing/
├── runs/
│   └── <timestamp>-<diff-sha>/
│       ├── tests/             generated test files
│       ├── mutants/           generated mutants (intent-aware only)
│       ├── traces/            execution logs
│       ├── catches.json       ranked weak catches with scores
│       └── report.md          human-readable summary
└── .gitignore                 contains "*"
```

The skill writes a top-level `.gitignore` rule for `.jit-testing/` during install. After report generation, the run dir may be archived or purged depending on config.

## 8. Language Targets

The skill is **generic across stacks**. Each target is a self-contained **runner** under `lib/runners/<target>.py` exposing the same contract:

```
runner.dodgy_diff(repo, diff, run_dir, max_tests)   → list[catch]
runner.intent_aware(repo, diff, run_dir, intent, max_tests) → list[catch]
```

The orchestrator does not care about the language; it only sees the catch contract. Adding a new stack means writing one runner + adding it to the registry.

Targets:

| Target | Build tool | Test framework | Detection signal |
|---|---|---|---|
| `java8` | maven / gradle | JUnit 4 | `pom.xml` / `build.gradle` + source 1.8 or 8 |
| `java25` | maven / gradle | JUnit 5 | source / release 25 |
| `scala3-sbt` | sbt | MUnit / ScalaTest | `build.sbt` |
| `scala2-bazel` | bazel | ScalaTest | `WORKSPACE` / `MODULE.bazel` with `scala_*` rules |
| `kotlin` | gradle (kts) | JUnit 5 | `build.gradle.kts` with `kotlin` plugin or `*.kt` files |
| `python3` | venv + pip | pytest | `pyproject.toml` / `requirements.txt`, no Django |
| `python3-django` | venv + pip | pytest-django | `manage.py` + `django` in deps |
| `nodejs` | npm / pnpm | node assert / vitest | `package.json` |

A repo can match multiple targets; the user (or auto-detection) picks one per run.

### Sample projects

The repo ships sample projects under `samples/<name>/` — one per supported stack — that the user can run `/jit` against. Each sample is a real, self-contained project (with its own build file) that contains:

- the current (buggy) source at the project root, and
- a `.parent/` directory holding the pre-diff version of the same files, and
- a `setup-git.sh` script that initializes a git history (commit parent → commit diff) so `/jit` has a real diff to inspect.

Current samples:

- `samples/java-shipping/` — Java 8 + maven, the canonical shipping policy example.
- `samples/python-pricing/` — Python 3, same bug in pure Python.
- `samples/nodejs-discount/` — Node.js + CommonJS, same bug in JavaScript.
- `samples/scala-discount/` — Scala 3 + sbt (detect-only).
- `samples/kotlin-discount/` — Kotlin + Gradle (detect-only).

Sample projects are **not** fixtures used internally by the runners. They are stand-alone projects the user navigates into and exercises the skill against.

## 8.1 Comparison Modes — Git vs Snapshot

`/jit` needs two source trees to compare: the **parent** (pre-change) and the **head** (current). It can get them two ways:

### Git mode (default in real repos)

Reads parent and head from git history. The diff range is `HEAD~1..HEAD` by default; the user can pass `--diff <range>` for anything git understands.

### Snapshot mode (default in directories shipping a `.parent/` tree)

Reads parent from a `.parent/` directory at the repo root, head from the working tree. No git history required. Use cases:

- **Sample projects** shipped inside this repo or another parent repo, where the sample itself has no git history.
- **Pre-staged comparisons** — a user drops the previous version into `.parent/` and runs `/jit` against the current working tree.
- **Diff snapshots from outside git** — patches arriving from code review tools that produce two trees rather than commits.

`.parent/` may optionally contain an `INTENT` file with one paragraph describing the diff intent. The first line becomes the diff title for the intent-aware workflow.

### Auto-detection

The pipeline picks the mode automatically:

- If `<repo>/.parent/` exists → snapshot mode.
- Otherwise → git mode.

The user can force a mode with `--mode {git|snapshot}` and override the parent directory with `--parent-dir <name>`.

The runner contract is mode-agnostic — every runner uses the same helpers in `BaseRunner` that internally dispatch to git or snapshot reads. Adding a new stack does not require thinking about modes.

## 8.2 Slash Commands

The skill exposes two slash commands. Naming is final:

- **`/jit`** — run the catching pipeline on the current diff.
- **`/jit-dashboard`** — open the local UI.

The pipeline binary that backs both commands lives at `~/.claude/jit/bin/jit`. Subcommands: `detect`, `run`, `verdict`, `dashboard`.

## 9. Install / Uninstall

Two scripts in the skill root:

### `install.sh`

- Installs the binary tree to `~/.claude/jit/`.
- Installs the two skill prompts to `~/.claude/skills/jit/SKILL.md` and `~/.claude/skills/jit-dashboard/SKILL.md`, registering `/jit` and `/jit-dashboard`.
- Verifies each runner's external dependencies (`java`, `javac`, `mvn`, `gradle`, `kotlinc`, `sbt`, `bazel`, `python3`, `node`) and prints which targets are usable on this machine.
- Adds `.jit-testing/` to global gitignore (`~/.config/git/ignore`) so it is never committed accidentally.
- Does **not** modify any project repos.

### `uninstall.sh`

- Removes `~/.claude/jit/` and both skill registrations.
- Removes the global gitignore entry it added.
- Optionally (`--purge-runs <root>`) purges leftover `.jit-testing/` dirs under a user-specified root.

Both scripts are pure bash, no dependencies beyond coreutils + git.

## 10. Skill Entry Point

The user invokes the skill on a diff with `/jit`. Inputs:

- Working directory (repo root).
- A diff specifier: `HEAD`, `HEAD~1..HEAD`, a branch name, or a staged set.
- Optional: target language override.
- Optional: workflow choice (`dodgy`, `intent`, `both`; default `both`).

Output:

- `report.md` printed to the user with ranked catches.
- For each catch above threshold: sense-check message → optional test code drill-down.

## 10.1 Second Command — `/jit-dashboard`

Alongside `/jit-testing` (which runs catches on a diff), the skill exposes a second slash command: `/jit-dashboard`. It opens a local web UI that visualizes everything sitting under `.jit-testing/runs/`.

### Behavior

- Starts a local HTTP server bound to `127.0.0.1` on a free port.
- Opens the user's default browser at that URL.
- Serves a single-page app that reads from `.jit-testing/runs/*/catches.json` and `report.md`.
- Server stops when the user runs `/jit-dashboard --stop` or closes the terminal.

### Look & feel

- **Light theme only. Never dark.** This is a hard requirement, not a default.
  - No dark-mode toggle, no CSS dark variant, no `prefers-color-scheme: dark` branch.
  - Even if the OS reports dark mode, the dashboard renders light.
- Light OS palette: white / off-white background, dark text, subtle borders.
- Sans-serif, generous whitespace.
- Card-based layout. No heavyweight UI framework — vanilla HTML + CSS + a small JS file. No build step.

### Sections (top to bottom)

1. **Header counters** — large numeric tiles:
   - Total runs
   - Total weak catches
   - Strong-catch candidates (score above surface threshold)
   - Dismissed false positives
   - Catches per language target (mini breakdown)
2. **Trend strip** — sparkline of catches per run over the last N runs.
3. **Runs table** — one row per run: timestamp, diff SHA, target language, workflow, # tests generated, # weak catches, # surfaced, status (open / reviewed / dismissed). Click a row to expand.
4. **Catch detail panel** (on row expand):
   - The sense-check question.
   - Score breakdown: RubFake score, LLM-judge score, combined.
   - Matched FP patterns and matched TP patterns (chips).
   - Collapsible test code + execution trace.
   - Buttons: *Confirm bug* / *Expected change* / *Defer* — writes back to `catches.json`.
5. **Per-language report cards** — for each target, a card with: runs, catches, true-positive rate (from confirmed buttons), most common FP pattern, most common TP pattern.

### Data flow

- The UI is read-mostly. Writes are confined to user verdicts on individual catches, which the server persists back into the same `catches.json` under `verdict: confirmed | expected | deferred` and a `verdict_ts`.
- No telemetry, no external network calls.

### Out of scope for the dashboard

- Editing or re-running tests from the UI (use `/jit-testing` for that).
- Multi-repo aggregation.
- Auth — it's local-only and bound to loopback.

## 11. Pipeline (end-to-end)

```
1. Detect target language        (lib/detect.py)
2. Pick comparison mode           (auto: .parent/ exists → snapshot, else git)
3. Resolve parent + head          (git history OR .parent/ snapshot, §8.1)
4. Build parent and head          (runner.build, where applicable)
5. Run chosen workflow(s)         (§3)
6. Filter: keep tests that pass on parent, fail on head
7. Score each with RubFake + LLM-judge
8. Rank, threshold, group
9. Emit sense-check report
10. Persist artifacts under .jit-testing/runs/<id>/
```

## 12. Configuration

`jit-testing.config.yaml` at repo root (optional, all keys have defaults):

```yaml
workflow: both              # dodgy | intent | both
target: auto                # auto | java8 | java25 | scala3-sbt | kotlin | nodejs | ...
mode: auto                  # auto | git | snapshot
parent_dir: .parent         # used only in snapshot mode
max_tests_per_diff: 20
score_threshold_surface: 0.3
score_threshold_discard: -0.3
purge_runs_after_days: 7
llm_judge_models: [primary]
```

## 13. Out of Scope (v1)

- Multi-language diffs (a diff touching both Python and Java) — handle one language per run.
- Concurrency / parallel execution of mutants.
- Persistent training of the RubFake rule set from past runs.
- Integration with CI — skill runs locally on demand only.
- Production-style targeting (Diff Risk Score). Every diff is a candidate.
- Dashboard: remote access, multi-repo aggregation, in-UI test editing.
- Dark mode for the dashboard — **explicitly forbidden**, not just deferred. Light theme is the only theme, period.

## 14. Acceptance Criteria

For each supported target, the skill must:

1. Detect a sample repo as the correct target.
2. Build parent + diff successfully (where applicable).
3. Generate at least one syntactically valid test under at least one workflow.
4. Correctly identify a planted bug as a weak catch.
5. Correctly classify an intended behavior change (parent → diff with no bug) as NOT a catch, via the assessors.
6. Leave the repository working tree clean (no committed artifacts).
7. Work in **snapshot mode** with no git history — running `/jit` inside a sample directory with a `.parent/` snapshot must succeed without any setup script.
8. Work in **git mode** on real repos with regular commit history.

The Java shipping-policy example from §4 is the canonical scenario; equivalent sample projects exist for each language under `samples/`.

### POC scope

For the initial POC, **`java8`**, **`python3`** and **`nodejs`** runners are fully working (behavior-diff probing via input matrices). **`java25`** inherits from `java8`. **`scala3-sbt`**, **`scala2-bazel`**, **`kotlin`** and **`python3-django`** are detection-only stubs. The architecture supports filling them in without changes to the orchestrator.

For `/jit-dashboard`:

7. After at least one `/jit-testing` run, `/jit-dashboard` opens a browser tab showing non-zero counters.
8. Dashboard renders in light theme on macOS, Linux and Windows default browsers, **including when the OS is set to dark mode** — the dashboard must still be light.
9. Verdict buttons persist back to `catches.json` and the change survives a server restart.
