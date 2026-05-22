# jit-testing-skill

A Claude Code skill that catches bugs *before* they land.

It reads your diff, generates **ephemeral catching tests** designed to fail when the change introduces a regression, runs them, and tells you in plain English what behavior changed. Tests are never committed — they live in a scratch dir, run once, and get discarded.

Inspired by Meta's *Just-in-Time Catching Test Generation* (Becker et al., FSE Companion 2026): [arxiv.org/pdf/2601.22832](https://arxiv.org/pdf/2601.22832).

## Why

Most LLM test generators write *hardening* tests — tests that pass and stay in the repo to guard against future regressions. This skill writes the opposite: tests that **fail on your current change** if (and only if) the change accidentally broke something.

| | Hardening tests | Catching tests |
|---|---|---|
| When written | Anytime | At diff submission |
| Pass on diff? | Yes | No (by design) |
| Land in repo? | Yes | No, ever |
| What it catches | Future regressions | Bugs in the change *right now* |

A test that **passes on the parent revision** and **fails on your diff** is a *weak catch*. If the failure reflects a real bug (not just an intended behavior change), it becomes a *strong catch*. The skill's job is to find weak catches and rank how likely each one is to be strong.

## What it does

1. **Reads your diff** (parent ↔ child).
2. **Generates tests** using two workflows:
   - **Dodgy diff** — treat the diff as a possible mutant and look for any behavior change.
   - **Intent-aware** — infer what the diff *meant* to do, enumerate ways it could go wrong, build risk-mutants of the parent, generate tests that kill those mutants, run them against the actual diff.
3. **Runs tests** against parent (must pass) and diff (must fail).
4. **Scores each weak catch** with a rule-based assessor (RubFake) and an LLM-as-judge.
5. **Asks you a sense-check question** in plain English:
   > "On the parent, `qualifiesForFreeShipping($49.99 delivery)` returned `false`. On your change, it returns `true`. Expected?"
6. **Throws everything away** when done. Only the report survives.

## Supported targets

The skill is **generic across stacks**. The orchestrator dispatches to a per-language runner; adding a new stack means one new runner file.

| Target | Build tool | Test framework | POC status |
|---|---|---|---|
| Java 8 | maven / gradle | JUnit 5 | working |
| Java 25 | maven / gradle | JUnit 5 | working |
| Scala 3 | sbt | MUnit / ScalaTest | detect-only stub |
| Scala 2 | bazel | ScalaTest | detect-only stub |
| Kotlin | gradle (kts) | JUnit 5 | detect-only stub |
| Python 3 | pip / venv | pytest | working |
| Python 3 + Django | pip / venv | pytest-django | detect-only stub |
| Node.js | npm / pnpm | node assert | working |

## Install

```bash
./install.sh
```

Installs the binary tree to `~/.claude/jit/`, registers `/jit` and `/jit-dashboard` as Claude Code skills, and adds `.jit-testing/` to your global gitignore so generated tests can never accidentally be committed. Prints which language targets are usable on this machine.

## Uninstall

```bash
./uninstall.sh
```

Removes the skill and the gitignore entry. Optionally purges leftover `.jit-testing/` scratch dirs (prompted).

## Usage

The skill ships two slash commands.

### `/jit` — run catches on a diff

From inside any supported repo, on a branch with a diff:

```
/jit
```

Optional flags:

- `--target java25` — force a specific language target
- `--workflow intent` — run only the intent-aware workflow (`dodgy` / `intent` / `both`, default `both`)
- `--diff HEAD~3..HEAD` — pick a diff range

You'll get a ranked list of behavior changes, each with:

- A one-line sense-check question.
- A confidence score (`-1` very likely false positive, `+1` very likely real bug).
- An expandable view of the test code and execution trace.

Dismissing a false positive is one click. Real bugs surface fast.

#### Sample output

```
⏺ The pipeline found 18 catches across both workflows. Collapsing duplicates, there are really two distinct behavior changes — one
  matches the stated intent ("Add free shipping for store-pickup orders"), but the other is not covered by the intent and looks
  suspicious.

  Sense-check questions:

  1. Pickup orders now always get free shipping. On the parent, free_shipping(0, True) returned False. On your change, it returns True
  (same for 1, 100, 499, 500, 4999 when pickup=True). → This matches the stated intent. Expected?
  2. Non-pickup threshold dropped from $50 → $5. On the parent, free_shipping(500, False) returned False (parent required ≥ 5000 cents).
  On your change, it returns True (now requires ≥ 500 cents). → This is not mentioned in the intent ("Add free shipping for store-pickup
  orders"). Was the threshold change deliberate?
  3. Same threshold drop affects higher amounts too. On the parent, free_shipping(4999, False) returned False. On your change, it returns
   True. → Same root cause as #2.

  For each, pick: Confirm bug / Expected change / Show test code.

  The threshold change (catches #7 and #16) is the one I'd flag — it's an off-intent change in pricing.py:4 (>= 500 vs parent's >= 5000).

  You can open the dashboard with /jit-dashboard.
```

### `/jit-dashboard` — visual UI

```
/jit-dashboard
```

Starts a local web server (loopback only) and opens your default browser at a dashboard that visualizes every run under `.jit-testing/`. **Light theme only — always.** No dark mode, no toggle, no auto-switching even if the OS is in dark mode. No build step, no telemetry.

What you see:

- **Big counters** at the top — total runs, weak catches, strong-catch candidates, dismissed false positives, breakdown per language target.
- **Trend strip** — catches per run over time.
- **Runs table** — every `/jit-testing` execution, click to expand.
- **Catch detail** — sense-check question, score breakdown (RubFake + LLM-judge), matched FP/TP patterns as chips, test code + trace on demand, verdict buttons (*Confirm bug* / *Expected* / *Defer*).
- **Per-language report cards** — runs, catches, true-positive rate, most common patterns.

Stop the dashboard with `/jit-dashboard --stop`.

#### Screenshots

![JIT dashboard — overview](./jit-dashboard-1.png)

The landing view: top-row counters summarize every run found under `.jit-testing/` (total runs, weak catches, strong-catch candidates, dismissed false positives), the trend strip plots catches per run over time, and the runs table lists each execution sorted by recency. Per-language report cards on the right break down runs and true-positive rate by target.

![JIT dashboard — catch detail](./jit-dashboard-2.png)

Expanded catch view: the sense-check question is rendered in plain English, the score breakdown shows the RubFake rule-based assessor next to the LLM-as-judge score, matched FP/TP patterns appear as chips, and the generated test code plus execution trace are available on demand. The verdict buttons (*Confirm bug* / *Expected* / *Defer*) feed back into the per-language true-positive rate.

## Where things go

```
<repo>/.jit-testing/
└── runs/<timestamp>-<sha>/
    ├── tests/        generated tests (ephemeral)
    ├── mutants/      risk-mutants of the parent
    ├── traces/       execution logs
    ├── catches.json  ranked weak catches
    └── report.md     human-readable summary
```

Everything under `.jit-testing/` is gitignored. The dir auto-purges after 7 days (configurable).

## Sample projects

Each supported stack has a runnable sample under `samples/`. They are real, self-contained projects you can `cd` into and run `/jit` against. Each contains the current (buggy) state and a hidden `.parent/` snapshot with the pre-diff source.

```
samples/
├── java-shipping/      Java 8 + maven, the canonical example
├── python-pricing/     Python 3
├── nodejs-discount/    Node.js + CommonJS
├── scala-discount/     Scala 3 + sbt
└── kotlin-discount/    Kotlin + Gradle
```

To try one — just `cd` and run `/jit`. No git setup, no scripts.

### Run every sample (snapshot mode, no setup)

From the skill root, invoke the CLI directly against each sample. These commands are equivalent to running `/jit` from inside each sample dir.

```bash
./bin/jit run --repo samples/python-pricing
./bin/jit run --repo samples/nodejs-discount
./bin/jit run --repo samples/java-shipping
./bin/jit run --repo samples/scala-discount
./bin/jit run --repo samples/kotlin-discount
```

Or, inside each sample:

```bash
cd samples/python-pricing   && /jit && cd -
cd samples/nodejs-discount  && /jit && cd -
cd samples/java-shipping    && /jit && cd -
cd samples/scala-discount   && /jit && cd -
cd samples/kotlin-discount  && /jit && cd -
```

### Inspect what was caught

```bash
ls samples/python-pricing/.jit-testing/runs/
cat samples/python-pricing/.jit-testing/runs/*/report.md
```

### Open the dashboard against any sample

```bash
cd samples/python-pricing
./bin/jit dashboard --repo .
```

Stop it with:

```bash
./bin/jit dashboard --stop
```

### Run the full smoke test

```bash
./test.sh
```

Exercises all five samples in snapshot mode plus a git-mode regression check.

### Run a sample in git mode instead of snapshot

Each sample ships a `setup-git.sh` that materializes a real parent → diff commit history. After running it, snapshot mode is dropped (the script removes `.parent/`) and `/jit` falls back to git diff.

```bash
cd samples/java-shipping
./setup-git.sh
./bin/jit run --repo . --diff HEAD~1..HEAD --mode git
```

### Workflow and target flags

```bash
./bin/jit run --repo samples/python-pricing --workflow dodgy
./bin/jit run --repo samples/python-pricing --workflow intent
./bin/jit run --repo samples/python-pricing --workflow both
./bin/jit run --repo samples/java-shipping --target java25
./bin/jit run --repo samples/python-pricing --max-tests 50
```

### Detect-only check

```bash
./bin/jit detect samples/python-pricing
./bin/jit detect samples/nodejs-discount
./bin/jit detect samples/java-shipping
./bin/jit detect samples/scala-discount
./bin/jit detect samples/kotlin-discount
```

### Install / uninstall

```bash
./install.sh
./uninstall.sh
./uninstall.sh --purge-runs "$HOME"
```

### After install — inside Claude Code

Once `./install.sh` finishes, the two slash commands are registered. Open Claude Code in any sample directory (or any of your own projects) and type the slash command. Each sample needs nothing more than `cd` + the slash command.

```bash
cd samples/python-pricing
```
then in Claude Code:
```
/jit
```

Same flow for every sample:

```bash
cd samples/python-pricing   # then /jit  in Claude Code
cd samples/nodejs-discount  # then /jit  in Claude Code
cd samples/java-shipping    # then /jit  in Claude Code
cd samples/scala-discount   # then /jit  in Claude Code
cd samples/kotlin-discount  # then /jit  in Claude Code
```

Open the dashboard from any of them:

```
/jit-dashboard
```

Stop the dashboard:

```
/jit-dashboard --stop
```

Pass flags through the slash command exactly like the CLI:

```
/jit --target java25
/jit --workflow dodgy
/jit --workflow intent
/jit --mode git --diff HEAD~3..HEAD
/jit --max-tests 50
```

## Snapshot mode vs git mode

`/jit` picks the comparison mode automatically:

- **Snapshot mode** (default when `.parent/` exists in the cwd) — compares current files against `.parent/`. No git history needed. This is what the samples use.
- **Git mode** (default in real repos) — compares `HEAD~1..HEAD` or any range you pass via `--diff`.

Force a mode with `--mode {git|snapshot}` and override the snapshot directory with `--parent-dir <name>`.

## End-to-end test

```bash
./test.sh
```

Exercises the Python, Node.js, and Java samples in a temp dir and verifies that `/jit detect` picks the right target, the pipeline runs, and the report names the changed function.

## Design

See [`design-doc.md`](./design-doc.md) for the full design — workflows, assessors, runner contract, pipeline, and acceptance criteria.

## Reference

Becker, M. et al. (2026). *Just-in-Time Catching Test Generation at Meta*. FSE Companion '26. [arXiv:2601.22832](https://arxiv.org/pdf/2601.22832).

Key results from the paper this skill is built on:

- Diff-aware catch generation produces ~**4× more weak catches** than hardening baselines and **20×** more than coincidental catching.
- LLM-as-judge ensemble reduces human review load by **~70%** at >98% precision on filtering false positives.
- From 41 engineer reach-outs, **8 real bugs caught**, of which **4 would have caused severe production failures**.
