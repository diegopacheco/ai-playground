# bash-pipe-poc

A research repo that answers the question: **"why does bash orchestration from a Claude Code skill (running `mvn` / `sbt` / `bazel` / Django / Python) hang, run slowly, return phantom-running results, or leave us blind?"** — and proves the fix by running eight working POCs across that exact stack matrix, graded by an immutable oracle.

> Full research, rules, and architecture: **[design-doc.md](./design-doc.md)** (17 sections).

---

## The four pain points this repo addresses

The original symptoms reported when a skill called bash to drive build/test/run:

1. **Slowness** — operations take longer from a skill than by hand.
2. **No visibility** — the model has no signal during a long run.
3. **Stuck / hang** — the command never returns.
4. **Phantom-running** — the underlying process finished but the tool call still reports it as running.

## The verdict — where the problem actually lives

Not "bash is broken." Four overlapping layers (see [design-doc §2](./design-doc.md#2-the-verdict--where-the-problem-actually-lives)):

| Rank | Layer                                | Share of failures |
|------|--------------------------------------|-------------------|
| 1    | How the skill writes Bash            | ~50%              |
| 2    | Stack-specific quirks (sbt, mvn, Bazel, Django) | ~25%   |
| 3    | Claude Code Bash tool semantics      | ~20%              |
| 4    | Genuine Bash language pitfalls       | ~5%               |

Layer 3 is the source of "phantom-running": the Bash tool waits for **EOF on stdout/stderr**, and any forked child that inherits those file descriptors (Surefire JVM, sbt server, Django auto-reloader, Bazel daemon) keeps the pipe open even after the parent exits.

## The fix — two axioms and fourteen rules

Everything below rests on two axioms (see [design-doc §3](./design-doc.md#3-guiding-principles)):

- **A1 — The pipe must close.** Every long-lived child must redirect its stdio to a file and detach.
- **A2 — Everything is bounded.** No unbounded `sleep`. No "wait forever for the server".

The fourteen prescriptive rules (see [design-doc §4](./design-doc.md#4-the-guidelines-g1g14)):

| ID  | One-line summary |
|-----|------------------|
| G1  | Non-interactive flags on every tool (`mvn -B -ntp`, `sbt --batch ...`, `pip --no-input`, `pytest -q`, Django `--noinput`, `python -u`). |
| G2  | Servers start `nohup CMD > LOG 2>&1 < /dev/null & echo $! > PIDFILE`. Both the redirect AND `< /dev/null` are mandatory. |
| G3  | Stop kills children first (`pkill -TERM -P $PID`), then parent, with bounded escalation to `KILL`. |
| G4  | Readiness is a bounded poll loop with `curl --max-time 2`; never a fixed `sleep` to wait for a server. |
| G5  | Disable build-tool daemons in scripted runs: `-Dsbt.server.autostart=false`, `bazel shutdown` in stop.sh, never mvnd. |
| G6  | Pre-warm caches once in setup.sh; warm runs are the SLA target. |
| G7  | Every long-running command is bounded by a wall-clock timeout. |
| G8  | `set -euo pipefail` + `\|\| true` on greps that may not match. |
| G9  | Logs to files; on failure tail ≤ 80 lines; on success ≤ 20. |
| G10 | Every POC ships the exact same script set: `setup build test start hc stop clean` + `bash-pipe.env`. |
| G11 | Each POC exposes a real HC endpoint that returns 200; the path is discovered (`/actuator/health` for Spring, `/health/` for Django, `/health` otherwise). |
| G12 | The orchestrator emits `results.json` and `problems.md`. The judge grades them. |
| G13 | Self-lint refuses interactive tools (bare `sbt`, `pip install` without `--no-input`, etc.). |
| G14 | Cap log volume returned to the model: tail-on-failure, never full transcript. |

## Architecture — three layers

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: The Skill            (regenerated each invocation)  │
│   skill/bash-pipe-skill/SKILL.md                             │
│   Reads each POC, runs discovery, emits scripts per G1–G14.  │
└──────────────────────────────────────────────────────────────┘
                          │ emits
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Layer 2: The Orchestrator     (regenerated)                  │
│   run-all.sh + per-POC scripts                               │
│   Runs build→test→start→hc→stop for every POC, writes        │
│   results.json (machine) + problems.md (human).              │
└──────────────────────────────────────────────────────────────┘
                          │ emits
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Layer 3: The Judge        (IMMUTABLE — never modified)       │
│   judge.sh                                                   │
│   Hardcoded expectations. Reads results.json. Exits 0 iff    │
│   every expectation matched, with a structured diff on FAIL. │
└──────────────────────────────────────────────────────────────┘
```

The judge is a property test **for the skill itself**: regenerate the skill, re-run the orchestrator, run the judge. PASS means the skill is doing its job. FAIL means the skill regressed. Details: [design-doc §14–16](./design-doc.md#14-three-layer-architecture-skill--orchestrator--judge).

## The eight POCs

Each POC has its own dir, app code, three tests (`test_health_endpoint_returns_200`, `test_health_payload_shape`, `test_root_or_app_specific_smoke`), and the standard script set from G10. Together they exercise every stack the user originally listed.

| #  | Dir                       | Stack                                | Build  | Notes                                              |
|----|---------------------------|--------------------------------------|--------|----------------------------------------------------|
| 1  | `python3-plain/`          | Python 3 + stdlib `http.server`      | venv   | Pattern validator — simplest end-to-end.           |
| 2  | `django-python3/`         | Django 5 + Python 3.13               | venv   | `SimpleTestCase` to skip DB; `--noreload` runserver. |
| 3  | `java8-mvn-sb2/`          | Java 8 + Spring Boot 2.7             | Maven  | SDKMAN `corretto-8.0.462`. Tests use TestRestTemplate. |
| 4  | `java25-mvn-sb4/`         | Java 25 + Spring Boot 4              | Maven  | JDK HttpClient in tests (SB4 dropped TestRestTemplate). |
| 5  | `java25-kotlin-mvn-sb4/`  | Java 25 + Kotlin 2.3.21 + SB4        | Maven  | Kotlin Maven plugin + Spring allopen.              |
| 6  | `java25-scala3-sbt-sb4/`  | Java 25 + Scala 3 + SB4              | sbt    | Scala 3.7.3 (3.3 can't parse SB4 bytecode); ScalaTest. |
| 7  | `scala3-sbt/`             | Scala 3 + Cask (no Spring)           | sbt    | Embedded server in tests via `app.port` sys prop.  |
| 8  | `scala2-bazel/`           | Scala 2.13 + Bazel + rules_scala     | Bazel  | bzlmod + `MODULE.bazel`; stop.sh calls `bazel shutdown`. |

POC matrix and per-stack templates: [design-doc §6](./design-doc.md#6-poc-matrix).

## The output contract

**`results.json`** — machine-readable, schema_version=1 ([design-doc §15.1](./design-doc.md#151-resultsjson-schema)):

```json
{
  "stacks": {
    "<stack>": {
      "build":  { "pass": BOOL, "duration_sec": INT, "exit_code": INT },
      "tests":  { "pass": BOOL, "total": INT, "passed": INT, "failed": INT, "skipped": INT, "duration_sec": INT },
      "start":  { "pass": BOOL, "duration_sec": INT, "pid": INT },
      "hc":     { "pass": BOOL, "status_code": INT, "attempts": INT, "duration_sec": INT },
      "stop":   { "pass": BOOL, "duration_sec": INT, "orphans": INT }
    }
  },
  "summary": { "total_stacks": 8, "passed_stacks": 8, "total_tests": 24, "total_tests_passed": 24 }
}
```

The `stop.orphans` field is the empirical anti-G3-violation check: `pgrep -P $PID | wc -l` measured 2 seconds after stop.sh returns. Non-zero means the script kept descendants alive — judge fails the stack even if `stop.pass=true`.

**`problems.md`** — one H2 per failing stack with a ≤ 20-line log tail. Empty H1-only when everything passes.

## The judge — what it checks

`judge.sh` is hardcoded and immutable. For each of the 8 stacks it asserts ([design-doc §16](./design-doc.md#16-the-judge--judgesh-immutable-oracle)):

- stack present in results.json
- `build.pass == true`
- `tests.pass == true && total == 3 && failed == 0 && skipped == 0`
- `start.pass == true`
- `hc.pass == true && status_code == 200`
- `stop.pass == true && orphans == 0`

Plus run-level: `total_stacks == 8`, `passed_stacks == 8`, `wall_clock_sec <= 900`.

PASS output:
```
JUDGE: PASS — 8/8 stacks, 24/24 tests, wall=102s (cap=900s)
```

FAIL output is a structured per-field diff.

## Layout

```
design-doc.md                   the research, 17 sections
README.md                       (this file)
lib/common.sh                   shared bash functions (bp_start_app, bp_wait_hc, bp_stop_app, bp_use_java, parsers)
run-all.sh                      the generated orchestrator
judge.sh                        IMMUTABLE oracle — never modified by the skill
results.json                    orchestrator output (machine)
problems.md                     orchestrator output (human)

skill/bash-pipe-skill/SKILL.md  the generator skill
skill/bash-pipe-skill.md        slash-command pointer

python3-plain/                  8 POCs, one dir each
django-python3/
java8-mvn-sb2/
java25-mvn-sb4/
java25-kotlin-mvn-sb4/
java25-scala3-sbt-sb4/
scala3-sbt/
scala2-bazel/
```

Each POC dir contains exactly: `bash-pipe.env setup.sh build.sh test.sh start.sh hc.sh stop.sh clean.sh` plus its app source and tests. Every script sources `../lib/common.sh` and stays under ~30 lines.

## How to run

From this directory:

```
bash run-all.sh
bash judge.sh
```

Warm full sweep is ~100s on macOS + SDKMAN + Corretto. First run installs missing JDKs and downloads Maven/sbt/Bazel dependencies — set aside 5–15 minutes for cold setup.

## Warm SLA targets (per-POC)

From [design-doc §7](./design-doc.md#7-warm-sla-targets-after-setupsh):

| Stack                  | build | test | start→hc | stop |
|------------------------|-------|------|----------|------|
| java8-mvn-sb2          | 30 s  | 30 s | 20 s     | 5 s  |
| java25-mvn-sb4         | 45 s  | 45 s | 25 s     | 5 s  |
| java25-kotlin-mvn-sb4  | 60 s  | 60 s | 25 s     | 5 s  |
| java25-scala3-sbt-sb4  | 90 s  | 90 s | 30 s     | 5 s  |
| scala3-sbt             | 60 s  | 60 s | 25 s     | 5 s  |
| scala2-bazel           | 60 s  | 60 s | 25 s     | 5 s  |
| python3-plain          | 5 s   | 10 s | 10 s     | 5 s  |
| django-python3         | 10 s  | 15 s | 15 s     | 5 s  |

## Regenerating with the skill

Install once:

```
ln -s "$(pwd)/skill/bash-pipe-skill" ~/.claude/skills/bash-pipe-skill
ln -s "$(pwd)/skill/bash-pipe-skill.md" ~/.claude/commands/bash-pipe-skill.md
```

Then in Claude Code: `/bash-pipe-skill`. The skill regenerates the orchestrator and per-POC scripts, runs the orchestrator, then runs the judge. It will **never** touch `judge.sh` (the self-lint refuses any write to that filename).

The skill itself (SKILL.md) encodes G1–G14 as prescriptive output requirements, the per-stack templates from [design-doc §6](./design-doc.md#6-poc-matrix), and the discovery rules from [design-doc §11](./design-doc.md#11-skill-discovery-rules-port--hc-path). See [design-doc §10 & §12](./design-doc.md#10-skill-authoring-implications-the-skill-is-a-bash-generator) for why the rules must live in the skill's prompt and not just in the agent's head.

## Honest limits

We do NOT claim ([design-doc §5](./design-doc.md#5-honest-non-goals-and-limits)):

- That a cold `mvn` first run fits inside the 10-minute Bash tool ceiling on every machine. **G6** makes this a one-time setup cost, not per-run.
- That a misbehaving third-party JVM agent can't spawn a child that escapes process-group kill. **G12 step 5** (the `stop.orphans` counter) detects this and fails loudly.
- That the model "knows what is happening" during a long build. The Bash tool does not stream. Background mode + periodic `tail` on the log is the closest substitute.

## Current verdict

```
JUDGE: PASS — 8/8 stacks, 24/24 tests, wall=102s (cap=900s)
```

All four original failure modes are structurally prevented by G1–G14. The orphan counter at `stop.orphans=0` across every stack is the empirical confirmation that no leaked children survive any of the 8 stop.sh invocations.
