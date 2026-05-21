# Design Doc — Reliable Build/Test/Run Bash for Claude Code Skills

Owner: diego
Date: 2026-05-20
Status: DRAFT — research + guidelines, no code yet

---

## 1. Problem Statement

When a Claude Code skill drives a build/test/run cycle for stacks like Maven, sbt, Django, or plain Python via Bash, the user observes four recurring failure modes:

1. **Slowness** — operations take much longer than running them by hand.
2. **No visibility** — the model has no signal about what is happening mid-run.
3. **Stuck / hang** — the command never returns.
4. **Phantom-running** — the underlying process is done, but the tool call still reports it as running or pending.

The user wants:

- A clear verdict on **what the hell the problem actually is**.
- A set of guidelines that, if followed, make these failures structurally impossible (or at least eliminate every common class).
- A POC matrix across Java 8, Java 25, sbt+Scala 3, Bazel+Scala 2, Python 3, Django+Python 3 that empirically proves the guidelines work for `build`, `test`, `run`, and `hc` (health check).

---

## 2. The Verdict — Where the Problem Actually Lives

It is **not "bash is broken"**. It is **four overlapping layers**, each contributing failures. In order of how often each is the real culprit in skill-driven runs:

| Rank | Layer                            | Share of failures (estimated) |
|------|----------------------------------|-------------------------------|
| 1    | How the skill writes Bash        | ~50%                          |
| 2    | Stack-specific quirks (sbt, mvn, Django) | ~25%                  |
| 3    | Claude Code Bash tool semantics  | ~20%                          |
| 4    | Genuine Bash language pitfalls   | ~5%                           |

### 2.1 The Claude Code Bash tool semantics (layer 3)

These are not bugs — they are the shape of the tool. Skills must be written knowing them:

- **2 min default timeout, 10 min ceiling.** A cold Maven repo download, a first-time sbt resolve, or a Django pip install can blow past this.
- **stdout/stderr are captured, not streamed.** The model sees nothing until the command returns. A 6-minute mvn build looks identical to a hang.
- **EOF-on-pipe is the completion signal.** The tool waits for the shell's stdout/stderr pipes to close. **Any forked child that inherits those FDs keeps the pipe open even after the parent exits.** This is the #1 cause of "process is done but it's still pending". Surefire-forked JVMs and detached sbt servers are notorious for this.
- **Background mode tracks the shell PID, not the process group.** Children that double-fork or `setsid` themselves escape the tracker, and the shell looks "done" while orphans linger.
- **No PTY.** Tools that detect a non-TTY behave differently: sbt switches output mode, mvn drops ANSI, pip suppresses progress bars (good), some tools prompt for input on stdin (bad).

### 2.2 Stack-specific quirks (layer 2)

- **Maven**: cold dependency download dominates first run; Surefire forks JVMs that inherit stdout; ANSI/progress noise is huge without `-B -ntp`.
- **sbt**: bare `sbt` opens an **interactive REPL** — it will hang forever waiting for stdin. Requires `--batch`. Also runs a long-lived **sbt server** that can outlive the shell. Cold first-run downloads Scala, Coursier cache, all plugins.
- **Bazel**: spawns a long-lived **Bazel server** per workspace. `bazel build` returns, but the daemon stays alive holding sockets and FDs. Without `--noblock_for_lock` and proper shutdown, a second invocation can stall on the workspace lock.
- **Python/Django**: `manage.py runserver` is a server — never returns. Auto-reloader spawns a child process; killing the parent does not always kill the child. `pip` can prompt with `--require-virtualenv` misconfig.
- **JDK 25**: not pre-installed on most machines; SDKMAN/Homebrew install can take minutes; skill assumes `java -version` works.

### 2.3 How skills typically mis-write the Bash (layer 1)

Recurring anti-patterns observed across skill outputs:

- `mvn clean install` instead of `mvn -B -ntp clean install`.
- `sbt test` instead of `sbt --batch --no-colors "test"`.
- `python manage.py runserver &` with no stdout redirect, no `nohup`, no PID file.
- No readiness probe — assumes "started" === "ready".
- `sleep 10 && curl ...` instead of a bounded poll loop.
- Killing by PID instead of process group, leaving JVM children alive.
- Massive log dumps fed back into Claude, blowing context for no signal.
- No `set -euo pipefail`, so first failure is silently passed over.

### 2.4 Genuine Bash pitfalls (layer 4)

- `cmd | tee log` masks `cmd`'s exit code without `pipefail`.
- Unquoted vars, glob expansion, word splitting.
- `trap` not installed → no cleanup on script abort.

---

## 3. Guiding Principles

Two non-negotiable axioms behind every guideline below:

- **A1 — The pipe must close.** A command is "done" only when nothing holds its stdout/stderr open. Every long-lived child must redirect its stdio to a file and detach.
- **A2 — Everything is bounded.** Every wait, retry, build, and health check has a hard cap. No unbounded `sleep`. No "wait forever for the server".

---

## 4. The Guidelines (G1–G14)

These are the rules the skill generator MUST follow. The POCs in Section 6 will prove each one.

### G1 — Non-interactive flags, always

| Tool   | Required flags                                                                |
|--------|-------------------------------------------------------------------------------|
| Maven  | `-B -ntp` (batch mode, no transfer progress)                                  |
| sbt    | `--batch --no-colors -Dsbt.color=false -Dsbt.log.noformat=true -Dsbt.server.autostart=false` |
| Bazel  | `--noshow_progress --color=no --curses=no`                                    |
| pip    | `--no-input --disable-pip-version-check --quiet`                              |
| pytest | `-q --no-header --color=no`                                                   |
| Django | `--noinput` on management commands; `python -u` for unbuffered stdout         |

**Why:** removes prompts, ANSI noise, and TTY-detection branches.

### G2 — Servers always start detached with stdio redirected

The canonical pattern:

```
nohup <CMD> > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
```

- `< /dev/null` is mandatory — closes stdin so the server cannot block on input.
- `> "$LOG" 2>&1` is mandatory — frees the parent shell's pipe so the Bash tool sees EOF.
- `nohup` survives the parent shell exit.
- PID file is the single source of truth for stop.sh.

### G3 — Kill by process group, never by PID alone

Start servers in their own process group, kill the whole group on stop:

- Linux: `setsid <CMD>` then `kill -TERM -<PGID>`.
- macOS (no setsid by default): start with `nohup`, capture PID, and kill descendants explicitly:
  ```
  pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null
  ```
  Followed by a bounded wait then `KILL -9` escalation.

**Why:** JVM, Django auto-reloader, Bazel server, sbt server all fork children that survive a bare `kill PID`.

### G4 — Bounded readiness probes (no fixed sleep)

Replace `sleep 10 && curl ...` with:

```
for i in $(seq 1 60); do
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" && exit 0
  sleep 1
done
echo "not ready in 60s"; tail -n 80 "$LOG"; exit 1
```

- Cap = 60 attempts × 1s = 60s wall clock.
- Each curl bounded by `--max-time 2`.
- On failure: tail the log so the model sees *why*, not megabytes of noise.

### G5 — Disable build-tool daemons in skill-driven runs

| Daemon              | Disable flag / env                                          |
|---------------------|-------------------------------------------------------------|
| Gradle daemon       | `GRADLE_OPTS=-Dorg.gradle.daemon=false`                     |
| sbt server          | `-Dsbt.server.autostart=false`, prefer `--client false`     |
| Bazel server        | `bazel shutdown` in stop.sh + `--max_idle_secs=10`          |
| mvnd                | do not use; prefer plain `mvn`                              |

**Why:** daemons hold FDs and ports across runs and are the #1 source of phantom-running.

### G6 — Pre-warm caches once, treat warm runs as the SLA

Cold runs are unavoidable but should not gate the skill's normal loop. The setup script does the cold work once:

- `mvn -B -ntp -q dependency:go-offline`
- `sbt --batch update`
- `pip install --no-input -r requirements.txt`
- `bazel build //... --nobuild`

After warm-up, every subsequent build/test/run runs against the warm SLA targets in Section 7.

### G7 — Wrap every command in a wall-clock timeout

The Bash tool's timeout is the outer fence, but every individual command needs its own:

- Install GNU coreutils on macOS (`brew install coreutils`) so `gtimeout` is available, OR ship a portable Bash timeout helper:
  ```
  run_with_timeout() { ( "$@" ) & local p=$!; ( sleep "$1"; kill -TERM "$p" 2>/dev/null ) & ...; }
  ```
- Apply timeouts to: build, test, hc, stop. Never to the long-running server itself.

### G8 — `set -euo pipefail` in every script, plus a trap

```
set -euo pipefail
trap 'rc=$?; echo "FAIL line $LINENO rc=$rc"; tail -n 40 "${LOG:-/dev/null}" 2>/dev/null; exit $rc' ERR
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
```

**Why:** silent failures (`mvn` exits 1, the next line keeps going) are the dominant cause of "tests passed" being a lie.

### G9 — Logs to file, tail to model

The model gets the **tail of the log on failure**, never the full transcript. This protects context and surfaces the actual error line.

- Build/test scripts: redirect `>"$LOG" 2>&1`, on success print last 20 lines, on failure print last 80.
- Server logs: `tail -n 80 "$LOG"` only when hc fails.

### G10 — Standard script set per POC

Every POC ships exactly these scripts, with identical CLI:

| Script   | Contract                                                                 |
|----------|--------------------------------------------------------------------------|
| setup.sh | Idempotent. Verifies/installs toolchain, pre-warms caches.               |
| build.sh | Bounded. Exits non-zero on any failure. Tails log on failure.            |
| test.sh  | Bounded. Same.                                                           |
| start.sh | Starts server detached per G2/G3. Writes PID file. Returns in < 5 s.     |
| hc.sh    | Bounded readiness probe per G4. Exits 0 when /health is 2xx.             |
| stop.sh  | Kills process group per G3. Idempotent. Cleans PID file.                 |
| clean.sh | Tears down caches and build outputs. Used between full reruns.           |

### G11 — Health endpoint contract

Every runnable POC exposes `GET /health` returning `200 {"status":"ok"}`. No auth. No DB dependency. The endpoint is the **only** signal hc.sh trusts.

### G12 — Empirical proof gate

A POC passes only if all of these succeed in CI-like conditions:

1. `clean.sh && setup.sh` returns 0.
2. `build.sh` returns 0 within the warm SLA (Section 7).
3. `test.sh` returns 0 within the warm SLA.
4. `start.sh` returns 0 in < 5 s; `hc.sh` returns 0 in < 30 s.
5. `stop.sh` returns 0 in < 5 s; **and** no descendant of the start.sh PID is alive 2 s later (`pgrep -P` returns nothing).
6. Re-running `start.sh` + `hc.sh` + `stop.sh` three times in a row succeeds every time (proves no daemon/lock leak).

### G13 — Detect and refuse interactive tools

The skill must lint generated scripts for known interactive traps before running them:

- Bare `sbt` (no `--batch`)
- `pip install` without `--no-input`
- `gh auth login`, `npm login`, `mvn release:prepare` without `-B`
- Reads from `/dev/tty`

### G14 — Cap log volume returned to the model

Hard caps:

- ≤ 80 lines on failure
- ≤ 20 lines on success
- Truncate the middle with `...[N lines elided]...` if needed.

---

## 5. Honest Non-Goals and Limits

We cannot promise:

- That a cold `mvn` first run completes inside the Bash tool's 10-minute ceiling on every machine. **Mitigation:** G6 makes this a one-time setup cost, not a per-run cost.
- That a misbehaving third-party JVM agent won't spawn a child that escapes our process-group kill. **Mitigation:** G12 step 5 detects this and fails loudly.
- That the model "knows what is happening" during a long build. The Bash tool does not stream. **Mitigation:** background mode + periodic `tail` on the log is the closest substitute.

These limits are stated explicitly so the guidelines are not over-sold.

---

## 6. POC Matrix

Eight POCs, one directory each under `pocs/bash-pipe-poc/`:

| # | POC dir                | Stack                                        | Build tool | JDK (via SDKMAN) | HC path           | Port |
|---|------------------------|----------------------------------------------|-----------|------------------|-------------------|------|
| 1 | `java8-mvn-sb2`        | Java 8 + Spring Boot 2.7 (pure Java)         | Maven     | corretto-8       | `/actuator/health`| 8081 |
| 2 | `java25-mvn-sb4`       | Java 25 + Spring Boot 4 (pure Java)          | Maven     | corretto-25      | `/actuator/health`| 8082 |
| 3 | `java25-kotlin-mvn-sb4`| Java 25 + Kotlin + Spring Boot 4             | Maven     | corretto-25      | `/actuator/health`| 8083 |
| 4 | `java25-scala3-sbt-sb4`| Java 25 + Scala 3 + Spring Boot 4            | sbt       | corretto-25      | `/actuator/health`| 8084 |
| 5 | `scala3-sbt`           | Scala 3 + sbt + http4s or Cask (no Spring)   | sbt       | corretto-25      | `/health`         | 8085 |
| 6 | `scala2-bazel`         | Scala 2.13 + Bazel + rules_scala (no Spring) | Bazel     | corretto-25      | `/health`         | 8086 |
| 7 | `python3-plain`        | Python 3.12 + stdlib `http.server`           | pip + venv| n/a              | `/health`         | 8087 |
| 8 | `django-python3`       | Django 5 + Python 3.12                       | pip + venv| n/a              | `/health/`        | 8088 |

Each POC contains the standard script set from G10, a minimal app exposing its HC endpoint, and a `README.md` with the warm SLA target (§7) and the exact `clean → setup → build → test → start → hc → stop` loop.

**JDK toolchain:** every Java/Kotlin/Scala POC's `setup.sh` begins with:

```
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk use java <corretto-id>
```

The JDK id is pinned per POC (`corretto-8` for #1, `corretto-25` for #2–#6). Cold install (`sdk install java <id>`) happens on demand if `sdk list java | grep '* installed'` does not include the pin.

**Note on Bazel + Scala 2:** confirmed. rules_scala latest, no specific pin. This is the slowest-to-bootstrap POC — cold setup.sh may take minutes — but warm SLA in §7 still applies.

**Note on Spring Boot 4 + Scala 3:** Spring Boot with Scala is unusual; per Diego's project memory, SB4 also dropped `TestRestTemplate`, `@AutoConfigureGraphQlTester`, and the `com.fasterxml.jackson` package. ITs in this POC will use JDK `HttpClient` + string JSON, matching the established pattern.

---

## 7. Warm SLA Targets (after setup.sh)

These are the numbers the POCs must hit on a warm cache. Used as the pass/fail gate in G12.

| Stack                  | build.sh | test.sh | start.sh→hc.sh | stop.sh |
|------------------------|----------|---------|----------------|---------|
| java8-mvn-sb2          | ≤ 30 s   | ≤ 30 s  | ≤ 20 s         | ≤ 5 s   |
| java25-mvn-sb4         | ≤ 45 s   | ≤ 45 s  | ≤ 25 s         | ≤ 5 s   |
| java25-kotlin-mvn-sb4  | ≤ 60 s   | ≤ 60 s  | ≤ 25 s         | ≤ 5 s   |
| java25-scala3-sbt-sb4  | ≤ 90 s   | ≤ 90 s  | ≤ 30 s         | ≤ 5 s   |
| scala3-sbt             | ≤ 60 s   | ≤ 60 s  | ≤ 25 s         | ≤ 5 s   |
| scala2-bazel           | ≤ 60 s   | ≤ 60 s  | ≤ 25 s         | ≤ 5 s   |
| python3-plain          | ≤ 5 s    | ≤ 10 s  | ≤ 10 s         | ≤ 5 s   |
| django-python3         | ≤ 10 s   | ≤ 15 s  | ≤ 15 s         | ≤ 5 s   |

---

## 8. Reference Patterns (pseudocode only — no implementation yet)

### 8.1 start.sh skeleton (applies to every stack)

```
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/run.log"; PIDFILE="$HERE/run.pid"
[ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && { echo "already running"; exit 0; }
: > "$LOG"
nohup <STACK_RUN_CMD> > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE") log=$LOG"
```

### 8.2 hc.sh skeleton

```
#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8080}"; LOG="${2:-run.log}"
for i in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "healthy after ${i}s"; exit 0
  fi
  sleep 1
done
echo "unhealthy after 60s"; tail -n 80 "$LOG" 2>/dev/null || true; exit 1
```

### 8.3 stop.sh skeleton (macOS-portable)

```
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$HERE/run.pid"
[ -f "$PIDFILE" ] || { echo "not running"; exit 0; }
PID="$(cat "$PIDFILE")"
pkill -TERM -P "$PID" 2>/dev/null || true
kill -TERM "$PID" 2>/dev/null || true
for i in $(seq 1 5); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$PID" 2>/dev/null || true
pkill -KILL -P "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "stopped"
```

---

## 9. Resolved Decisions (was: Open Questions)

| # | Question | Decision |
|---|----------|----------|
| 1 | OS target | **macOS only.** Scripts drop Linux/`setsid` paths; rely on `pkill -P` + bounded escalation. |
| 2 | JDK install | **SDKMAN + Amazon Corretto.** Pin per POC: `corretto-8` for Java 8, `corretto-25` for Java 25. |
| 3 | Bazel + Scala 2 | **Latest rules_scala, no version pin.** |
| 4 | Java 25 web layer | **Spring Boot 4 in three flavors:** pure Java + Maven, Kotlin + Maven, Scala 3 + sbt. |
| 5 | HC endpoint | **The skill discovers it** — see §12 for the rules. Spring POCs use `/actuator/health`, Django uses `/health/`, others use `/health`. |
| 6 | Layout | **All POCs under `pocs/bash-pipe-poc/<stack>/`.** |

---

## 10. Skill-Authoring Implications (the skill is a Bash generator)

Diego's skills do not call `mvn` themselves — they **generate the Bash scripts** that the agent then executes. So the failure surface is not "did the agent run a good command?" but "did the skill emit good Bash?". Three implications:

### 10.1 Guidelines must be in the skill's prompt, not just in the agent's head

The SKILL.md (or equivalent) MUST embed the rules from §4 as **prescriptive output requirements**, e.g.:

> When generating any `start.sh`, you MUST:
> - redirect stdout/stderr to a log file
> - redirect stdin from /dev/null
> - prefix the command with nohup
> - write the PID to a `.pid` file
> - return within 5 seconds without waiting for readiness (hc.sh does that)

Without this, the LLM regresses to "natural" bash like `mvn clean install &` which is exactly the broken pattern we are trying to eliminate.

### 10.2 The skill MUST emit the standard script set, not ad-hoc commands

The skill's contract with the agent is: emit `setup.sh / build.sh / test.sh / start.sh / hc.sh / stop.sh / clean.sh`, each conforming to G10. The agent then runs them. This shifts the surface from "the agent invented a one-liner" to "the agent ran a known-good script".

### 10.3 The skill MUST run a self-lint before handing scripts back

After generating scripts, the skill runs a lint pass that rejects emissions containing any of:

- `mvn ` without `-B`
- `sbt ` not followed by `--batch`
- ` &` not preceded by `nohup` and not followed by stdio redirection
- bare `sleep` of more than 1 in a wait loop
- absence of `set -euo pipefail` at the top of every script
- `curl ` without `--max-time` in a poll loop
- absence of a log-tail-on-failure path

If the lint fails, the skill regenerates. This makes G1–G14 **enforceable**, not aspirational.

### 10.4 The POCs double as golden outputs

The six POCs in §6 are not only proof that the guidelines work — they are the **exemplars the skill embeds verbatim** as templates. When the skill needs to emit a `start.sh` for a Java/Maven project, it adapts `pocs/bash-pipe-poc/java8-mvn/start.sh`. This eliminates LLM creativity in the exact place where creativity hurts.

---

## 11. Skill Discovery Rules (port + HC path)

The skill cannot hardcode `/health` on port 8080 — it must read the codebase and decide. The discovery walk runs in `setup.sh` once and writes the result to a `bash-pipe.env` file the other scripts source.

### 11.1 Framework detection

Order matters — first match wins.

| Probe | Framework | Output |
|-------|-----------|--------|
| `pom.xml` contains `spring-boot-starter-actuator` | Spring Boot (Java/Kotlin) | `FRAMEWORK=spring-boot` |
| `pom.xml` contains `spring-boot` (no actuator) | Spring Boot, no actuator | `FRAMEWORK=spring-boot-bare` |
| `build.sbt` contains `spring-boot-starter` | Spring Boot via sbt | `FRAMEWORK=spring-boot-sbt` |
| `build.sbt` contains `http4s` or `cask` | Scala HTTP | `FRAMEWORK=scala-http` |
| `WORKSPACE` or `MODULE.bazel` + `rules_scala` | Bazel + Scala | `FRAMEWORK=bazel-scala` |
| `manage.py` present and imports `django` | Django | `FRAMEWORK=django` |
| `pyproject.toml` or `requirements.txt`, no Django | Python plain | `FRAMEWORK=python-plain` |
| Fallback | unknown | fail loudly, ask user |

### 11.2 HC path discovery

| FRAMEWORK | HC path | Notes |
|-----------|---------|-------|
| spring-boot, spring-boot-sbt | `/actuator/health` | requires actuator dep |
| spring-boot-bare | `/health` | POC must add a `@RestController` |
| scala-http, bazel-scala, python-plain | `/health` | POC must implement |
| django | `/health/` | trailing slash matters; POC adds a view |

Override: if the project root contains `bash-pipe.env` with `HC_PATH=...` already set, the discovery skips and trusts the file.

### 11.3 Port discovery

Per-framework probe order:

- **Spring Boot:** `application.properties` → `server.port=`; `application.yml` → `server.port:`; env `SERVER_PORT`; fallback `8080`.
- **Django:** env `PORT`; `manage.py runserver` arg in any wrapper script the project ships; fallback `8000`.
- **Python plain:** grep main module for `bind`/`listen`/`HTTPServer(("", N))`; env `PORT`; fallback `8000`.
- **Bazel/sbt non-Spring:** read source for the framework's bind call (http4s: `Host.fromString` + `Port.fromInt`); fallback per POC table in §6.

If no port can be determined, `setup.sh` fails loudly with the list of files it checked. No silent defaults beyond the per-framework fallback above.

### 11.4 The contract written to `bash-pipe.env`

```
FRAMEWORK=spring-boot
HC_PATH=/actuator/health
PORT=8082
RUN_CMD="./mvnw -B -ntp spring-boot:run"
BUILD_CMD="./mvnw -B -ntp -DskipTests package"
TEST_CMD="./mvnw -B -ntp test"
JDK_ID=corretto-25
```

`start.sh`, `build.sh`, `test.sh`, `hc.sh` all `source bash-pipe.env` and use those vars. No string interpolation of project paths in the script bodies — only env vars from this file.

---

## 12. The Generator Skill (`bash-pipe-skill`)

Diego asked for "the skill that produces a skill". I am reading that as **one** of these two scopes — please confirm which in §14:

### 12.A — Scripts-only generator (smaller scope)

Given a project directory, the skill emits the standard script set (G10) into the project, plus `bash-pipe.env` from §11. The skill itself is the orchestrator; there is no per-project SKILL.md generated.

### 12.B — Meta-skill (larger scope, "skill that produces a skill")

Given a project directory, the skill emits **a new project-local SKILL.md** under `.claude/skills/<project>-bash-pipe/SKILL.md` plus the scripts. That generated SKILL.md is what Diego invokes thereafter for that project. The meta-skill is run once per project; the generated skill is run on every change.

Both share the same generator logic; only the output surface differs. I lean toward **12.A** unless Diego wants the per-project skill artifact for cataloguing in Claude Code's skill registry — say the word.

### 12.3 Generator responsibilities (common to both scopes)

1. Walk the project to run §11 detection. Write `bash-pipe.env`.
2. Emit the seven scripts from G10, all of them sourcing `bash-pipe.env`.
3. Run the §10.3 self-lint on the emitted scripts. Regenerate on any lint failure.
4. Run the §G12 proof gate against the project itself: clean → setup → build → test → start → hc → stop, with the warm SLA from §7 (or a per-project override).
5. Print a one-screen summary (≤ 40 lines) of what was emitted, what the gate measured, and any guideline that could not be satisfied (and why).

The eight POCs in §6 are the generator's **golden templates**. The generator picks a template by `FRAMEWORK` from §11.1, then patches it with the project-specific env vars. This eliminates LLM creativity exactly where it hurts.

---

## 13. Decision Log

- **Picked plain mvn over mvnd:** mvnd's daemon is exactly the FD-leak class we are trying to eliminate.
- **Picked sbt --batch over sbt-server:** server mode is faster but its lifecycle escapes the Bash tool's pipe-EOF signal.
- **Picked bounded poll over fixed sleep:** fixed sleep is either too short (flaky) or too long (slow).
- **Picked tail-on-failure over full log:** preserves model context budget; the last 80 lines almost always contain the actual error.
- **Picked process-group kill over PID kill:** matches the observed failure mode (JVM children outliving the parent).
- **Picked SDKMAN + Corretto pinned per POC:** matches Diego's setup; `sdk use java <id>` in each setup.sh isolates JDK selection per POC without polluting the global default.
- **Expanded Java 25 into three flavors (Java/Kotlin/Scala on SB4):** Diego's stated need; also exercises the skill's framework-detection path on three variants that all answer "yes" to "is this Spring Boot?" but differ in build tool and language.
- **Made the HC path/port discovered, not hardcoded:** Diego's requirement — the skill must read the codebase, not assume `/health` on 8080.
- **Wrote `bash-pipe.env` as the single source of truth:** keeps the seven scripts dumb and identical across POCs; per-project variance lives in env vars only.

---

## 14. Three-Layer Architecture (skill → orchestrator → judge)

Confirmed scope per Diego's latest message: **12.A — scripts-only generator.** The skill emits a top-level **orchestrator** bash plus per-POC scripts. Crucially, a separate **judge** script — written once and never modified by the skill — grades the orchestrator's output. This gives a clean three-layer separation:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: The Skill            (regenerated each invocation)     │
│   - Reads project / POC dirs                                    │
│   - Emits orchestrator + per-POC scripts per §G10               │
│   - Runs self-lint per §10.3                                    │
└─────────────────────────────────────────────────────────────────┘
                              │ emits
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: The Generated Orchestrator       (regenerated)         │
│   - Runs each POC's build → test → start → hc → stop            │
│   - Counts tests, captures durations, records failures          │
│   - Writes results.json (machine) + problems.md (human)         │
└─────────────────────────────────────────────────────────────────┘
                              │ emits
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: The Judge        (IMMUTABLE — written once, never      │
│                            modified by the skill or by Claude)  │
│   - Knows expected test counts, build/hc pass per stack         │
│   - Reads results.json + problems.md                            │
│   - Exits 0 iff every expectation matched                       │
│   - Prints a diff of expected vs actual on failure              │
└─────────────────────────────────────────────────────────────────┘
```

This makes the judge a **property test for the skill itself**: if the skill is regenerated and starts producing bad bash, the judge catches it because the ground truth is encoded outside the skill.

---

## 15. Output Contract — `results.json` and `problems.md`

The orchestrator's job ends when these two files are written. The judge reads only these files (plus the per-POC log dir as supporting evidence).

### 15.1 `results.json` schema

```json
{
  "schema_version": 1,
  "started_at": "2026-05-20T14:30:00Z",
  "wall_clock_sec": 312,
  "stacks": {
    "java8-mvn-sb2": {
      "build":  { "pass": true, "duration_sec": 27, "exit_code": 0 },
      "tests":  { "pass": true, "total": 3, "passed": 3, "failed": 0, "skipped": 0, "duration_sec": 18 },
      "start":  { "pass": true, "duration_sec": 4, "pid": 12345 },
      "hc":     { "pass": true, "status_code": 200, "attempts": 3, "duration_sec": 3 },
      "stop":   { "pass": true, "duration_sec": 2, "orphans": 0 }
    },
    "java25-mvn-sb4":        { "...": "same shape" },
    "java25-kotlin-mvn-sb4": { "...": "same shape" },
    "java25-scala3-sbt-sb4": { "...": "same shape" },
    "scala3-sbt":            { "...": "same shape" },
    "scala2-bazel":          { "...": "same shape" },
    "python3-plain":         { "...": "same shape" },
    "django-python3":        { "...": "same shape" }
  },
  "summary": {
    "total_stacks": 8,
    "passed_stacks": 8,
    "failed_stacks": 0,
    "total_tests": 24,
    "total_tests_passed": 24
  }
}
```

Hard rules:

- Every stack key in §6 MUST appear, even on failure. Missing key === orchestrator bug, judge fails.
- `tests.total` MUST come from the test runner's machine-readable output (Surefire XML, pytest `-q` summary, Django `Ran N tests`), never guessed.
- `stop.orphans` is `pgrep -P "$PID" | wc -l` measured **2 seconds after stop.sh returns**. Non-zero means G3 was violated. Judge treats `orphans > 0` as a failure even if `stop.pass=true`.
- All durations in **seconds, integer**. Rounded up.

### 15.2 `problems.md` schema

Human-readable. Used for debugging — judge only checks it exists when any `pass=false`.

```markdown
# Problems — 2026-05-20T14:30:00Z

## scala2-bazel
- build.sh exit 1, duration 12s
- Root cause hint (tail of run.log:
  ```
  ERROR: ... Bazel server lock held by pid 99887
  ```

## django-python3
- hc.sh failed after 30s, port 8088 not listening
- See django-python3/run.log lines 40–80
```

Required content rules:

- One H2 per failing stack. No H2 if all stacks pass.
- Each failure line includes the script name, exit code, duration.
- Always include a log tail (≤ 20 lines per failure) — protects context per G14.

### 15.3 Per-POC contribution

Each POC dir contributes a small `result.fragment.json` that the orchestrator merges. This isolates test-runner parsing (Surefire vs pytest vs sbt vs Bazel) inside each POC, where it belongs.

```
java8-mvn-sb2/
  result.fragment.json   <-- written by test.sh / hc.sh / stop.sh
  run.log
  test.log
  build.log
```

The orchestrator's job is then just: iterate POCs, run their scripts in order, read fragments, aggregate.

---

## 16. The Judge — `judge.sh` (immutable oracle)

Diego writes this **once** and never modifies. The skill never touches it. It encodes the ground truth.

### 16.1 What the judge knows (hardcoded expectations)

```bash
declare -A EXPECT_TESTS=(
  [java8-mvn-sb2]=3
  [java25-mvn-sb4]=3
  [java25-kotlin-mvn-sb4]=3
  [java25-scala3-sbt-sb4]=3
  [scala3-sbt]=3
  [scala2-bazel]=3
  [python3-plain]=3
  [django-python3]=3
)
EXPECT_STACKS=8
EXPECT_BUILD_PASS_ALL=true
EXPECT_HC_PASS_ALL=true
EXPECT_STOP_ORPHANS=0
EXPECT_MAX_WALL_CLOCK_SEC=900
```

These numbers are deliberately **exact, not "at least"** — that is the whole point of having an oracle. Bumping a count is a deliberate edit to the judge that requires explicit human approval. If a POC's test count drifts unintentionally, the judge fires.

### 16.2 What the judge checks

For each stack in `EXPECT_TESTS`:

1. Stack key exists in `results.json`.
2. `build.pass == true`.
3. `tests.pass == true` AND `tests.total == EXPECT_TESTS[stack]` AND `tests.failed == 0` AND `tests.skipped == 0`.
4. `start.pass == true`.
5. `hc.pass == true` AND `hc.status_code == 200`.
6. `stop.pass == true` AND `stop.orphans == 0`.

Across the run:

7. `summary.total_stacks == 8`, `passed_stacks == 8`.
8. `wall_clock_sec <= EXPECT_MAX_WALL_CLOCK_SEC`.
9. If any check above fails, `problems.md` MUST exist and contain an H2 for each failing stack.

### 16.3 Judge output

On pass:

```
JUDGE: PASS — 8/8 stacks, 24/24 tests, wall=312s (cap=900s)
```

On failure, a structured diff and a non-zero exit:

```
JUDGE: FAIL
  scala2-bazel.tests.total  expected=3 actual=0
  scala2-bazel.build.pass   expected=true actual=false
  django-python3.hc.pass    expected=true actual=false
problems.md present: yes
exit 2
```

### 16.4 The judge as the skill's quality gate

Workflow:

1. Regenerate the skill (or the orchestrator) however you like.
2. Run the orchestrator: `./run-all.sh` → writes `results.json` + `problems.md`.
3. Run `./judge.sh`. **If exit 0, the skill is doing its job. If non-zero, the skill regressed.**

The judge is the single, authoritative answer to "did the change to the skill break anything?".

### 16.5 Immutability protections

- `judge.sh` lives at `pocs/bash-pipe-poc/judge.sh` and is checked in.
- `judge.sh` is mentioned in CLAUDE.md (or a local `.claude-rules`) as **off-limits to the skill and to the agent unless Diego explicitly approves an edit**.
- The skill's self-lint (§10.3) also refuses to write to anything named `judge.sh`.

---

## 17. Final Open Question Before Code

The judge bakes in expected test counts. I proposed **3 tests per POC × 8 POCs = 24 total** in §16.1 to keep every POC honest with the minimum useful coverage:

- `test_health_endpoint_returns_200`
- `test_health_payload_shape`
- `test_root_or_app_specific_smoke`

Please confirm or override:

- **Option A (recommended):** 3 tests per POC, exact match — clean, uniform, tight.
- **Option B:** a per-stack number you want me to use instead (give me the table).

Once you confirm, the build order is:

1. Lock the eight POCs (apps + scripts) — each must pass the §G12 proof gate locally.
2. Write `judge.sh` with the agreed expected counts. **This is the immutable artifact — written once.**
3. Write the generator skill so that re-running it regenerates the orchestrator + per-POC scripts, and `judge.sh` passes.
4. Sanity loop: regenerate skill → run orchestrator → run judge → green.
