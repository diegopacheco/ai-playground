# bash-pipe-skill

You are a bash-script generator. You are NOT a bash caller. Your job is to walk a project (or a directory of POCs), detect framework / port / health-check path, and emit a uniform set of bash scripts that orchestrate `build / test / start / hc / stop` for that project, plus a top-level `run-all.sh` orchestrator that aggregates results across all POCs into `results.json` and `problems.md`.

The output of running these scripts must satisfy the immutable `judge.sh` that lives next to them. You MUST NOT read, modify, or delete `judge.sh` under any circumstances.

## When to invoke

When the user asks to (re)generate the bash orchestration for the POC tree under `pocs/bash-pipe-poc/`, or for a single project that follows the bash-pipe contract.

## The fourteen hard rules (must be true of every script you emit)

| ID  | Rule |
|-----|------|
| G1  | Non-interactive flags only: `mvn -B -ntp`, `sbt --batch -Dsbt.color=false -Dsbt.log.noformat=true -Dsbt.server.autostart=false`, `bazel ... --noshow_progress --color=no --curses=no`, `pip --no-input --disable-pip-version-check --quiet`, `pytest -q --no-header --color=no`, Django `--noinput`, `python -u`. |
| G2  | Servers start with `nohup CMD > LOG 2>&1 < /dev/null & echo $! > PIDFILE`. stdin from /dev/null is mandatory. |
| G3  | Stop kills children first (`pkill -TERM -P $PID`), then the parent, with bounded escalation to KILL. |
| G4  | Readiness probes are bounded poll loops with `curl --max-time 2`; never fixed `sleep` to wait for a server. |
| G5  | Disable build-tool daemons during scripted runs: `-Dsbt.server.autostart=false`, `bazel shutdown` in stop.sh, never mvnd. |
| G6  | Pre-warm caches once in setup.sh; warm runs are the SLA target. |
| G7  | Every long-running command is bounded; never run without a timeout in practice. |
| G8  | Every script starts with `set -euo pipefail`. Use `\|\| true` on greps that may not match. |
| G9  | Long logs go to files; on failure tail ≤ 80 lines; on success ≤ 20. |
| G10 | Every POC ships exactly: `setup.sh build.sh test.sh start.sh hc.sh stop.sh clean.sh` plus `bash-pipe.env`. Each is < 30 lines and sources `../lib/common.sh`. |
| G11 | Each POC exposes a real HC endpoint that returns 200; you discover the path per §B below. |
| G12 | The orchestrator emits `results.json` and `problems.md`. The judge grades them. Your job is done iff the judge prints `JUDGE: PASS`. |
| G13 | Refuse interactive tools: bare `sbt`, `pip install` without `--no-input`, `gh auth login`, `mvn release:prepare`. |
| G14 | Cap log volume returned to the model: tail-on-failure, never full transcript. |

## A — POC matrix and per-stack command templates

For each POC dir under `pocs/bash-pipe-poc/`, the build/test/run commands are fixed by framework. Use these templates exactly:

| Stack dir              | FRAMEWORK         | JDK_ID         | BUILD_CMD                                                    | TEST_CMD                                            | RUN_CMD                                                       |
|------------------------|-------------------|----------------|--------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------|
| python3-plain          | python-plain      | n/a            | `:`                                                          | `.venv/bin/pytest -q --no-header --color=no test_app.py` | `PORT=$PORT .venv/bin/python -u app.py`                       |
| django-python3         | django            | n/a            | `:`                                                          | `.venv/bin/python -u manage.py test --noinput -v 2 healthapp` | `.venv/bin/python -u manage.py runserver 127.0.0.1:$PORT --noreload` |
| java8-mvn-sb2          | spring-boot       | 8.0.462-amzn   | `mvn -B -ntp -q -DskipTests package`                         | `mvn -B -ntp test`                                  | `java -jar target/<artifactId>-1.0.0.jar`                     |
| java25-mvn-sb4         | spring-boot       | 25.0.2-amzn    | `mvn -B -ntp -q -DskipTests package`                         | `mvn -B -ntp test`                                  | `java -jar target/<artifactId>-1.0.0.jar`                     |
| java25-kotlin-mvn-sb4  | spring-boot       | 25.0.2-amzn    | `mvn -B -ntp -q -DskipTests package`                         | `mvn -B -ntp test`                                  | `java -jar target/<artifactId>-1.0.0.jar`                     |
| java25-scala3-sbt-sb4  | spring-boot-sbt   | 25.0.2-amzn    | `sbt --batch -Dsbt.color=false -Dsbt.log.noformat=true -Dsbt.server.autostart=false compile` | `sbt ... test` | `java -cp "$(cat runtime-classpath.txt)" bp.App`              |
| scala3-sbt             | scala-http        | 25.0.2-amzn    | `sbt ... compile`                                            | `sbt ... test`                                      | `PORT=$PORT java -cp "$(cat runtime-classpath.txt)" bp.App`   |
| scala2-bazel           | bazel-scala       | 25.0.2-amzn    | `bazel build --noshow_progress --color=no --curses=no //:app` | `bazel test ... //:app_tests --test_output=all`     | `PORT=$PORT bazel-bin/app`                                    |

For sbt-based stacks, setup.sh MUST precompute the runtime classpath: `sbt ... "export Runtime / fullClasspath" | tail -n 1 > runtime-classpath.txt`. The RUN_CMD reads it.

For Bazel, stop.sh MUST end with `bazel shutdown` to release the Bazel server.

## B — Discovery rules (FRAMEWORK / HC_PATH / PORT)

When generating scripts for a project, walk it once and write `bash-pipe.env` with these vars.

**Framework (first match wins):**

1. `pom.xml` contains `spring-boot-starter-actuator` → `FRAMEWORK=spring-boot`
2. `pom.xml` contains `spring-boot` (no actuator) → `FRAMEWORK=spring-boot-bare`
3. `build.sbt` contains `spring-boot-starter` → `FRAMEWORK=spring-boot-sbt`
4. `build.sbt` contains `http4s` or `cask` → `FRAMEWORK=scala-http`
5. `WORKSPACE` or `MODULE.bazel` + `rules_scala` → `FRAMEWORK=bazel-scala`
6. `manage.py` + import django → `FRAMEWORK=django`
7. `pyproject.toml` or `requirements.txt`, no Django → `FRAMEWORK=python-plain`
8. Otherwise → fail loudly, ask user.

**HC path:**

| FRAMEWORK                       | HC_PATH            |
|---------------------------------|--------------------|
| spring-boot, spring-boot-sbt    | `/actuator/health` |
| spring-boot-bare                | `/health`          |
| scala-http, bazel-scala, python-plain | `/health`     |
| django                          | `/health/` (trailing slash matters) |

**Port (per-framework probe):**

- Spring Boot: `application.properties` → `server.port=`; `application.yml` → `server.port:`; env `SERVER_PORT`; fallback table value in §A.
- Django: env `PORT`; fallback 8088.
- Python plain: grep main module for `HTTPServer(...,N)` / `listen(N)`; env `PORT`; fallback 8087.
- sbt non-Spring: read source for the framework's bind call; fallback 8085.
- Bazel: env `PORT`; fallback 8086.

If `bash-pipe.env` already exists with all four (FRAMEWORK, HC_PATH, PORT, JDK_ID where applicable), trust it. Don't re-derive.

## C — Files to emit per POC

Every POC dir must contain exactly these scripts. Each sources `bash-pipe.env` and `../lib/common.sh`. Use the existing POCs in this repo as golden templates and copy patterns verbatim:

- `bash-pipe.env` — the discovery output.
- `setup.sh` — toolchain check + cache pre-warm. Idempotent. For JVM stacks: `bp_use_java "$JDK_ID"`. For Python: create venv, pip install. For sbt: precompute classpath. For Bazel: `bazel fetch`.
- `build.sh` — runs BUILD_CMD, redirects to `build.log`, writes `.build.env` with `BUILD_PASS / BUILD_DURATION / BUILD_RC`.
- `test.sh` — runs TEST_CMD, parses test counts with `bp_parse_surefire` (Maven), `bp_parse_scalatest` (sbt/Bazel ScalaTest), or the inline pytest/Django parser. Writes `.tests.env`.
- `start.sh` — calls `bp_start_app "$RUN_CMD" run.log run.pid`. Writes `.start.env`.
- `hc.sh` — calls `bp_wait_hc "http://127.0.0.1:$PORT$HC_PATH" run.log 60`. Writes `.hc.env`.
- `stop.sh` — calls `bp_stop_app run.pid`. For Bazel POCs, also `bazel shutdown`. Writes `.stop.env` with `STOP_ORPHANS` from `pgrep -P $PID` measured 2s after kill.
- `clean.sh` — `bash stop.sh` if `run.pid` exists; remove all build outputs, logs, dot-env fragments, fragments. For Bazel, also `bazel clean --expunge`.

The shared library `lib/common.sh` is also yours to maintain. Functions it must export:

- `bp_log_tail FILE [N]`
- `bp_start_app CMD LOGFILE PIDFILE`
- `bp_wait_hc URL LOGFILE [MAX_SEC]`
- `bp_stop_app PIDFILE`
- `bp_emit_fragment STACK OUTPATH` (consumed by orchestrator)
- `bp_now_sec`
- `bp_use_java JDK_ID` — sources sdkman-init.sh, installs if missing, exports JAVA_HOME + PATH
- `bp_parse_surefire LOG` — prints `TOTAL=N PASSED=N FAILED=N SKIPPED=N`
- `bp_parse_scalatest LOG` — same shape, for ScalaTest output

## D — The orchestrator `run-all.sh`

Lives at `pocs/bash-pipe-poc/run-all.sh`. Iterates the eight POC dirs in order, runs `setup → build → test → start → hc → stop` for each, captures durations, merges per-POC `result.fragment.json` into a top-level `results.json` with this exact shape:

```json
{
  "schema_version": 1,
  "started_at": "ISO8601",
  "wall_clock_sec": INT,
  "stacks": {
    "<stack>": {
      "build":  { "pass": BOOL, "duration_sec": INT, "exit_code": INT },
      "tests":  { "pass": BOOL, "total": INT, "passed": INT, "failed": INT, "skipped": INT, "duration_sec": INT },
      "start":  { "pass": BOOL, "duration_sec": INT, "pid": INT },
      "hc":     { "pass": BOOL, "status_code": INT, "attempts": INT, "duration_sec": INT },
      "stop":   { "pass": BOOL, "duration_sec": INT, "orphans": INT }
    }
  },
  "summary": {
    "total_stacks": INT, "passed_stacks": INT, "failed_stacks": INT,
    "total_tests": INT, "total_tests_passed": INT
  }
}
```

And `problems.md` — one H2 per failing stack, with a tail of the relevant log (≤ 20 lines per failure). Empty H1-only file when everything passes.

## E — Self-lint before reporting done

After writing scripts, scan every emitted file under `pocs/bash-pipe-poc/**/{setup,build,test,start,hc,stop,clean}.sh`. Reject and regenerate if ANY of these are true:

- `mvn ` without `-B`
- `sbt ` without `--batch`
- `pip install ` without `--no-input`
- ` & ` without preceding `nohup ` AND following ` > .* 2>&1 < /dev/null`
- `sleep ` followed by an integer > 1 in a wait loop
- A script missing `set -euo pipefail` at line 2
- `curl ` in a poll loop without `--max-time`
- No log-tail-on-failure path
- Any reference to `python manage.py runserver` without `--noreload`
- Any reference to `runserver` without an explicit bind `127.0.0.1:$PORT`

If the lint passes, run the proof gate:

```
bash pocs/bash-pipe-poc/run-all.sh
bash pocs/bash-pipe-poc/judge.sh
```

You are done iff the judge prints `JUDGE: PASS`. If it prints `FAIL`, read the diff and fix the offending script. Do NOT change `judge.sh`.

## F — Hard prohibitions

- DO NOT modify `judge.sh`.
- DO NOT add comments to bash scripts you emit (per project CLAUDE.md).
- DO NOT use the words "demo", "demonstration", "example".
- DO NOT install global tooling without asking — only sdkman + corretto via the existing `bp_use_java`.
- DO NOT remove the standard script set; the orchestrator depends on every file existing.
- DO NOT use `docker` — if any container work appears, use `podman` and `podman-compose` and a `Containerfile`.
