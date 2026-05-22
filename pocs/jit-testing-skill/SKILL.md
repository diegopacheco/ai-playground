---
name: jit
description: Generate ephemeral catching tests on the current diff, run them, and report behavior regressions. Tests are never committed. Works across Java, Kotlin, Scala, Python, and Node.js stacks.
---

You are running the JIT Catching Test skill. The user invoked `/jit`. Your job is to find bugs introduced by the user's pending diff, without writing any test that lands in the repo.

## Pipeline

1. Detect the language target:
   ```
   ~/.claude/jit/bin/jit detect
   ```
2. Decide the comparison mode. The pipeline auto-detects:
   - **Snapshot mode** (default if `.parent/` exists in the cwd): compares the current files against `.parent/`. No git history required. Used by sample projects and any directory that ships a parent snapshot.
   - **Git mode** (otherwise): compares `HEAD~1..HEAD` (or a user-supplied range).
3. Honor `--diff <range>`, `--target <name>`, `--workflow {dodgy|intent|both}`, `--mode {auto|git|snapshot}`, `--parent-dir <dir>` if the user passed them.
4. Run the pipeline:
   ```
   ~/.claude/jit/bin/jit run --repo . [--diff <range>] [--target X] [--workflow Y] [--mode Z]
   ```
5. The pipeline writes its output to `.jit-testing/runs/<run_id>/`. Read `report.md` from that directory.
6. Present the top three catches as **sense-check questions**. Do not show test code first. Example:
   > On the parent, `qualifiesForFreeShipping($49.99 delivery)` returned `false`. On your change, it returns `true`. Is this expected?
7. For each catch, offer three actions: *Confirm bug*, *Expected change*, *Show test code*.
8. If the user picks *Show test code*, print the relevant section of the report.
9. On a verdict, persist it:
   ```
   ~/.claude/jit/bin/jit verdict <run_id> <catch_id> {confirmed|expected|deferred}
   ```

## Supported targets

`java8`, `java25`, `scala3-sbt`, `scala2-bazel`, `kotlin`, `python3`, `python3-django`, `nodejs`.

If detection fails, stop and ask the user to pass `--target` explicitly.

## Hard rules

- Never commit anything from `.jit-testing/`.
- Never write tests outside `.jit-testing/runs/<run_id>/tests/`.
- Keep dismissals fast: do not lecture the user when they say a change is expected.

## After the run

Tell the user they can open the dashboard with `/jit-dashboard`.
