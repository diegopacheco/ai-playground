---
description: Render the latest lint report as a modern web report in Podman
argument-hint: "[path]"
---

Render the lint report for the codebase at `$ARGUMENTS` (default: current working
directory) as a web report. Steps:

1. Verify `<target>/.lint/report.json` exists. If it does not, tell the user to
   run `/lint $ARGUMENTS` first and stop.

2. Start the web stack (frontend + backend in Podman), passing the target so the
   backend can mount `<target>/.lint` and the source tree read-only:
   `bash ~/.claude/skills/agent-skill-linter/site/start.sh $ARGUMENTS`
   The script brings up both containers and waits on readiness.

3. Print the URL (http://localhost:8088) and open it. The report has five tabs:
   Dashboard, Tests, Charts and trends, Rules, and Code viewer, with a global
   search box that filters the active tab.

4. To shut it down: `bash ~/.claude/skills/agent-skill-linter/site/stop.sh`.

The user only ever sees the URL; Podman, the JVM, and Bun stay hidden.
