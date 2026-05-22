#!/usr/bin/env bash
# Anti-pattern: child process inherits the parent shell's stdout/stderr
# Source: design-doc.md §2.1 -- the central case behind 'phantom-running'
# Why it breaks: the Claude Code Bash tool waits for EOF on stdout/stderr.
# A Surefire JVM, sbt server, Django auto-reloader, or Bazel daemon that
# inherits those FDs keeps the pipe open even after the parent exits. The
# tool call therefore appears 'running' indefinitely.

set -euo pipefail

echo "ANTI-PATTERN: child inherits parent's stdout/stderr -> phantom-running"
echo "WHY: Bash tool waits for pipe EOF; inherited FDs never close."
echo
echo "DEMO: launch a child that inherits FDs; show the FDs it still holds."

outer=$(mktemp /tmp/bp_ap12_outer.XXXXXX)
bash -c '
  sleep 30 &
  echo PHANTOM_PID=$! > /tmp/bp_ap12.pid
' > "$outer" 2>&1

phantom=$(grep -oE '[0-9]+' /tmp/bp_ap12.pid | head -n1)
echo "  spawned phantom pid=$phantom"
echo "  parent shell already exited, yet:"
if ps -p "$phantom" >/dev/null 2>&1; then
  if command -v lsof >/dev/null 2>&1; then
    lsof -p "$phantom" 2>/dev/null | awk 'NR==1 || $4 ~ /^[012]/ {print "    " $0}' | head -n 6
  fi
  kill "$phantom" 2>/dev/null || true
fi
rm -f /tmp/bp_ap12.pid "$outer"
echo
echo "Correct form (A1, G2): close the pipe explicitly --"
echo "  nohup CMD > LOG 2>&1 < /dev/null & echo \$! > PIDFILE"
