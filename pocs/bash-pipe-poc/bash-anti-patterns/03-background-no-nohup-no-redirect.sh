#!/usr/bin/env bash
# Anti-pattern: 'cmd &' with no nohup, no stdout/stderr redirect, no PID file
# Source: design-doc.md §2.3, countered by G2
# Why it breaks: the child inherits the parent shell's stdout pipe, so the
# Bash tool never sees EOF and the call appears to run forever. No PID
# file means stop.sh has nothing to kill. No nohup means a SIGHUP races
# the orphan.

set -euo pipefail

echo "ANTI-PATTERN: backgrounding a server with bare '&'"
echo "WHY: child inherits stdout -> pipe stays open -> phantom-running."
echo "     No PID file -> stop.sh cannot find what to kill."
echo
echo "DEMO: spawn a child the wrong way, then look at the pipe it holds."

bash -c 'sleep 30 &
echo CHILD_PID=$!
' > /tmp/bp_ap03.out 2>&1
cat /tmp/bp_ap03.out
child_pid=$(grep -oE 'CHILD_PID=[0-9]+' /tmp/bp_ap03.out | cut -d= -f2)

echo
echo "Parent shell exited but child $child_pid is still alive:"
if ps -p "$child_pid" >/dev/null 2>&1; then
  echo "  ps: yes, $child_pid is running"
  if command -v lsof >/dev/null 2>&1; then
    echo "  FDs it still holds (note inherited pipe to the original stdout):"
    lsof -p "$child_pid" 2>/dev/null | awk 'NR==1 || /PIPE|REG/ {print "    " $0}' | head -n 8
  fi
  kill "$child_pid" 2>/dev/null || true
else
  echo "  ps: already gone"
fi
rm -f /tmp/bp_ap03.out
echo
echo "Correct form (G2): nohup CMD > LOG 2>&1 < /dev/null & echo \$! > PIDFILE"
