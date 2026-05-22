#!/usr/bin/env bash
# Anti-pattern: using mvnd (Maven Daemon) from a skill instead of plain mvn
# Source: design-doc.md §2.2 + G5 + Decision Log
# Why it breaks: mvnd keeps a long-lived JVM with open FDs across runs --
# exactly the FD-leak class that causes phantom-running. The same applies
# to the Gradle daemon and the sbt server.

set -euo pipefail

echo "ANTI-PATTERN: using mvnd / Gradle daemon / sbt server from a skill"
echo "WHY: daemons hold FDs and ports across invocations -> phantom-running"
echo "     and 'address already in use' on the next start."
echo
echo "DEMO: a daemon-like process that survives the parent and keeps FDs."

log=$(mktemp /tmp/bp_ap14.XXXXXX.log)
daemon_pid_file=$(mktemp /tmp/bp_ap14.XXXXXX.pid)

bash -c "
  ( sleep 30 ) > '$log' 2>&1 &
  echo \$! > '$daemon_pid_file'
" >/dev/null 2>&1

pid=$(cat "$daemon_pid_file")
echo "  spawned fake daemon pid=$pid, log=$log"
if ps -p "$pid" >/dev/null 2>&1; then
  echo "  parent shell already returned, daemon is still alive."
  if command -v lsof >/dev/null 2>&1; then
    held=$(lsof -p "$pid" 2>/dev/null | awk '/REG|PIPE/ {n++} END {print n+0}')
    echo "  open FDs the daemon is still holding: $held"
  fi
  kill "$pid" 2>/dev/null || true
fi

rm -f "$log" "$daemon_pid_file"
echo
echo "Correct form (G5): plain 'mvn' (no daemon). For sbt: -Dsbt.server.autostart=false."
echo "                  For Bazel: 'bazel shutdown' in stop.sh."
