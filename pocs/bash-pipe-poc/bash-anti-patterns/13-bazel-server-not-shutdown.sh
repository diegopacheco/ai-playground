#!/usr/bin/env bash
# Anti-pattern: leaving the Bazel server running between invocations
# Source: design-doc.md §2.2 + G5
# Why it breaks: 'bazel build' spawns a long-lived server that holds the
# workspace lock, sockets, and file descriptors. A second invocation can
# stall on the lock; from the Bash tool's perspective the call hangs.

set -euo pipefail

echo "ANTI-PATTERN: not running 'bazel shutdown' in stop.sh"
echo "WHY: Bazel server holds workspace lock + sockets + FDs across runs."
echo
echo "DEMO: simulate the lock-held condition with an atomic mkdir lock."

workspace=$(mktemp -d)
lockdir="$workspace/.bazel-server.lock"

if mkdir "$lockdir" 2>/dev/null; then
  echo $$ > "$lockdir/pid"
  echo "  fake bazel server acquired $lockdir (pid=$$)"
fi

if mkdir "$lockdir" 2>/dev/null; then
  echo "  second invocation acquired the lock (unexpected)"
else
  held_by=$(cat "$lockdir/pid" 2>/dev/null || echo unknown)
  echo "  RESULT: second 'bazel' invocation BLOCKED -- lock held by pid=$held_by"
  echo "  in the real tool this presents as a silent hang until the ceiling."
fi

rm -rf "$workspace"
echo
echo "Correct form (G5): stop.sh runs 'bazel shutdown' and uses --max_idle_secs=10."
