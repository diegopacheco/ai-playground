#!/usr/bin/env bash
# Anti-pattern: 'kill $PID' on a parent whose children are the real workers
# Source: design-doc.md §2.3, countered by G3
# Why it breaks: Surefire-forked JVMs, Django auto-reloader children,
# sbt server children all survive 'kill PID'. They keep ports bound and
# log files open. stop.orphans > 0 is the empirical signal.

set -euo pipefail

echo "ANTI-PATTERN: kill by PID (not by process group)"
echo "WHY: parent dies, children survive, ports stay bound, logs stay open."
echo
echo "DEMO: spawn a parent that forks a child; kill only the parent."

parent_script=$(mktemp)
cat > "$parent_script" <<'EOF'
sleep 30 &
child=$!
echo $child > /tmp/bp_ap06.child
sleep 30
EOF

bash "$parent_script" >/dev/null 2>&1 &
parent=$!
sleep 1
child=$(cat /tmp/bp_ap06.child 2>/dev/null || echo 0)

echo "  parent=$parent child=$child"
kill "$parent" 2>/dev/null || true
sleep 1

if ps -p "$child" >/dev/null 2>&1; then
  echo "  RESULT: parent killed, child $child IS STILL ALIVE -- orphan!"
  pkill -P "$child" 2>/dev/null || true
  kill "$child" 2>/dev/null || true
else
  echo "  child also gone (unexpected on this OS)"
fi

rm -f "$parent_script" /tmp/bp_ap06.child
echo
echo "Correct form (G3): pkill -TERM -P \$PID; kill -TERM \$PID;"
echo "  bounded wait; then pkill -KILL -P \$PID; kill -KILL \$PID."
