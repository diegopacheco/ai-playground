#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PID="$(pid_of report)"
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
fi
rm -f "$RUN_DIR/report.pid"
tries=0
while port_up "$REPORT"; do
  tries=$((tries + 1))
  if [ "$tries" -ge 10 ]; then
    echo "port $REPORT is still in use by another process" >&2
    exit 1
  fi
  sleep 1
done
echo "report server stopped"
